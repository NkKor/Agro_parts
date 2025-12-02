"""
Построение FAISS индексов для поиска
1. Индекс по всем изображениям
2. Индекс по центроидам деталей
С поддержкой инкрементального обновления и правильной обработки ошибок
"""

import argparse
import numpy as np
from pathlib import Path
import json
import time
from datetime import datetime
import sys
import logging
import os


# --- Настройка путей ---
current_file = Path(__file__).resolve()
project_root = current_file.parent.parent
utils_path = project_root / "utils"
src_path = project_root / "src"

# --- Настройка TORCH_HOME ---
torch_home = project_root / 'data' / 'models'
torch_home.mkdir(parents=True, exist_ok=True)
os.environ['TORCH_HOME'] = str(torch_home)

# Добавляем пути в sys.path для корректных импортов
sys.path.insert(0, str(project_root))
sys.path.insert(0, str(utils_path))
sys.path.insert(0, str(src_path))

# --- Настройка логирования ---
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s %(name)-12s %(levelname)-8s %(message)s',
    datefmt='%m-%d %H:%M',
    handlers=[
        logging.FileHandler('./logs/build_index.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

def load_existing_indexes(indexes_dir: Path):
    """Загрузка существующих индексов"""
    images_index_file = indexes_dir / "image_index.faiss"
    centroids_index_file = indexes_dir / "centroid_index.faiss"
    meta_file = indexes_dir / "metadata.json"
    
    if images_index_file.exists() and meta_file.exists():
        try:
            import faiss
            images_index = faiss.read_index(str(images_index_file))
            
            centroids_index = None
            if centroids_index_file.exists():
                try:
                    centroids_index = faiss.read_index(str(centroids_index_file))
                    logger.info(f" Загружены существующие индексы: {images_index.ntotal} изображений, {centroids_index.ntotal if centroids_index else 0} центроидов")
                except Exception as e:
                    logger.warning(f"  Ошибка загрузки индекса центроидов: {e}")
            
            # Загрузка метаданных
            metadata = {}
            if meta_file.exists():
                with open(meta_file, 'r') as f:
                    metadata = json.load(f)
            
            return images_index, centroids_index, metadata
        except Exception as e:
            logger.warning(f"  Ошибка загрузки существующих индексов: {e}")
            return None, None, {}
    else:
        logger.info(" Создание новых индексов...")
        return None, None, {}

def remove_existing_indexes(indexes_dir: Path):
    """Удаление существующих индексов при --force режиме"""
    files_to_remove = [
        "image_index.faiss",
        "centroid_index.faiss", 
        "metadata.json",
        "image_ids.npy",
        "centroid_names.npy"
    ]
    
    backup_files = []
    for filename in files_to_remove:
        file_path = indexes_dir / filename
        if file_path.exists():
            file_backup = indexes_dir / f"{filename}.backup"
            if file_backup.exists():
                try:
                    file_backup.unlink()
                except Exception as e:
                    print(f"  Ошибка удаления старого бэкапа {file_backup}: {e}")
                    logger.warning(f"  Ошибка удаления старого бэкапа {file_backup}: {e}")
            backup_path = indexes_dir / f"{filename}.backup"
            try:
                file_path.rename(backup_path)
                backup_files.append(backup_path)
                logger.info(f" Создан бэкап: {backup_path.name}")
            except Exception as e:
                logger.warning(f"  Ошибка создания бэкапа {filename}: {e}")
    
    return backup_files

def build_indexes(embeddings_dir: Path, centroids_dir: Path, out_dir: Path, force: bool = False):
    """Построение FAISS индексов с поддержкой инкрементального обновления"""
    logger.info(" Построение FAISS индексов")
    logger.info(f" Директория эмбеддингов: {embeddings_dir}")
    logger.info(f" Директория центроидов: {centroids_dir}")
    logger.info(f" Директория индексов: {out_dir}")
    logger.info(f" Режим: {'Полная перезапись' if force else 'Инкрементальное обновление'}")
    
    start_time = time.time()
    
    # Обработка --force режима
    if force:
        logger.info("  Принудительная перезапись - удаление существующих индексов...")
        backup_files = remove_existing_indexes(out_dir)
        if backup_files:
            logger.info(f" Созданы бэкапы: {len(backup_files)} файлов")
    
    # Попытка импорта FAISS
    try:
        import faiss
        logger.info(" FAISS загружен")
    except ImportError:
        logger.error(" FAISS не установлен")
        logger.info(" Установите: pip install faiss-cpu")
        logger.info(" Или для GPU: pip install faiss-gpu")
        return False
    except Exception as e:
        logger.error(f" Ошибка загрузки FAISS: {e}")
        return False
    
    # Загрузка существующих индексов (если не force)
    existing_images_index, existing_centroids_index, metadata = None, None, {}
    if not force:
        existing_images_index, existing_centroids_index, metadata = load_existing_indexes(out_dir)
        
        # Если загрузка не удалась из-за ошибок, предлагаем --force
        images_index_file = out_dir / "image_index.faiss"
        if existing_images_index is None and images_index_file.exists():
            logger.error(" Невозможно выполнить инкрементальное обновление из-за ошибок в существующих индексах")
            logger.info(" Выполните команду с --force для принудительной перезаписи:")
            logger.info(f"   python {Path(__file__).name} --embeddings {embeddings_dir} --centroids {centroids_dir} --out {out_dir} --force")
            return False
    
    # Загрузка эмбеддингов
    emb_file = embeddings_dir / "per_image.npy"
    ids_file = embeddings_dir / "part_ids.npy"
    
    if not emb_file.exists() or not ids_file.exists():
        logger.error(f" Эмбеддинги не найдены в {embeddings_dir}")
        return False
    
    logger.info(" Загрузка эмбеддингов...")
    try:
        embeddings = np.load(emb_file).astype(np.float32)
        part_ids = np.load(ids_file, allow_pickle=True)
        logger.info(f" Загружено {len(embeddings)} эмбеддингов")
    except Exception as e:
        logger.error(f" Ошибка загрузки эмбеддингов: {e}")
        return False
    
    # Проверка соответствия размеров
    if len(embeddings) != len(part_ids):
        logger.error(f" Несоответствие размеров: {len(embeddings)} эмбеддингов vs {len(part_ids)} ID")
        return False
    
    # Загрузка центроидов
    centroid_file = centroids_dir / "per_part.npy"
    names_file = centroids_dir / "part_names.npy"
    
    if not centroid_file.exists() or not names_file.exists():
        logger.error(f" Центроиды не найдены в {centroids_dir}")
        return False
    
    logger.info(" Загрузка центроидов...")
    try:
        centroids = np.load(centroid_file).astype(np.float32)
        part_names = np.load(names_file, allow_pickle=True)
        logger.info(f" Загружено {len(centroids)} центроидов")
    except Exception as e:
        logger.error(f" Ошибка загрузки центроидов: {e}")
        return False
    
    # Проверка размерности
    if len(embeddings) > 0 and len(centroids) > 0:
        if embeddings.shape[1] != centroids.shape[1]:
            logger.error(f" Несоответствие размерности: {embeddings.shape[1]} vs {centroids.shape[1]}")
            return False
    
    dim = embeddings.shape[1] if len(embeddings) > 0 else (centroids.shape[1] if len(centroids) > 0 else 2048)
    logger.info(f" Размерность эмбеддингов: {dim}")
    
    # Построение индекса по всем изображениям
    logger.info(" Построение индекса по изображениям...")
    
    try:
        # Используем более эффективный индекс для больших данных
        if len(embeddings) > 100000:
            # IVF (Inverted File) индекс для больших наборов
            nlist = min(10000, max(100, len(embeddings) // 100))  # Количество кластеров
            logger.info(f"    Создание IVF индекса с {nlist} кластерами...")
            quantizer = faiss.IndexFlatL2(dim)
            index_images = faiss.IndexIVFFlat(quantizer, dim, nlist, faiss.METRIC_L2)
            
            # Обучение индекса (на подмножестве данных для скорости)
            train_size = min(100000, len(embeddings))
            logger.info(f"    Обучение индекса на {train_size} векторах...")
            index_images.train(embeddings[:train_size].astype(np.float32))
        else:
            # Простой Flat индекс для небольших наборов
            logger.info("    Создание Flat индекса...")
            index_images = faiss.IndexFlatL2(dim)
        
        # Добавление эмбеддингов в индекс
        logger.info(f"    Добавление {len(embeddings)} векторов в индекс...")
        index_images.add(embeddings.astype(np.float32))
        logger.info(f"    Индекс по изображениям построен ({index_images.ntotal} векторов)")
        
    except Exception as e:
        logger.error(f" Ошибка построения индекса изображений: {e}")
        return False
    
    # Построение индекса по центроидам
    logger.info(" Построение индекса по центроидам...")
    
    try:
        if len(centroids) > 0:
            index_centroids = faiss.IndexFlatL2(dim)
            index_centroids.add(centroids.astype(np.float32))
            logger.info(f" Индекс по центроидам построен ({index_centroids.ntotal} векторов)")
        else:
            index_centroids = None
            logger.info("  Нет центроидов для построения индекса")
    except Exception as e:
        logger.error(f" Ошибка построения индекса центроидов: {e}")
        index_centroids = None
    
    # Сохранение индексов
    out_dir.mkdir(parents=True, exist_ok=True)
    
    images_index_file = out_dir / "image_index.faiss"
    centroids_index_file = out_dir / "centroid_index.faiss"
    meta_file = out_dir / "metadata.json"
    
    # Бэкап существующих файлов
    backup_files = []
    if images_index_file.exists():
        backup_path = out_dir / "image_index.faiss.backup"
        try:
            images_index_file.rename(backup_path)
            backup_files.append(backup_path)
            logger.info(f" Создан бэкап индекса изображений: {backup_path.name}")
        except Exception as e:
            logger.warning(f"  Ошибка создания бэкапа индекса изображений: {e}")
    
    if centroids_index_file.exists() and index_centroids is not None:
        backup_path = out_dir / "centroid_index.faiss.backup"
        try:
            centroids_index_file.rename(backup_path)
            backup_files.append(backup_path)
            logger.info(f" Создан бэкап индекса центроидов: {backup_path.name}")
        except Exception as e:
            logger.warning(f"  Ошибка создания бэкапа индекса центроидов: {e}")
    
    # Сохранение индексов
    try:
        faiss.write_index(index_images, str(images_index_file))
        logger.info(f" Индекс изображений сохранен: {images_index_file}")
        
        if index_centroids is not None:
            faiss.write_index(index_centroids, str(centroids_index_file))
            logger.info(f" Индекс центроидов сохранен: {centroids_index_file}")
        
    except Exception as e:
        logger.error(f" Ошибка сохранения индексов: {e}")
        # Восстанавливаем бэкапы если были
        for backup_path in backup_files:
            original_path = backup_path.with_suffix('')
            if backup_path.exists() and not original_path.exists():
                try:
                    backup_path.rename(original_path)
                    logger.info(f" Восстановлен бэкап: {original_path.name}")
                except Exception as restore_err:
                    logger.error(f" Ошибка восстановления бэкапа {backup_path.name}: {restore_err}")
        return False
    
    # Сохранение ID для маппинга
    try:
        ids_mapping_file = out_dir / "image_ids.npy"
        names_mapping_file = out_dir / "centroid_names.npy"
        
        np.save(ids_mapping_file, np.array(part_ids, dtype=object))
        np.save(names_mapping_file, np.array(part_names, dtype=object))
        logger.info(f" Маппинги ID сохранены")
    except Exception as e:
        logger.error(f" Ошибка сохранения маппингов: {e}")
    
    # Метаданные
    metadata = {
        'created_at': datetime.now().isoformat(),
        'total_images': len(embeddings),
        'total_parts': len(centroids),
        'embedding_dim': dim,
        'index_type_images': 'IVFFlat' if len(embeddings) > 100000 else 'Flat',
        'index_type_centroids': 'Flat' if index_centroids is not None else 'None',
        'source_embeddings': str(embeddings_dir),
        'source_centroids': str(centroids_dir),
        'processing_time': time.time() - start_time
    }
    
    try:
        with open(meta_file, 'w') as f:
            json.dump(metadata, f, indent=2, ensure_ascii=False)
        logger.info(f" Метаданные сохранены")
    except Exception as e:
        logger.error(f" Ошибка сохранения метаданных: {e}")
    
    # Статистика
    total_time = time.time() - start_time
    
    logger.info(f"\n РЕЗУЛЬТАТЫ")
    images_size_mb = images_index_file.stat().st_size / (1024 * 1024)
    logger.info(f" Индекс изображений: {images_size_mb:.2f} MB ({index_images.ntotal} векторов)")
    
    if index_centroids is not None:
        centroids_size_mb = centroids_index_file.stat().st_size / (1024 * 1024)
        logger.info(f" Индекс центроидов: {centroids_size_mb:.2f} MB ({index_centroids.ntotal} векторов)")
    
    logger.info(f"  Время обработки: {total_time:.2f} сек")
    logger.info(f" Сохранено в: {out_dir}")
    
    # Производительность
    if len(embeddings) > 0:
        vectors_per_second = len(embeddings) / total_time
        logger.info(f" Производительность: {vectors_per_second:.2f} векторов/сек")
    
    return True

def main():
    parser = argparse.ArgumentParser(
        description="Построение FAISS индексов",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Примеры:
  %(prog)s --embeddings data/embeddings --centroids data/centroids --out data/indexes
  %(prog)s --embeddings data/embeddings --centroids data/centroids --out data/indexes --force
        """
    )
    
    parser.add_argument("--embeddings", type=Path, default=Path("data/embeddings"),
                       help="Директория с эмбеддингами")
    parser.add_argument("--centroids", type=Path, default=Path("data/centroids"),
                       help="Директория с центроидами")
    parser.add_argument("--out", type=Path, default=Path("data/indexes"),
                       help="Директория для сохранения индексов")
    parser.add_argument("--force", action="store_true",
                       help="Полная перезапись (удалить существующие индексы)")
    
    args = parser.parse_args()
    
    # Проверка существования директорий
    if not args.embeddings.exists():
        logger.error(f" Директория эмбеддингов не найдена: {args.embeddings}")
        return 1
    
    if not args.centroids.exists():
        logger.error(f" Директория центроидов не найдена: {args.centroids}")
        return 1
    
    try:
        success = build_indexes(args.embeddings, args.centroids, args.out, args.force)
        if success:
            logger.info("\n Построение индексов завершено!")
            return 0
        else:
            logger.error("\n Ошибка построения индексов")
            return 1
    except KeyboardInterrupt:
        logger.info("\n  Обработка прервана пользователем")
        return 1
    except Exception as e:
        logger.error(f"\n Критическая ошибка: {e}")
        import traceback
        traceback.print_exc()
        return 1

if __name__ == "__main__":
    sys.exit(main())
