"""
Построение центроидов для каждой детали
Из эмбеддингов → центроиды деталей
С поддержкой инкрементального обновления и правильной обработки путей
"""

import argparse
import numpy as np
from pathlib import Path
import json
import time
from datetime import datetime
from collections import defaultdict
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
        logging.FileHandler('./logs/build_centroids.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

def load_existing_centroids(centroids_dir: Path):
    """Загрузка существующих центроидов"""
    centroid_file = centroids_dir / "per_part.npy"
    names_file = centroids_dir / "part_names.npy"
    meta_file = centroids_dir / "metadata.json"
    
    if centroid_file.exists() and names_file.exists():
        try:
            centroids = np.load(centroid_file).astype(np.float32)
            part_names = np.load(names_file, allow_pickle=True)
            
            # Загрузка метаданных
            metadata = {}
            if meta_file.exists():
                with open(meta_file, 'r') as f:
                    metadata = json.load(f)
            
            logger.info(f" Загружено существующих центроидов: {len(centroids)}")
            return centroids, part_names, metadata, set(part_names.tolist())
        except Exception as e:
            logger.warning(f"  Ошибка загрузки существующих центроидов: {e}")
            return None, None, {}, set()
    else:
        logger.info(" Создание новых файлов центроидов...")
        return None, None, {}, set()

def remove_existing_centroids(centroids_dir: Path):
    """Удаление существующих центроидов при --force режиме"""
    centroid_file = centroids_dir / "per_part.npy"
    names_file = centroids_dir / "part_names.npy"
    meta_file = centroids_dir / "metadata.json"
    backup_files = []
    
    # Создаем бэкапы
    if centroid_file.exists():
        centroid_backup = centroids_dir / "per_part.npy.backup"
        if centroid_backup.exists():
            try:
                centroid_backup.unlink()
                print(f"  Удален старый бэкап {centroid_backup.name}")
            except Exception as e:
                print(f"  Ошибка удаления бэкапа {centroid_backup.name}")

        backup_path = centroids_dir / "per_part.npy.backup"
        try:
            centroid_file.rename(backup_path)
            backup_files.append(backup_path)
            logger.info(f" Создан бэкап центроидов: {backup_path.name}")
        except Exception as e:
            logger.warning(f"  Ошибка создания бэкапа {centroid_file.name}: {e}")
    
    if names_file.exists():
        names_backup = centroids_dir / "part_names.npy.backup"
        if names_backup.exists():
            try:
                names_backup.unlink()
                print(f"  Удален старый бэкап {names_backup.name}")
            except Exception as e:
                print(f"  Ошибка удаления бэкапа {names_backup.name}")

        backup_path = centroids_dir / "part_names.npy.backup"
        try:
            names_file.rename(backup_path)
            backup_files.append(backup_path)
            logger.info(f" Создан бэкап имен: {backup_path.name}")
        except Exception as e:
            logger.warning(f"  Ошибка создания бэкапа {names_file.name}: {e}")
    
    if meta_file.exists():
        meta_backup = centroids_dir / "metadata.json.backup"
        if meta_backup.exists():
            try:
                meta_backup.unlink()
                print(f"  Удален старый бэкап {meta_backup.name}")
            except Exception as e:
                print(f"  Ошибка удаления бэкапа {meta_backup.name}")

        backup_path = centroids_dir / "metadata.json.backup"
        try:
            meta_file.rename(backup_path)
            backup_files.append(backup_path)
            logger.info(f" Создан бэкап метаданных: {backup_path.name}")
        except Exception as e:
            logger.warning(f"  Ошибка создания бэкапа {meta_file.name}: {e}")
    
    return backup_files

def save_centroids(centroids_dir: Path, centroids: np.ndarray, part_names: np.ndarray, metadata: dict):
    """Сохранение центроидов с оптимизацией размера"""
    centroids_dir.mkdir(parents=True, exist_ok=True)
    
    centroid_file = centroids_dir / "per_part.npy"
    names_file = centroids_dir / "part_names.npy"
    meta_file = centroids_dir / "metadata.json"
    
    # Создаем бэкап существующих файлов
    if centroid_file.exists():
        centroid_file.rename(centroids_dir / "per_part.npy.backup")
    if names_file.exists():
        names_file.rename(centroids_dir / "part_names.npy.backup")
    
    # Оптимизация: float32 и убедимся в правильном типе
    centroids_opt = centroids.astype(np.float32)
    part_names_opt = np.array(part_names, dtype=object)
    
    # Сохраняем оптимизированные данные
    try:
        np.save(centroid_file, centroids_opt)
        np.save(names_file, part_names_opt)
        logger.info(f" Центроиды сохранены: {len(centroids_opt)} векторов")
    except Exception as e:
        logger.error(f" Ошибка сохранения центроидов: {e}")
        # Восстанавливаем бэкапы если были
        for backup_path in [centroids_dir / "per_part.npy.backup", centroids_dir / "part_names.npy.backup"]:
            if backup_path.exists():
                original_path = backup_path.with_suffix('')
                if not original_path.exists():
                    backup_path.rename(original_path)
        return False
    
    # Обновляем метаданные
    metadata['updated_at'] = datetime.now().isoformat()
    metadata['total_parts'] = len(centroids_opt)
    metadata['embedding_dim'] = centroids_opt.shape[1] if len(centroids_opt.shape) > 1 else 2048
    metadata['data_type'] = str(centroids_opt.dtype)
    
    try:
        with open(meta_file, 'w') as f:
            json.dump(metadata, f, indent=2, ensure_ascii=False)
        logger.info(" Метаданные сохранены")
    except Exception as e:
        logger.warning(f"  Ошибка сохранения метаданных: {e}")
    
    size_mb = (centroid_file.stat().st_size + names_file.stat().st_size) / (1024 * 1024)
    logger.info(f" Сохранено {len(centroids_opt)} центроидов ({size_mb:.2f} MB)")
    return True

def build_centroids(embeddings_dir: Path, out_dir: Path, update: bool = True, force: bool = False):
    """Построение центроидов для каждой детали"""
    logger.info(" Построение центроидов")
    logger.info(f" Директория эмбеддингов: {embeddings_dir}")
    logger.info(f" Директория центроидов: {out_dir}")
    logger.info(f" Режим: {'Инкрементальное обновление' if update and not force else 'Полная перезапись'}")
    
    start_time = time.time()
    
    # Обработка --force режима
    if force:
        logger.info("  Принудительная перезапись - удаление существующих центроидов...")
        backup_files = remove_existing_centroids(out_dir)
        if backup_files:
            logger.info(f" Созданы бэкапы: {len(backup_files)} файлов")
        # Устанавливаем update=False для полной перезаписи
        update = False
    
    # Загрузка существующих центроидов (если update=True)
    existing_centroids, existing_names, metadata, seen_parts = None, None, {}, set()
    if update:
        existing_centroids, existing_names, metadata, seen_parts = load_existing_centroids(out_dir)
        
        # Если загрузка не удалась из-за ошибок, предлагаем --force
        if existing_centroids is None and (out_dir / "per_part.npy").exists():
            logger.error(" Невозможно выполнить инкрементальное обновление из-за ошибок в существующих центроидах")
            logger.info(" Выполните команду с --force для принудительной перезаписи:")
            logger.info(f"   python {Path(__file__).name} --embeddings {embeddings_dir} --out {out_dir} --force")
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
    
    # Группировка по деталям
    logger.info(" Группировка по деталям...")
    part_embeddings = defaultdict(list)
    
    for emb, full_id in zip(embeddings, part_ids):
        try:
            # Извлекаем part_id из full_id (part_id/image_name.jpg)
            part_id = str(full_id).split('/')[0]  # Преобразуем в строку для безопасности
            
            # Если в режиме update и part_id уже есть, пропускаем
            if update and part_id in seen_parts:
                continue
                
            part_embeddings[part_id].append(emb)
        except Exception as e:
            logger.warning(f"  Ошибка обработки ID {full_id}: {e}")
            continue
    
    logger.info(f" Найдено {len(part_embeddings)} новых деталей для обработки")
    
    if len(part_embeddings) == 0:
        if update:
            logger.info(" Новых деталей нет для обработки")
        else:
            logger.warning(" Нет данных для построения центроидов")
        return True
    
    # Вычисление центроидов
    logger.info(" Вычисление центроидов...")
    new_centroids = []
    new_part_names = []
    
    for part_id, emb_list in part_embeddings.items():
        try:
            # Среднее по всем эмбеддингам этой детали
            centroid = np.mean(emb_list, axis=0).astype(np.float32)
            new_centroids.append(centroid)
            new_part_names.append(part_id)
        except Exception as e:
            logger.warning(f"  Ошибка вычисления центроида для {part_id}: {e}")
            continue
    
    if len(new_centroids) == 0:
        logger.error(" Не удалось вычислить ни одного центроида")
        return False
    
    new_centroids = np.array(new_centroids, dtype=np.float32)
    new_part_names = np.array(new_part_names, dtype=object)
    
    logger.info(f" Вычислено {len(new_centroids)} новых центроидов")
    
    # Объединение с существующими центроидами
    if existing_centroids is not None and existing_names is not None:
        try:
            final_centroids = np.concatenate([existing_centroids, new_centroids], axis=0)
            final_names = np.concatenate([existing_names, new_part_names], axis=0)
            logger.info(f" Всего центроидов после объединения: {len(final_centroids)}")
        except Exception as e:
            logger.error(f" Ошибка объединения центроидов: {e}")
            logger.info(" Рекомендуется выполнить принудительную перезапись с --force")
            return False
    else:
        final_centroids = new_centroids
        final_names = new_part_names
        logger.info(f" Новых центроидов: {len(final_centroids)}")
    
    # Сохранение центроидов
    success = save_centroids(out_dir, final_centroids, final_names, metadata)
    if not success:
        return False
    
    # Финальная статистика
    total_time = time.time() - start_time
    logger.info(f"\n РЕЗУЛЬТАТЫ")
    logger.info(f" Успешно обработано: {len(new_centroids)} центроидов")
    logger.info(f" Всего: {len(final_centroids)} центроидов")
    if len(new_centroids) > 0:
        logger.info(f"  Время обработки: {total_time:.2f} сек")
        logger.info(f" Производительность: {len(new_centroids)/total_time:.2f} центроидов/сек")
    logger.info(f" Сохранено в: {out_dir}")
    
    # Пример статистики
    embeddings_per_part = [len(emb_list) for emb_list in part_embeddings.values()]
    if embeddings_per_part:
        logger.info(f" Статистика по изображениям на деталь:")
        logger.info(f"   Среднее: {np.mean(embeddings_per_part):.1f}")
        logger.info(f"   Минимум: {np.min(embeddings_per_part)}")
        logger.info(f"   Максимум: {np.max(embeddings_per_part)}")
    
    return True

def main():
    parser = argparse.ArgumentParser(
        description="Построение центроидов деталей",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Примеры:
  %(prog)s --embeddings data/embeddings --out data/centroids
  %(prog)s --embeddings data/embeddings --out data/centroids --update
  %(prog)s --embeddings data/embeddings --out data/centroids --force
        """
    )
    
    parser.add_argument("--embeddings", type=Path, default=Path("data/embeddings"),
                       help="Директория с эмбеддингами")
    parser.add_argument("--out", type=Path, default=Path("data/centroids"),
                       help="Директория для сохранения центроидов")
    parser.add_argument("--update", action="store_true", default=True,
                       help="Инкрементальное обновление (по умолчанию)")
    parser.add_argument("--force", action="store_true",
                       help="Полная перезапись (удалить существующие центроиды)")
    
    args = parser.parse_args()
    
    # Если указан --force, отключаем update
    if args.force:
        args.update = False
    
    # Проверка существования директории эмбеддингов
    if not args.embeddings.exists():
        logger.error(f" Директория эмбеддингов не найдена: {args.embeddings}")
        return 1
    
    try:
        success = build_centroids(args.embeddings, args.out, args.update, args.force)
        if success:
            logger.info("\n Построение центроидов завершено!")
            return 0
        else:
            logger.error("\n Ошибка построения центроидов")
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
