#!/usr/bin/env python3
"""
Построение FAISS индексов для поиска
1. Индекс по всем изображениям
2. Индекс по центроидам деталей
"""

import argparse
import numpy as np
from pathlib import Path
import json
import time
from datetime import datetime
import sys

def build_indexes(embeddings_dir: Path, centroids_dir: Path, out_dir: Path):
    """Построение FAISS индексов"""
    print(f"🔄 Построение FAISS индексов")
    print(f"📂 Директория эмбеддингов: {embeddings_dir}")
    print(f"📂 Директория центроидов: {centroids_dir}")
    print(f"📂 Директория индексов: {out_dir}")
    
    start_time = time.time()
    
    # Попытка импорта FAISS
    try:
        import faiss
        print("✅ FAISS загружен")
    except ImportError:
        print("❌ FAISS не установлен")
        print("💡 Установите: pip install faiss-cpu")
        print("💡 Или для GPU: pip install faiss-gpu")
        return False
    except Exception as e:
        print(f"❌ Ошибка загрузки FAISS: {e}")
        return False
    
    # Загрузка эмбеддингов
    emb_file = embeddings_dir / "per_image.npy"
    ids_file = embeddings_dir / "part_ids.npy"
    
    if not emb_file.exists() or not ids_file.exists():
        print(f"❌ Эмбеддинги не найдены в {embeddings_dir}")
        return False
    
    print("🔄 Загрузка эмбеддингов...")
    try:
        embeddings = np.load(emb_file).astype(np.float32)
        part_ids = np.load(ids_file, allow_pickle=True)
        print(f"✅ Загружено {len(embeddings)} эмбеддингов")
    except Exception as e:
        print(f"❌ Ошибка загрузки эмбеддингов: {e}")
        return False
    
    # Проверка соответствия размеров
    if len(embeddings) != len(part_ids):
        print(f"❌ Несоответствие размеров: {len(embeddings)} эмбеддингов vs {len(part_ids)} ID")
        return False
    
    # Загрузка центроидов
    centroid_file = centroids_dir / "per_part.npy"
    names_file = centroids_dir / "part_names.npy"
    
    if not centroid_file.exists() or not names_file.exists():
        print(f"❌ Центроиды не найдены в {centroids_dir}")
        return False
    
    print("🔄 Загрузка центроидов...")
    try:
        centroids = np.load(centroid_file).astype(np.float32)
        part_names = np.load(names_file, allow_pickle=True)
        print(f"✅ Загружено {len(centroids)} центроидов")
    except Exception as e:
        print(f"❌ Ошибка загрузки центроидов: {e}")
        return False
    
    # Проверка размерности
    if len(embeddings) > 0 and len(centroids) > 0:
        if embeddings.shape[1] != centroids.shape[1]:
            print(f"❌ Несоответствие размерности: {embeddings.shape[1]} vs {centroids.shape[1]}")
            return False
    
    dim = embeddings.shape[1] if len(embeddings) > 0 else (centroids.shape[1] if len(centroids) > 0 else 2048)
    print(f"📊 Размерность эмбеддингов: {dim}")
    
    # Построение индекса по всем изображениям
    print("🔄 Построение индекса по изображениям...")
    
    try:
        # Используем более эффективный индекс для больших данных
        if len(embeddings) > 100000:
            # IVF (Inverted File) индекс для больших наборов
            nlist = min(10000, max(100, len(embeddings) // 100))  # Количество кластеров
            print(f"   🎓 Создание IVF индекса с {nlist} кластерами...")
            quantizer = faiss.IndexFlatL2(dim)
            index_images = faiss.IndexIVFFlat(quantizer, dim, nlist, faiss.METRIC_L2)
            
            # Обучение индекса (на подмножестве данных для скорости)
            train_size = min(100000, len(embeddings))
            print(f"   🎓 Обучение индекса на {train_size} векторах...")
            index_images.train(embeddings[:train_size])
        else:
            # Простой Flat индекс для небольших наборов
            print("   🎓 Создание Flat индекса...")
            index_images = faiss.IndexFlatL2(dim)
        
        # Добавление эмбеддингов в индекс
        print(f"   ➕ Добавление {len(embeddings)} векторов в индекс...")
        index_images.add(embeddings)
        print(f"   ✅ Индекс по изображениям построен ({index_images.ntotal} векторов)")
        
    except Exception as e:
        print(f"❌ Ошибка построения индекса изображений: {e}")
        return False
    
    # Построение индекса по центроидам
    print("🔄 Построение индекса по центроидам...")
    
    try:
        if len(centroids) > 0:
            index_centroids = faiss.IndexFlatL2(dim)
            index_centroids.add(centroids)
            print(f"✅ Индекс по центроидам построен ({index_centroids.ntotal} векторов)")
        else:
            index_centroids = None
            print("⚠️  Нет центроидов для построения индекса")
    except Exception as e:
        print(f"❌ Ошибка построения индекса центроидов: {e}")
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
        images_index_file.rename(backup_path)
        backup_files.append(backup_path)
    if centroids_index_file.exists() and index_centroids is not None:
        backup_path = out_dir / "centroid_index.faiss.backup"
        centroids_index_file.rename(backup_path)
        backup_files.append(backup_path)
    
    # Сохранение индексов
    try:
        faiss.write_index(index_images, str(images_index_file))
        print(f"✅ Индекс изображений сохранен: {images_index_file}")
        
        if index_centroids is not None:
            faiss.write_index(index_centroids, str(centroids_index_file))
            print(f"✅ Индекс центроидов сохранен: {centroids_index_file}")
        
    except Exception as e:
        print(f"❌ Ошибка сохранения индексов: {e}")
        # Восстанавливаем бэкапы если были
        for backup_path in backup_files:
            original_path = backup_path.with_suffix('')
            if backup_path.exists() and not original_path.exists():
                backup_path.rename(original_path)
        return False
    
    # Сохранение ID для маппинга
    try:
        ids_mapping_file = out_dir / "image_ids.npy"
        names_mapping_file = out_dir / "centroid_names.npy"
        
        np.save(ids_mapping_file, part_ids)
        np.save(names_mapping_file, part_names)
        print(f"✅ Маппинги ID сохранены")
    except Exception as e:
        print(f"⚠️  Ошибка сохранения маппингов: {e}")
    
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
        print(f"✅ Метаданные сохранены")
    except Exception as e:
        print(f"⚠️  Ошибка сохранения метаданных: {e}")
    
    # Статистика
    total_time = time.time() - start_time
    
    print(f"\n🏁 РЕЗУЛЬТАТЫ")
    images_size_mb = images_index_file.stat().st_size / (1024 * 1024)
    print(f"✅ Индекс изображений: {images_size_mb:.2f} MB ({index_images.ntotal} векторов)")
    
    if index_centroids is not None:
        centroids_size_mb = centroids_index_file.stat().st_size / (1024 * 1024)
        print(f"✅ Индекс центроидов: {centroids_size_mb:.2f} MB ({index_centroids.ntotal} векторов)")
    
    print(f"⏱️  Время обработки: {total_time:.2f} сек")
    print(f"💾 Сохранено в: {out_dir}")
    
    return True

def main():
    parser = argparse.ArgumentParser(
        description="Построение FAISS индексов",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Примеры:
  %(prog)s --embeddings data/embeddings --centroids data/centroids --out data/indexes
        """
    )
    
    parser.add_argument("--embeddings", type=Path, default=Path("data/embeddings"),
                       help="Директория с эмбеддингами")
    parser.add_argument("--centroids", type=Path, default=Path("data/centroids"),
                       help="Директория с центроидами")
    parser.add_argument("--out", type=Path, default=Path("data/indexes"),
                       help="Директория для сохранения индексов")
    
    args = parser.parse_args()
    
    try:
        success = build_indexes(args.embeddings, args.centroids, args.out)
        if success:
            print("\n🎉 Построение индексов завершено!")
            return 0
        else:
            print("\n❌ Ошибка построения индексов")
            return 1
    except KeyboardInterrupt:
        print("\n⚠️  Обработка прервана пользователем")
        return 1
    except Exception as e:
        print(f"\n❌ Критическая ошибка: {e}")
        import traceback
        traceback.print_exc()
        return 1

if __name__ == "__main__":
    sys.exit(main())