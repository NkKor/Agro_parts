#!/usr/bin/env python3
"""
Построение центроидов для каждой детали
Из эмбеддингов → центроиды деталей
"""

import argparse
import numpy as np
from pathlib import Path
import json
import time
from datetime import datetime
from collections import defaultdict
import sys

def build_centroids(embeddings_dir: Path, out_dir: Path):
    """Построение центроидов для каждой детали"""
    print(f"🔄 Построение центроидов")
    print(f"📂 Директория эмбеддингов: {embeddings_dir}")
    print(f"📂 Директория центроидов: {out_dir}")
    
    start_time = time.time()
    
    # Загрузка эмбеддингов
    emb_file = embeddings_dir / "per_image.npy"
    ids_file = embeddings_dir / "part_ids.npy"
    
    if not emb_file.exists() or not ids_file.exists():
        print(f"❌ Эмбеддинги не найдены в {embeddings_dir}")
        return False
    
    print("🔄 Загрузка эмбеддингов...")
    try:
        embeddings = np.load(emb_file)
        part_ids = np.load(ids_file, allow_pickle=True)
        print(f"✅ Загружено {len(embeddings)} эмбеддингов")
    except Exception as e:
        print(f"❌ Ошибка загрузки эмбеддингов: {e}")
        return False
    
    # Проверка соответствия размеров
    if len(embeddings) != len(part_ids):
        print(f"❌ Несоответствие размеров: {len(embeddings)} эмбеддингов vs {len(part_ids)} ID")
        return False
    
    # Группировка по деталям
    print("🔄 Группировка по деталям...")
    part_embeddings = defaultdict(list)
    
    for emb, full_id in zip(embeddings, part_ids):
        # Извлекаем part_id из full_id (part_id/image_name.jpg)
        try:
            part_id = full_id.split('/')[0]
            part_embeddings[part_id].append(emb)
        except Exception as e:
            print(f"⚠️  Ошибка обработки ID {full_id}: {e}")
            continue
    
    print(f"📊 Найдено {len(part_embeddings)} уникальных деталей")
    
    if len(part_embeddings) == 0:
        print("❌ Нет данных для построения центроидов")
        return False
    
    # Вычисление центроидов
    print("🔄 Вычисление центроидов...")
    centroids = []
    part_names = []
    
    for part_id, emb_list in part_embeddings.items():
        try:
            # Среднее по всем эмбеддингам этой детали
            centroid = np.mean(emb_list, axis=0).astype(np.float32)
            centroids.append(centroid)
            part_names.append(part_id)
        except Exception as e:
            print(f"⚠️  Ошибка вычисления центроида для {part_id}: {e}")
            continue
    
    if len(centroids) == 0:
        print("❌ Не удалось вычислить ни одного центроида")
        return False
    
    centroids = np.array(centroids, dtype=np.float32)
    part_names = np.array(part_names, dtype=object)
    
    print(f"✅ Вычислено {len(centroids)} центроидов")
    
    # Сохранение центроидов
    out_dir.mkdir(parents=True, exist_ok=True)
    
    centroid_file = out_dir / "per_part.npy"
    names_file = out_dir / "part_names.npy"
    meta_file = out_dir / "metadata.json"
    
    # Бэкап существующих файлов
    backup_files = []
    if centroid_file.exists():
        backup_path = out_dir / "per_part.npy.backup"
        centroid_file.rename(backup_path)
        backup_files.append(backup_path)
    if names_file.exists():
        backup_path = out_dir / "part_names.npy.backup"
        names_file.rename(backup_path)
        backup_files.append(backup_path)
    
    # Сохранение
    try:
        np.save(centroid_file, centroids)
        np.save(names_file, part_names)
        print(f"✅ Центроиды сохранены")
    except Exception as e:
        print(f"❌ Ошибка сохранения центроидов: {e}")
        # Восстанавливаем бэкапы если были
        for backup_path in backup_files:
            original_path = backup_path.with_suffix('')
            if backup_path.exists() and not original_path.exists():
                backup_path.rename(original_path)
        return False
    
    # Метаданные
    metadata = {
        'created_at': datetime.now().isoformat(),
        'total_parts': len(centroids),
        'embedding_dim': centroids.shape[1] if len(centroids.shape) > 1 else (2048 if len(centroids) > 0 else 0),
        'data_type': str(centroids.dtype),
        'source_embeddings': str(embeddings_dir),
        'processing_time': time.time() - start_time
    }
    
    try:
        with open(meta_file, 'w') as f:
            json.dump(metadata, f, indent=2, ensure_ascii=False)
        print(f"✅ Метаданные сохранены")
    except Exception as e:
        print(f"⚠️  Ошибка сохранения метаданных: {e}")
    
    # Статистика
    size_mb = (centroid_file.stat().st_size + names_file.stat().st_size) / (1024 * 1024)
    total_time = time.time() - start_time
    
    print(f"\n🏁 РЕЗУЛЬТАТЫ")
    print(f"✅ Сохранено {len(centroids)} центроидов ({size_mb:.2f} MB)")
    print(f"⏱️  Время обработки: {total_time:.2f} сек")
    print(f"💾 Сохранено в: {out_dir}")
    
    # Пример статистики
    embeddings_per_part = [len(emb_list) for emb_list in part_embeddings.values()]
    if embeddings_per_part:
        print(f"📊 Статистика по изображениям на деталь:")
        print(f"   Среднее: {np.mean(embeddings_per_part):.1f}")
        print(f"   Минимум: {np.min(embeddings_per_part)}")
        print(f"   Максимум: {np.max(embeddings_per_part)}")
    
    return True

def main():
    parser = argparse.ArgumentParser(
        description="Построение центроидов деталей",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Примеры:
  %(prog)s --embeddings data/embeddings --out data/centroids
        """
    )
    
    parser.add_argument("--embeddings", type=Path, default=Path("data/embeddings"),
                       help="Директория с эмбеддингами")
    parser.add_argument("--out", type=Path, default=Path("data/centroids"),
                       help="Директория для сохранения центроидов")
    
    args = parser.parse_args()
    
    try:
        success = build_centroids(args.embeddings, args.out)
        if success:
            print("\n🎉 Построение центроидов завершено!")
            return 0
        else:
            print("\n❌ Ошибка построения центроидов")
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