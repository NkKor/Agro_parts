"""
Извлечение эмбеддингов в формате двух .npy файлов
С поддержкой инкрементального обновления и оптимизации размера
"""

import argparse
import torch
import numpy as np
from pathlib import Path
import cv2 as cv
import torchvision.transforms as T
from tqdm import tqdm
import json
import time
from datetime import datetime
import sys
import hashlib

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

# --- Импорт модулей ---
try:
    from src.models.encoder import ResNet50Encoder
    from utils.utils_cv import find_largest_foreground_bbox, pad_bbox, center_square_crop, resize_high_quality
    import utils.config as config
except ImportError:
    try:
        from models.encoder import ResNet50Encoder
        from utils_cv import find_largest_foreground_bbox, pad_bbox, center_square_crop, resize_high_quality
        import config as config
    except ImportError:
        try:
            sys.path.append(str(Path(__file__).parent))
            sys.path.append(str(Path(__file__).parent.parent))
            sys.path.append(str(Path(__file__).parent.parent / "utils"))
            
            from models.encoder import ResNet50Encoder
            from utils_cv import find_largest_foreground_bbox, pad_bbox, center_square_crop, resize_high_quality
            import config as config
        except ImportError as e:
            print(f" Ошибка импорта: {e}")
            print(" Проверьте структуру проекта:")
            print("   project/")
            print("   ├── src/")
            print("   │   ├── models/encoder.py")
            print("   │   └── utils_cv.py")
            print("   utils/")
            print("   └── config.py")
            sys.exit(1)

# --- Трансформации ---
transform = T.Compose([
    T.ToTensor(),
    T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

def get_device(device_str: str = "auto") -> str:
    """Определение устройства для вычислений"""
    if device_str == "auto":
        if torch.cuda.is_available():
            try:
                x = torch.zeros(1).cuda()
                del x
                print(" CUDA доступна и поддерживается")
                return "cuda"
            except Exception as e:
                print(f"  CUDA доступна, но PyTorch не скомпилирован с CUDA: {e}")
                print(" Используется CPU")
        elif torch.backends.mps.is_available() and torch.backends.mps.is_built():
            print(" Используется MPS (Apple Silicon)")
            return "mps"
        else:
            print(" Используется CPU")
            return "cpu"
    print(f" Используется {device_str}")
    return device_str

@torch.no_grad()
def embed_image(img_path, model, device):
    """Извлечение эмбеддинга из изображения с обходом проблемы кириллицы"""
    try:
        # Чтение изображения через numpy для обхода проблемы с кириллицей
        with open(img_path, 'rb') as f:
            file_bytes = np.asarray(bytearray(f.read()), dtype=np.uint8)
            img_bgr = cv.imdecode(file_bytes, cv.IMREAD_COLOR)
        
        if img_bgr is None:
            return None
        
        # Предобработка изображения (соответствует preprocess.py)
        bbox = find_largest_foreground_bbox(img_bgr, min_area_ratio=getattr(config, 'MIN_OBJ_AREA', 0.01))
        if bbox is not None:
            bbox = pad_bbox(bbox, img_bgr.shape, pad_ratio=getattr(config, 'PAD_RATIO', 0.1))
            x1, y1, x2, y2 = bbox
            crop = img_bgr[y1:y2, x1:x2]
        else:
            crop = center_square_crop(img_bgr)
        
        crop = resize_high_quality(crop, getattr(config, 'TARGET_SIZE', 384))
        
        # Конвертация в RGB и применение трансформаций
        import torchvision.transforms.functional as F
        img_rgb = cv.cvtColor(crop, cv.COLOR_BGR2RGB)
        pil = F.to_pil_image(img_rgb)
        x = transform(pil).unsqueeze(0).to(device)
        
        # Извлечение эмбеддинга
        emb = model(x).cpu().numpy()
        # ensure float32 и нормализация
        emb = emb.astype('float32')
        emb = emb / (np.linalg.norm(emb, axis=1, keepdims=True) + 1e-12)
        return emb[0]  # Возвращаем вектор (2048,)
        
    except Exception as e:
        print(f" Ошибка обработки изображения {img_path}: {e}")
        return None

def load_existing_data(embeddings_dir: Path):
    """Загрузка существующих эмбеддингов"""
    emb_file = embeddings_dir / "per_image.npy"
    ids_file = embeddings_dir / "part_ids.npy"
    meta_file = embeddings_dir / "metadata.json"
    
    if emb_file.exists() and ids_file.exists():
        try:
            embeddings = np.load(emb_file).astype(np.float32)
            part_ids = np.load(ids_file, allow_pickle=True)
            
            # Загрузка метаданных
            metadata = {}
            if meta_file.exists():
                with open(meta_file, 'r') as f:
                    metadata = json.load(f)
            
            print(f" Загружено существующих данных: {len(embeddings)} эмбеддингов")
            return embeddings, part_ids, metadata, set(part_ids.tolist())
        except Exception as e:
            print(f"  Ошибка загрузки существующих данных: {e}")
            print("  Рекомендуется выполнить принудительную перезапись с --force")
            return None, None, {}, set()
    else:
        print(" Создание новых файлов...")
        return None, None, {}, set()

def remove_existing_files(embeddings_dir: Path):
    """Удаление существующих файлов при --force режиме"""
    emb_file = embeddings_dir / "per_image.npy"
    ids_file = embeddings_dir / "part_ids.npy"
    meta_file = embeddings_dir / "metadata.json"
    backup_files = []
    
    # Создаем бэкапы
    if emb_file.exists():
        emb_backup = embeddings_dir / "per_image.npy.backup"
        if emb_backup.exists():
            try:
                emb_backup.unlink()
                print(f"  Удален старый бэкап {emb_backup.name}")
            except Exception as e:
                print(f"  Ошибка удаления бэкапа {emb_backup.name}")

        backup_path = embeddings_dir / "per_image.npy.backup"
        try:
            emb_file.rename(backup_path)
            backup_files.append(backup_path)
            print(f" Создан бэкап: {backup_path.name}")
        except Exception as e:
            print(f"  Ошибка создания бэкапа {emb_file.name}: {e}")
    
    if ids_file.exists():
        ids_backup = embeddings_dir / "part_ids.npy.backup"
        if ids_backup.exists():
            try:
                ids_backup.unlink()
                print(f"  Удален старый бэкап {ids_backup.name}")
            except Exception as e:
                print(f"  Ошибка удаления бэкапа {ids_backup.name}")

        backup_path = embeddings_dir / "part_ids.npy.backup"
        try:
            ids_file.rename(backup_path)
            backup_files.append(backup_path)
            print(f" Создан бэкап: {backup_path.name}")
        except Exception as e:
            print(f"  Ошибка создания бэкапа {ids_file.name}: {e}")
    
    if meta_file.exists():
        meta_backup = embeddings_dir / "metadata.json.backup"
        if meta_backup.exists():
            try:
                meta_backup.unlink()
                print(f"  Удален старый бэкап {meta_backup.name}")
            except Exception as e:
                print(f"  Ошибка удаления бэкапа {meta_backup.name}")

        backup_path = embeddings_dir / "metadata.json.backup"
        try:
            meta_file.rename(backup_path)
            backup_files.append(backup_path)
            print(f" Создан бэкап: {backup_path.name}")
        except Exception as e:
            print(f"  Ошибка создания бэкапа {meta_file.name}: {e}")
    
    return backup_files

def save_embeddings(embeddings_dir: Path, embeddings: np.ndarray, part_ids: np.ndarray, metadata: dict):
    """Сохранение эмбеддингов с оптимизацией размера"""
    embeddings_dir.mkdir(parents=True, exist_ok=True)
    
    emb_file = embeddings_dir / "per_image.npy"
    ids_file = embeddings_dir / "part_ids.npy"
    meta_file = embeddings_dir / "metadata.json"
    
    # Создаем бэкап существующих файлов
    if emb_file.exists():
        emb_file.rename(embeddings_dir / "per_image.npy.backup")
    if ids_file.exists():
        ids_file.rename(embeddings_dir / "part_ids.npy.backup")
    
    # Оптимизация: float32 и убедимся в правильном типе
    embeddings_opt = embeddings.astype(np.float32)
    part_ids_opt = np.array(part_ids, dtype=object)
    
    # Сохраняем оптимизированные данные
    np.save(emb_file, embeddings_opt)
    np.save(ids_file, part_ids_opt)
    
    # Обновляем метаданные
    metadata['updated_at'] = datetime.now().isoformat()
    metadata['total_embeddings'] = len(embeddings_opt)
    metadata['embedding_dim'] = embeddings_opt.shape[1] if len(embeddings_opt.shape) > 1 else 2048
    metadata['data_type'] = str(embeddings_opt.dtype)
    
    with open(meta_file, 'w') as f:
        json.dump(metadata, f, indent=2, ensure_ascii=False)
    
    size_mb = (emb_file.stat().st_size + ids_file.stat().st_size) / (1024 * 1024)
    print(f" Сохранено {len(embeddings_opt)} эмбеддингов ({size_mb:.1f} MB)")

def extract_embeddings(src: Path, out: Path, update: bool = True, force: bool = False):
    """Основная функция извлечения эмбеддингов"""
    print(f" Извлечение эмбеддингов")
    print(f" Исходная директория: {src}")
    print(f" Директория для эмбеддингов: {out}")
    print(f" Режим: {'Инкрементальное обновление' if update and not force else 'Полная перезапись'}")
    
    start_time = time.time()
    
    # Обработка --force режима
    if force:
        print("  Принудительная перезапись - удаление существующих файлов...")
        backup_files = remove_existing_files(out)
        if backup_files:
            print(f" Созданы бэкапы: {len(backup_files)} файлов")
        # Устанавливаем update=False для полной перезаписи
        update = False
    
    # Загрузка существующих данных (если update=True)
    existing_embs, existing_ids, metadata, seen_files = None, None, {}, set()
    if update:
        existing_embs, existing_ids, metadata, seen_files = load_existing_data(out)
        
        # Если загрузка не удалась из-за ошибок, предлагаем --force
        if existing_embs is None and (out / "per_image.npy").exists():
            print(" Невозможно выполнить инкрементальное обновление из-за ошибок в существующих файлах")
            print(" Выполните команду с --force для принудительной перезаписи:")
            print(f"   python {Path(__file__).name} --src {src} --out {out} --force")
            return False
    
    # Загрузка модели
    device = get_device(getattr(config, 'DEVICE', 'auto'))
    print(f" Используется устройство: {device}")
    
    model = ResNet50Encoder(out_dim=2048, pretrained=True)
    model = model.to(device).eval()
    print(" Модель загружена")
    
    # Сбор списка новых изображений
    all_images = []
    for part_dir in src.iterdir():
        if not part_dir.is_dir():
            continue
        part_id = part_dir.name  # Это ID детали!
        for img_path in part_dir.glob("*.jpg"):
            rel_id = f"{part_id}/{img_path.name}"  # part_id/image_name.jpg
            if update and rel_id in seen_files:
                continue
            all_images.append((img_path, part_id, rel_id))
    
    print(f" Найдено изображений для обработки: {len(all_images)}")
    
    if len(all_images) == 0:
        if update:
            print(" Новых изображений нет")
        else:
            print(" Нет изображений для обработки")
        return True
    
    # Извлечение эмбеддингов
    new_embeddings = []
    new_ids = []  # Это rel_id (part_id/image_name.jpg)
    error_count = 0
    success_count = 0
    
    print(" Извлечение эмбеддингов...")
    pbar = tqdm(all_images, desc="Обработка изображений")
    
    for img_path, part_id, rel_id in pbar:
        try:
            # Извлечение эмбеддинга
            emb = embed_image(img_path, model, device)
            if emb is None:
                error_count += 1
                continue
            
            new_embeddings.append(emb)
            new_ids.append(rel_id)  # Сохраняем part_id/image_name.jpg
            success_count += 1
            
            pbar.set_postfix({
                'Успех': success_count,
                'Ошибок': error_count
            })
            
        except Exception as e:
            print(f" Ошибка обработки {img_path}: {e}")
            error_count += 1
            continue
    
    if len(new_embeddings) == 0:
        print(" Не удалось извлечь ни одного эмбеддинга")
        return False
    
    # Конвертация в numpy массивы
    new_embeddings = np.array(new_embeddings, dtype=np.float32)
    new_ids = np.array(new_ids, dtype=object)
    
    print(f" Успешно извлечено: {len(new_embeddings)} эмбеддингов")
    if error_count > 0:
        print(f" Ошибок: {error_count}")
    
    # Объединение с существующими данными
    if existing_embs is not None and existing_ids is not None:
        final_embs = np.concatenate([existing_embs, new_embeddings], axis=0)
        final_ids = np.concatenate([existing_ids, new_ids], axis=0)
        print(f" Всего эмбеддингов после объединения: {len(final_embs)}")
    else:
        final_embs = new_embeddings
        final_ids = new_ids
        print(f" Новых эмбеддингов: {len(final_embs)}")
    
    # Сохранение результатов
    save_embeddings(out, final_embs, final_ids, metadata)
    
    # Финальная статистика
    total_time = time.time() - start_time
    print(f"\n РЕЗУЛЬТАТЫ")
    print(f" Успешно обработано: {success_count}")
    print(f" Ошибок: {error_count}")
    print(f" Всего: {success_count + error_count}")
    if success_count > 0:
        print(f"  Время обработки: {total_time:.2f} сек")
        print(f" Производительность: {success_count/total_time:.2f} изображений/сек")
    print(f" Сохранено в: {out}")
    
    return True

def main():
    parser = argparse.ArgumentParser(
        description="Извлечение эмбеддингов из изображений",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Примеры:
  %(prog)s --src data/processed --out data/embeddings
  %(prog)s --src data/processed --out data/embeddings --update
  %(prog)s --src data/processed --out data/embeddings --force
  %(prog)s --src data/processed --out data/embeddings --device cpu
        """
    )
    
    parser.add_argument("--src", type=Path, default=Path("data/processed"),
                       help="Директория с обработанными изображениями")
    parser.add_argument("--out", type=Path, default=Path("data/embeddings"),
                       help="Директория для сохранения эмбеддингов")
    parser.add_argument("--update", action="store_true", default=True,
                       help="Инкрементальное обновление (по умолчанию)")
    parser.add_argument("--force", action="store_true",
                       help="Полная перезапись (удалить существующие)")
    parser.add_argument("--device", type=str, default="auto",
                       help="Устройство: 'cpu', 'cuda', 'mps', 'auto' (по умолчанию)")
    
    args = parser.parse_args()
    
    # Если указан --force, отключаем update
    if args.force:
        args.update = False
    
    # Проверка существования исходной директории
    if not args.src.exists():
        print(f" Исходная директория не найдена: {args.src}")
        return 1
    
    # Устанавливаем устройство из аргументов
    if hasattr(config, 'DEVICE'):
        config.DEVICE = args.device
    
    try:
        success = extract_embeddings(args.src, args.out, args.update, args.force)
        if success:
            print("\n Извлечение эмбеддингов завершено!")
            return 0
        else:
            print("\n Ошибка извлечения эмбеддингов")
            return 1
    except KeyboardInterrupt:
        print("\n  Обработка прервана пользователем")
        return 1
    except Exception as e:
        print(f"\n Критическая ошибка: {e}")
        import traceback
        traceback.print_exc()
        return 1

if __name__ == "__main__":
    sys.exit(main())
