# Утилита препроцессинга фото поддерживает тестовый и полный запуск в shell с помощью команд, описанных ниже
"""
Предобработка изображений сельхоз-запчастей с YOLO
С кэшированием моделей и пропуском существующих файлов

CLI:
1. Обычная обработка (пропуск существующих):
python src/preprocess.py --src data/raw --dst data/processed --size 384

2. Принудительная обработка всех файлов:
python src/preprocess.py --src data/raw --dst data/processed --size 384 --force

3. Тестовая обработка новых файлов:
python src/preprocess.py --src data/raw --dst data/processed --size 512 --test --limit 50

--skip-existing (параметр по умолчанию) - пропускать существующие фото в датасете, позволяет быстро добавлять новые данные
--force - обработать все заново, включая все существующие фото в data/processed

# Все эти размеры поддерживаются ResNet50:
sizes = [224, 256, 288, 320, 384, 448, 512, 640, 768]

# Производительность vs Качество:
# 224×224 - быстрее, но может терять детали
# 384×384 - оптимальный баланс (рекомендуется)
# 512×512 - максимальное качество
"""

import argparse
from pathlib import Path
import cv2 as cv
import numpy as np
from tqdm import tqdm
import sys
from typing import Tuple, Optional, Dict, Any
import time
import torch
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

class YOLOPartPreprocessor:
    """Предобработчик изображений сельхоз-запчастей с YOLO"""
    
    def __init__(self, target_size: int = 384,
                 yolo_model: str = 'yolov8n.pt',
                 confidence_threshold: float = 0.25,
                 iou_threshold: float = 0.45,
                 preserve_aspect_ratio: bool = True,
                 device: str = 'auto',
                 skip_existing: bool = True,
                 models_dir: str = 'src/models'):
        self.target_size = target_size
        self.confidence_threshold = confidence_threshold
        self.iou_threshold = iou_threshold
        self.preserve_aspect_ratio = preserve_aspect_ratio
        self.skip_existing = skip_existing
        self.models_dir = Path(models_dir)
        
        # Создаем директорию для моделей
        self.models_dir.mkdir(parents=True, exist_ok=True)
        
        # Установка устройства
        if device == 'auto':
            self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        else:
            self.device = device
        
        print(f" Используется устройство: {self.device}")
        print(f" Директория моделей: {self.models_dir}")
        
        # Загрузка YOLO модели
        self.yolo_model = self._load_yolo_model(yolo_model)
    
    def _get_model_path(self, model_name: str) -> Path:
        """Получение пути к локальной копии модели"""
        # Если модель уже локальный файл
        if Path(model_name).exists():
            return Path(model_name)
        
        # Иначе ищем в директории моделей
        local_model_path = self.models_dir / model_name
        return local_model_path
    
    def _download_model_if_needed(self, model_name: str) -> str:
        """Скачивание модели если она не существует локально"""
        local_model_path = self._get_model_path(model_name)
        
        # Если модель уже существует локально
        if local_model_path.exists():
            print(f" Используется локальная модель: {local_model_path}")
            return str(local_model_path)
        
        # Если это стандартная модель YOLO, скачиваем
        standard_models = ['yolov8n.pt', 'yolov8s.pt', 'yolov8m.pt', 'yolov8l.pt', 'yolov8x.pt']
        
        if model_name in standard_models:
            print(f" Скачивание модели {model_name} в {local_model_path}...")
            try:
                from ultralytics import YOLO
                # Создаем временную модель для скачивания
                temp_model = YOLO(model_name)
                # Сохраняем модель локально
                temp_model.model.save(str(local_model_path))
                print(f" Модель сохранена: {local_model_path}")
                return str(local_model_path)
            except Exception as e:
                print(f"  Ошибка скачивания модели: {e}")
                print(" Используется онлайн-загрузка...")
                return model_name  # Вернем оригинальное имя для онлайн-загрузки
        else:
            # Для пользовательских моделей просто возвращаем имя
            return model_name
    
    def _load_yolo_model(self, model_name: str):
        """Загрузка YOLO модели с кэшированием"""
        try:
            from ultralytics import YOLO
            
            # Проверяем и скачиваем модель при необходимости
            local_model_path = self._download_model_if_needed(model_name)
            
            print(f" Загрузка YOLO модели: {local_model_path}")
            
            model = YOLO(local_model_path)
            # Установка устройства
            model.to(self.device)
            
            print(" YOLO модель загружена успешно")
            return model
            
        except ImportError:
            print(" Ошибка: библиотека 'ultralytics' не установлена")
            print(" Установите: pip install ultralytics")
            sys.exit(1)
        except Exception as e:
            print(f" Ошибка загрузки YOLO модели: {e}")
            sys.exit(1)
    
    def read_image_safe(self, path: Path) -> Optional[np.ndarray]:
        """Безопасное чтение изображения"""
        try:
            with open(path, 'rb') as f:
                file_bytes = np.asarray(bytearray(f.read()), dtype=np.uint8)
                img = cv.imdecode(file_bytes, cv.IMREAD_COLOR)
            return img
        except Exception as e:
            print(f" Ошибка чтения {path}: {e}")
            return None
    
    def write_image_safe(self, path: Path, img: np.ndarray) -> bool:
        """Безопасная запись изображения"""
        try:
            path.parent.mkdir(parents=True, exist_ok=True)
            is_success, buffer = cv.imencode(".jpg", img, [cv.IMWRITE_JPEG_QUALITY, 95])
            if is_success:
                with open(path, 'wb') as f:
                    f.write(buffer)
                return True
            return False
        except Exception as e:
            print(f" Ошибка записи {path}: {e}")
            return False
    
    def detect_part_with_yolo(self, image: np.ndarray) -> Optional[Tuple[int, int, int, int]]:
        """
        Детекция сельхоз-запчасти с помощью YOLO
        """
        try:
            results = self.yolo_model(
                image, 
                conf=self.confidence_threshold,
                iou=self.iou_threshold,
                device=self.device,
                verbose=False
            )
            
            if len(results) > 0:
                boxes = results[0].boxes
                
                if boxes is not None and len(boxes) > 0:
                    bboxes = boxes.xyxy.cpu().numpy()
                    confidences = boxes.conf.cpu().numpy()
                    
                    # Выбираем лучший бокс
                    best_idx = np.argmax(confidences)
                    box = bboxes[best_idx]
                    
                    x1, y1, x2, y2 = map(int, box)
                    width = x2 - x1
                    height = y2 - y1
                    
                    # Увеличенные отступы (30% для сельхоз деталей)
                    padding_x = int(width * 0.3)
                    padding_y = int(height * 0.3)
                    
                    # Корректируем координаты
                    h, w = image.shape[:2]
                    x1_padded = max(0, x1 - padding_x)
                    y1_padded = max(0, y1 - padding_y)
                    x2_padded = min(w, x2 + padding_x)
                    y2_padded = min(h, y2 + padding_y)
                    
                    final_width = x2_padded - x1_padded
                    final_height = y2_padded - y1_padded
                    
                    return (x1_padded, y1_padded, final_width, final_height)
            
            return None
            
        except Exception as e:
            print(f"  Ошибка YOLO детекции: {e}")
            return None
    
    def enhance_image_quality(self, image: np.ndarray) -> np.ndarray:
        """Улучшенное качество для сельхоз-деталей"""
        try:
            h, w = image.shape[:2]
            min_dim = min(h, w)
            
            if min_dim < 100:
                enhanced = cv.bilateralFilter(image, 5, 30, 30)
            elif min_dim < 200:
                lab = cv.cvtColor(image, cv.COLOR_BGR2LAB)
                l, a, b = cv.split(lab)
                clahe = cv.createCLAHE(clipLimit=2.5, tileGridSize=(6, 6))
                l = clahe.apply(l)
                enhanced = cv.merge([l, a, b])
                enhanced = cv.cvtColor(enhanced, cv.COLOR_LAB2BGR)
                enhanced = cv.bilateralFilter(enhanced, 7, 50, 50)
            else:
                lab = cv.cvtColor(image, cv.COLOR_BGR2LAB)
                l, a, b = cv.split(lab)
                clahe = cv.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
                l = clahe.apply(l)
                enhanced = cv.merge([l, a, b])
                enhanced = cv.cvtColor(enhanced, cv.COLOR_LAB2BGR)
                enhanced = cv.bilateralFilter(enhanced, 9, 75, 75)
            
            if min_dim > 150:
                kernel = np.array([[-1, -1, -1],
                                  [-1,  9, -1],
                                  [-1, -1, -1]])
                enhanced = cv.filter2D(enhanced, -1, kernel)
            
            return enhanced
            
        except Exception:
            return image
    
    def resize_with_padding_or_crop(self, image: np.ndarray) -> np.ndarray:
        """Изменение размера с padding или обрезкой"""
        h, w = image.shape[:2]
        
        if self.preserve_aspect_ratio:
            scale = self.target_size / max(h, w)
            new_w = int(w * scale)
            new_h = int(h * scale)
            
            # Изменяем размер
            if scale > 1:
                resized = cv.resize(image, (new_w, new_h), interpolation=cv.INTER_LANCZOS4)
            elif scale < 0.5:
                resized = cv.resize(image, (new_w, new_h), interpolation=cv.INTER_AREA)
            else:
                resized = cv.resize(image, (new_w, new_h), interpolation=cv.INTER_CUBIC)
            
            # Добавляем padding
            top = (self.target_size - new_h) // 2
            bottom = self.target_size - new_h - top
            left = (self.target_size - new_w) // 2
            right = self.target_size - new_w - left
            
            padded = cv.copyMakeBorder(
                resized, top, bottom, left, right, 
                cv.BORDER_CONSTANT, value=[128, 128, 128]
            )
            
            return padded
        else:
            return cv.resize(image, (self.target_size, self.target_size), 
                           interpolation=cv.INTER_LANCZOS4)
    
    def preprocess_single_image(self, src_path: Path, dst_path: Path) -> Dict[str, Any]:
        """
        Полная предобработка одного изображения
        """
        stats = {
            'success': False,
            'operations': [],
            'original_size': None,
            'final_size': None,
            'detection_confidence': 0.0,
            'processing_time': 0.0,
            'skipped': False
        }
        
        start_time = time.time()
        
        try:
            # Проверка существования выходного файла
            if self.skip_existing and dst_path.exists():
                stats['skipped'] = True
                stats['success'] = True
                stats['operations'].append('skipped_existing')
                stats['processing_time'] = time.time() - start_time
                return stats
            
            # Чтение изображения
            image = self.read_image_safe(src_path)
            if image is None:
                stats['error'] = 'read_failed'
                stats['processing_time'] = time.time() - start_time
                return stats
            
            stats['original_size'] = f"{image.shape[1]}x{image.shape[0]}"
            
            # Детекция объекта
            bbox = self.detect_part_with_yolo(image)
            
            if bbox is not None:
                x, y, w, h = bbox
                cropped = image[y:y+h, x:x+w]
                stats['operations'].append('yolo_detection')
                stats['detection_confidence'] = 0.5
            else:
                # Fallback: центральная обрезка до квадрата
                h_img, w_img = image.shape[:2]
                size = min(h_img, w_img)
                y_start = (h_img - size) // 2
                x_start = (w_img - size) // 2
                cropped = image[y_start:y_start+size, x_start:x_start+size]
                stats['operations'].append('center_crop')
                stats['detection_confidence'] = 0.0
            
            # Улучшение качества
            enhanced = self.enhance_image_quality(cropped)
            stats['operations'].append('quality_enhancement')
            
            # Изменение размера
            final_image = self.resize_with_padding_or_crop(enhanced)
            stats['operations'].append('resize')
            stats['final_size'] = f"{final_image.shape[1]}x{final_image.shape[0]}"
            
            # Сохранение результата
            success = self.write_image_safe(dst_path, final_image)
            stats['success'] = success
            
            if not success:
                stats['error'] = 'save_failed'
                
        except Exception as e:
            stats['error'] = str(e)
            stats['success'] = False
        
        stats['processing_time'] = time.time() - start_time
        return stats
    
    def get_pending_images(self, src_root: Path, dst_root: Path) -> list:
        """Получение списка изображений, требующих обработки"""
        # Поиск всех изображений
        files = list(src_root.rglob("*"))
        images = [p for p in files if p.suffix.lower() in [".jpg", ".jpeg", ".png"]]
        
        if not self.skip_existing:
            return images
        
        # Фильтрация: только те, которых нет в выходной директории
        pending_images = []
        for src_path in images:
            rel_path = src_path.relative_to(src_root)
            dst_path = dst_root / rel_path.with_suffix(".jpg")
            if not dst_path.exists():
                pending_images.append(src_path)
        
        return pending_images
    
    def process_dataset(self, src_root: Path, dst_root: Path, 
                       test_mode: bool = False, 
                       test_limit: int = 100) -> Dict[str, int]:
        """
        Обработка всего датасета с пропуском существующих файлов
        """
        print(f" Обработка датасета с YOLO: {src_root} → {dst_root}")
        print(f" Целевой размер: {self.target_size}×{self.target_size}")
        print(f" Сохранение пропорций: {'Да' if self.preserve_aspect_ratio else 'Нет'}")
        print(f" >Пропуск существующих: {'Да' if self.skip_existing else 'Нет'}")
        print(f" Директория моделей: {self.models_dir}")
        
        # Получение списка изображений для обработки
        all_images = list(src_root.rglob("*"))
        all_images = [p for p in all_images if p.suffix.lower() in [".jpg", ".jpeg", ".png"]]
        total_images = len(all_images)
        
        pending_images = self.get_pending_images(src_root, dst_root)
        pending_count = len(pending_images)
        
        print(f" Всего изображений: {total_images}")
        print(f" Требуют обработки: {pending_count}")
        
        if self.skip_existing and pending_count == 0:
            print(" Все изображения уже обработаны!")
            return {
                'processed': 0,
                'successful': 0,
                'failed': 0,
                'read_errors': 0,
                'save_errors': 0,
                'detection_success': 0,
                'skipped': total_images,
                'total_processing_time': 0.0
            }
        
        # Тестовый режим
        if test_mode:
            # Для теста берем из pending_images
            images_to_process = pending_images[:test_limit] if pending_images else all_images[:test_limit]
            print(f" Тестовый режим: обработка первых {len(images_to_process)} изображений")
        else:
            images_to_process = pending_images if self.skip_existing else all_images
            print(f" Обработка {len(images_to_process)} изображений")
        
        # Статистика
        stats = {
            'processed': 0,
            'successful': 0,
            'failed': 0,
            'read_errors': 0,
            'save_errors': 0,
            'detection_success': 0,
            'skipped': total_images - pending_count if self.skip_existing else 0,
            'total_processing_time': 0.0
        }
        
        pbar = tqdm(images_to_process, desc="Обработка", unit="img")
        
        for src_path in pbar:
            rel_path = src_path.relative_to(src_root)
            dst_path = dst_root / rel_path.with_suffix(".jpg")
            
            result = self.preprocess_single_image(src_path, dst_path)
            
            stats['processed'] += 1
            stats['total_processing_time'] += result.get('processing_time', 0)
            
            if result['success']:
                if result.get('skipped', False):
                    # Уже подсчитано в 'skipped'
                    pass
                else:
                    stats['successful'] += 1
                    if 'yolo_detection' in result.get('operations', []):
                        stats['detection_success'] += 1
            else:
                stats['failed'] += 1
                if result.get('error') == 'read_failed':
                    stats['read_errors'] += 1
                elif result.get('error') == 'save_failed':
                    stats['save_errors'] += 1
            
            # Обновление прогресс-бара
            pbar.set_postfix({
                'Новые': stats['successful'],
                'YOLO': stats['detection_success'],
                'Ошибка': stats['failed'],
                'Пропущ': stats['skipped']
            })
        
        return stats

def main():
    parser = argparse.ArgumentParser(
        description="Предобработка изображений сельхоз-запчастей с YOLO",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Рекомендуемые размеры:
  224 - стандартный (быстро, экономия памяти)
  384 - оптимальный баланс (рекомендуется)
  512 - максимальное качество

Примеры:
  %(prog)s --src data/raw --dst data/processed --size 384
  %(prog)s --src data/raw --dst data/processed --size 384 --test --limit 100
  %(prog)s --src data/raw --dst data/processed --force  # Обработать все заново
        """
    )
    
    parser.add_argument("--src", type=str, required=True,
                       help="Путь к исходным изображениям")
    parser.add_argument("--dst", type=str, required=True,
                       help="Путь для сохранения обработанных изображений")
    parser.add_argument("--test", action="store_true",
                       help="Тестовый режим")
    parser.add_argument("--limit", type=int, default=100,
                       help="Лимит изображений для теста (default: 100)")
    parser.add_argument("--size", type=int, default=384,
                       help="Целевой размер изображения (default: 384)")
    parser.add_argument("--model", type=str, default='yolov8n.pt',
                       help="YOLO модель (default: yolov8n.pt)")
    parser.add_argument("--conf", type=float, default=0.25,
                       help="Порог уверенности (default: 0.25)")
    parser.add_argument("--iou", type=float, default=0.45,
                       help="Порог IoU (default: 0.45)")
    parser.add_argument("--preserve-aspect", action="store_true", default=True,
                       help="Сохранять пропорции изображения (default: True)")
    parser.add_argument("--no-preserve-aspect", dest="preserve_aspect", action="store_false",
                       help="Не сохранять пропорции (растянуть до квадрата)")
    parser.add_argument("--device", type=str, default='auto',
                       help="Устройство: 'cpu', 'cuda', 'cuda:0', 'auto' (default: auto)")
    parser.add_argument("--force", action="store_true",
                       help="Обработать все файлы заново (игнорировать существующие)")
    parser.add_argument("--skip-existing", action="store_true", default=True,
                       help="Пропускать уже обработанные файлы (default: True)")
    parser.add_argument("--models-dir", type=str, default='src/models',
                       help="Директория для кэширования моделей (default: src/models)")
    
    args = parser.parse_args()
    
    # Если указан --force, отключаем пропуск существующих
    if args.force:
        args.skip_existing = False
    
    src_path = Path(args.src)
    dst_path = Path(args.dst)
    
    if not src_path.exists():
        print(f" Исходная директория не найдена: {src_path}")
        return 1
    
    print(" Начало предобработки изображений сельхоз-запчастей (YOLO)")
    print("=" * 70)
    print(f" Размер изображений: {args.size}×{args.size}")
    print(f" Сохранение пропорций: {'Да' if args.preserve_aspect else 'Нет'}")
    print(f" >Пропуск существующих: {'Да' if args.skip_existing else 'Нет'}")
    print(f" Устройство: {args.device}")
    print(f" Директория моделей: {args.models_dir}")
    
    preprocessor = YOLOPartPreprocessor(
        target_size=args.size,
        yolo_model=args.model,
        confidence_threshold=args.conf,
        iou_threshold=args.iou,
        preserve_aspect_ratio=args.preserve_aspect,
        device=args.device,
        skip_existing=args.skip_existing,
        models_dir=args.models_dir
    )
    
    start_time = time.time()
    stats = preprocessor.process_dataset(
        src_path, dst_path, 
        test_mode=args.test, 
        test_limit=args.limit
    )
    total_time = time.time() - start_time
    
    print(" РЕЗУЛЬТАТЫ ОБРАБОТКИ")
    print(f" Успешно обработано: {stats['successful']}")
    print(f" Ошибок: {stats['failed']}")
    print(f"   - Ошибки чтения: {stats['read_errors']}")
    print(f"   - Ошибки записи: {stats['save_errors']}")
    print(f" Успешная детекция YOLO: {stats['detection_success']}")
    if args.skip_existing:
        print(f"  >Пропущено существующих: {stats['skipped']}")
    print(f" Всего обработано: {stats['processed']}")
    
    if stats['processed'] > 0:
        success_rate = stats['successful'] / stats['processed'] * 100
        detection_rate = stats['detection_success'] / stats['processed'] * 100
        print(f" Процент успеха: {success_rate:.1f}%")
        print(f" Процент детекции: {detection_rate:.1f}%")
        
        if stats['processed'] > 0:
            avg_time = stats['total_processing_time'] / stats['processed']
            print(f"  Среднее время на изображение: {avg_time:.2f} сек")
    
    print(f"  Общее время обработки: {total_time:.2f} сек")
    
    if stats['successful'] > 0:
        images_per_second = stats['successful'] / total_time if total_time > 0 else 0
        print(f" Производительность: {images_per_second:.2f} изображений/сек")
    
    print(f"\n Результаты сохранены в: {dst_path}")
    
    # Рекомендации
    if args.size < 256:
        print("\n РЕКОМЕНДАЦИИ:")
        print("     Рекомендуется использовать размер ≥ 384 для сельхоз-запчастей")
        print("    384×384 - оптимальный баланс качество/производительность")
    
    return 0

if __name__ == "__main__":
    sys.exit(main())
