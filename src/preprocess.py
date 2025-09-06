#!/usr/bin/env python3
"""
Предобработка изображений сельхоз-запчастей с использованием YOLO
"""

import argparse
from pathlib import Path
import cv2 as cv
import numpy as np
from tqdm import tqdm
import sys
from typing import Tuple, Optional, Dict, Any
import time

class YOLOPartPreprocessor:
    """Предобработчик изображений сельхоз-запчастей с YOLO"""
    
    def __init__(self, target_size: int = 224, 
                 yolo_model: str = 'yolov8n.pt',
                 confidence_threshold: float = 0.25,
                 iou_threshold: float = 0.45):
        self.target_size = target_size
        self.confidence_threshold = confidence_threshold
        self.iou_threshold = iou_threshold
        
        # Загрузка YOLO модели
        self.yolo_model = self._load_yolo_model(yolo_model)
    
    def _load_yolo_model(self, model_name: str):
        """Загрузка YOLO модели"""
        try:
            from ultralytics import YOLO
            print(f"🔄 Загрузка YOLO модели: {model_name}")
            
            # Автоматическая загрузка модели (если не найдена - скачает)
            model = YOLO(model_name)
            print("✅ YOLO модель загружена успешно")
            return model
            
        except ImportError:
            print("❌ Ошибка: библиотека 'ultralytics' не установлена")
            print("💡 Установите: pip install ultralytics")
            sys.exit(1)
        except Exception as e:
            print(f"❌ Ошибка загрузки YOLO модели: {e}")
            sys.exit(1)
    
    def read_image_safe(self, path: Path) -> Optional[np.ndarray]:
        """Безопасное чтение изображения"""
        try:
            with open(path, 'rb') as f:
                file_bytes = np.asarray(bytearray(f.read()), dtype=np.uint8)
                img = cv.imdecode(file_bytes, cv.IMREAD_COLOR)
            return img
        except Exception as e:
            print(f"❌ Ошибка чтения {path}: {e}")
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
            print(f"❌ Ошибка записи {path}: {e}")
            return False
    
    def detect_part_with_yolo(self, image: np.ndarray) -> Optional[Tuple[int, int, int, int]]:
        """
        Детекция сельхоз-запчасти с помощью YOLO
        Возвращает (x, y, w, h) bounding box или None
        """
        try:
            # Выполняем детекцию
            results = self.yolo_model(
                image, 
                conf=self.confidence_threshold,
                iou=self.iou_threshold,
                verbose=False
            )
            
            # Обрабатываем результаты
            if len(results) > 0:
                boxes = results[0].boxes
                
                if boxes is not None and len(boxes) > 0:
                    # Получаем bounding boxes и уверенности
                    bboxes = boxes.xyxy.cpu().numpy()  # [x1, y1, x2, y2]
                    confidences = boxes.conf.cpu().numpy()
                    
                    # Выбираем бокс с максимальной уверенностью
                    best_idx = np.argmax(confidences)
                    best_conf = confidences[best_idx]
                    box = bboxes[best_idx]
                    
                    x1, y1, x2, y2 = map(int, box)
                    width = x2 - x1
                    height = y2 - y1
                    
                    # Добавляем отступы (20% от размера объекта)
                    padding_x = int(width * 0.2)
                    padding_y = int(height * 0.2)
                    
                    # Корректируем координаты с учетом границ изображения
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
            print(f"⚠️  Ошибка YOLO детекции: {e}")
            return None
    
    def enhance_image_quality(self, image: np.ndarray) -> np.ndarray:
        """Улучшение качества изображения для сельхоз-запчастей"""
        try:
            # LAB цветовое пространство для лучшей обработки яркости
            lab = cv.cvtColor(image, cv.COLOR_BGR2LAB)
            l, a, b = cv.split(lab)
            
            # CLAHE для улучшения контраста (адаптивный)
            clahe = cv.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
            l = clahe.apply(l)
            
            # Объединяем обратно
            enhanced = cv.merge([l, a, b])
            enhanced = cv.cvtColor(enhanced, cv.COLOR_LAB2BGR)
            
            # Мягкое шумоподавление
            enhanced = cv.bilateralFilter(enhanced, 9, 75, 75)
            
            # Легкое увеличение резкости
            kernel = np.array([[-1, -1, -1],
                              [-1,  9, -1],
                              [-1, -1, -1]])
            enhanced = cv.filter2D(enhanced, -1, kernel)
            
            return enhanced
            
        except Exception:
            # Fallback если что-то пошло не так
            return image
    
    def smart_resize_with_padding(self, image: np.ndarray) -> np.ndarray:
        """Умное изменение размера с padding до квадрата"""
        h, w = image.shape[:2]
        
        # Определяем коэффициент масштабирования
        scale = self.target_size / max(h, w)
        new_w = int(w * scale)
        new_h = int(h * scale)
        
        # Изменяем размер с высоким качеством
        if scale > 1:
            # Увеличение - используем лучший алгоритм
            resized = cv.resize(image, (new_w, new_h), interpolation=cv.INTER_LANCZOS4)
        else:
            # Уменьшение - используем area для лучшего антиалиасинга
            resized = cv.resize(image, (new_w, new_h), interpolation=cv.INTER_AREA)
        
        # Добавляем padding до квадрата
        top = (self.target_size - new_h) // 2
        bottom = self.target_size - new_h - top
        left = (self.target_size - new_w) // 2
        right = self.target_size - new_w - left
        
        # Заполняем серым цветом (нейтральный фон для технических деталей)
        padded = cv.copyMakeBorder(
            resized, top, bottom, left, right, 
            cv.BORDER_CONSTANT, value=[128, 128, 128]
        )
        
        return padded
    
    def preprocess_single_image(self, src_path: Path, dst_path: Path) -> Dict[str, Any]:
        """
        Полная предобработка одного изображения с YOLO
        """
        stats = {
            'success': False,
            'operations': [],
            'detection_confidence': 0.0,
            'processing_time': 0.0
        }
        
        start_time = time.time()
        
        try:
            # 1. Чтение изображения
            image = self.read_image_safe(src_path)
            if image is None:
                stats['error'] = 'read_failed'
                stats['processing_time'] = time.time() - start_time
                return stats
            
            # 2. Детекция объекта с помощью YOLO
            bbox = self.detect_part_with_yolo(image)
            
            if bbox is not None:
                x, y, w, h = bbox
                cropped = image[y:y+h, x:x+w]
                stats['operations'].append('yolo_detection')
                stats['detection_confidence'] = 0.5  # Примерное значение
            else:
                # Fallback: центральная обрезка
                h_img, w_img = image.shape[:2]
                size = min(h_img, w_img)
                y_start = (h_img - size) // 2
                x_start = (w_img - size) // 2
                cropped = image[y_start:y_start+size, x_start:x_start+size]
                stats['operations'].append('center_crop')
                stats['detection_confidence'] = 0.0
            
            # 3. Улучшение качества
            enhanced = self.enhance_image_quality(cropped)
            stats['operations'].append('quality_enhancement')
            
            # 4. Стандартизация размера
            final_image = self.smart_resize_with_padding(enhanced)
            stats['operations'].append('resize_padding')
            
            # 5. Сохранение результата
            success = self.write_image_safe(dst_path, final_image)
            stats['success'] = success
            
            if not success:
                stats['error'] = 'save_failed'
                
        except Exception as e:
            stats['error'] = str(e)
            stats['success'] = False
        
        stats['processing_time'] = time.time() - start_time
        return stats
    
    def process_dataset(self, src_root: Path, dst_root: Path, 
                       test_mode: bool = False, 
                       test_limit: int = 100) -> Dict[str, int]:
        """
        Обработка всего датасета с YOLO
        """
        print(f"🔍 Обработка датасета с YOLO: {src_root} → {dst_root}")
        print(f"🤖 Используется модель: YOLOv8")
        
        # Поиск всех изображений
        files = list(src_root.rglob("*"))
        images = [p for p in files if p.suffix.lower() in [".jpg", ".jpeg", ".png"]]
        
        print(f"📊 Найдено изображений: {len(images)}")
        
        # Тестовый режим
        if test_mode:
            images = images[:test_limit]
            print(f"🧪 Тестовый режим: обработка первых {len(images)} изображений")
        
        # Статистика
        stats = {
            'processed': 0,
            'successful': 0,
            'failed': 0,
            'read_errors': 0,
            'save_errors': 0,
            'detection_success': 0,
            'total_processing_time': 0.0
        }
        
        # Обработка с прогресс-баром
        pbar = tqdm(images, desc="Обработка", unit="img")
        
        for src_path in pbar:
            rel_path = src_path.relative_to(src_root)
            dst_path = dst_root / rel_path.with_suffix(".jpg")
            
            result = self.preprocess_single_image(src_path, dst_path)
            
            # Обновление статистики
            stats['processed'] += 1
            stats['total_processing_time'] += result.get('processing_time', 0)
            
            if result['success']:
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
                'Успех': stats['successful'],
                'YOLO': stats['detection_success'],
                'Ошибка': stats['failed']
            })
        
        return stats

def main():
    parser = argparse.ArgumentParser(
        description="Предобработка изображений сельхоз-запчастей с YOLO",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Примеры использования:
  %(prog)s --src data/raw --dst data/processed --test
  %(prog)s --src data/raw --dst data/processed --limit 100
  %(prog)s --src data/raw --dst data/processed --model yolov8s.pt
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
    parser.add_argument("--size", type=int, default=224,
                       help="Целевой размер изображения (default: 224)")
    parser.add_argument("--model", type=str, default='yolov8n.pt',
                       help="YOLO модель (default: yolov8n.pt)")
    parser.add_argument("--conf", type=float, default=0.25,
                       help="Порог уверенности (default: 0.25)")
    parser.add_argument("--iou", type=float, default=0.45,
                       help="Порог IoU (default: 0.45)")
    
    args = parser.parse_args()
    
    # Проверка путей
    src_path = Path(args.src)
    dst_path = Path(args.dst)
    
    if not src_path.exists():
        print(f"❌ Исходная директория не найдена: {src_path}")
        return 1
    
    # Создание препроцессора
    print("🚜 Начало предобработки изображений сельхоз-запчастей (YOLO)")
    print("=" * 70)
    
    preprocessor = YOLOPartPreprocessor(
        target_size=args.size,
        yolo_model=args.model,
        confidence_threshold=args.conf,
        iou_threshold=args.iou
    )
    
    # Выполнение обработки
    start_time = time.time()
    stats = preprocessor.process_dataset(
        src_path, dst_path, 
        test_mode=args.test, 
        test_limit=args.limit
    )
    total_time = time.time() - start_time
    
    # Финальная статистика
    print("\n" + "=" * 70)
    print("🏁 РЕЗУЛЬТАТЫ ОБРАБОТКИ")
    print("=" * 70)
    print(f"✅ Успешно обработано: {stats['successful']}")
    print(f"❌ Ошибок: {stats['failed']}")
    print(f"   - Ошибки чтения: {stats['read_errors']}")
    print(f"   - Ошибки записи: {stats['save_errors']}")
    print(f"🎯 Успешная детекция YOLO: {stats['detection_success']}")
    print(f"📊 Всего обработано: {stats['processed']}")
    
    if stats['processed'] > 0:
        success_rate = stats['successful'] / stats['processed'] * 100
        detection_rate = stats['detection_success'] / stats['processed'] * 100
        print(f"📈 Процент успеха: {success_rate:.1f}%")
        print(f"🎯 Процент детекции: {detection_rate:.1f}%")
        
        avg_time = stats['total_processing_time'] / stats['processed']
        print(f"⏱️  Среднее время на изображение: {avg_time:.2f} сек")
    
    print(f"⏱️  Общее время обработки: {total_time:.2f} сек")
    
    if stats['successful'] > 0:
        images_per_second = stats['successful'] / total_time
        print(f"⚡ Производительность: {images_per_second:.2f} изображений/сек")
    
    print(f"\n📁 Результаты сохранены в: {dst_path}")
    
    return 0

if __name__ == "__main__":
    sys.exit(main())