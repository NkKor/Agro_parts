#!/usr/bin/env python3
"""
Инструмент анализа качества датасета для проекта по распознаванию сельхоз-запчастей.
Анализирует:
- Качество изображений (размытость, экспозиция, контраст)
- Дубликаты изображений
- Сложность извлечения детали (фон, освещение, видимость)
- Смешанные категории (фото разных деталей в одном каталоге)
- Рекомендации по улучшению датасета
"""

import os
import sys
import cv2 as cv
import numpy as np
import pandas as pd
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Set
from collections import defaultdict
import hashlib
from PIL import Image
from tqdm import tqdm
import argparse
import warnings
warnings.filterwarnings('ignore')

def variance_of_laplacian(gray: np.ndarray) -> float:
    """Оценка размытости изображения"""
    return cv.Laplacian(gray, cv.CV_64F).var()

def overexposed_ratio(img: np.ndarray, thr: int = 245) -> float:
    """Доля очень светлых пикселей"""
    return (img >= thr).mean()

def find_largest_foreground_bbox(img_bgr: np.ndarray, min_area_ratio=0.12):
    """Нахождение bbox самой большой области переднего плана"""
    h, w = img_bgr.shape[:2]
    # усилить границы
    gray = cv.cvtColor(img_bgr, cv.COLOR_BGR2GRAY)
    blur = cv.GaussianBlur(gray, (5,5), 0)
    edges = cv.Canny(blur, 50, 150)
    # морфология
    kernel = cv.getStructuringElement(cv.MORPH_RECT, (5,5))
    edges = cv.morphologyEx(edges, cv.MORPH_CLOSE, kernel, iterations=2)
    # контуры
    cnts, _ = cv.findContours(edges, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)
    if not cnts:
        return None
    cnt = max(cnts, key=cv.contourArea)
    x,y,wc,hc = cv.boundingRect(cnt)
    area = wc*hc / (w*h)
    if area < min_area_ratio:
        return None
    return (x,y, x+wc, y+hc)

class DatasetAnalyzer:
    """Класс для анализа качества датасета изображений сельхоз-запчастей"""

    def __init__(self,
                 raw_dir: str = "data/raw",
                 target_size: int = 384,
                 yolo_model: str = 'yolov8n.pt',
                 confidence_threshold: float = 0.25):
        self.raw_dir = Path(raw_dir)
        self.target_size = target_size
        self.confidence_threshold = confidence_threshold
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'

        # Загрузка YOLO модели для детекции деталей
        print(f"Загрузка YOLO модели: {yolo_model} на устройстве {self.device}...")
        self.yolo_model = YOLO(yolo_model)
        self.yolo_model.to(self.device)

        # Результаты анализа
        self.analysis_results = {
            'part_stats': {},  # Статистика по каждой детали (ID)
            'duplicates': {},  # Дубликаты
            'quality_issues': {},  # Проблемы с качеством
            'mixed_categories': [],  # Смешанные категории
            'problematic_dirs': [],  # Проблемные каталоги
            'low_quality_images': [],  # Изображения низкого качества
            'background_issues': []  # Проблемы с фоном
        }

    def calculate_image_hash(self, image_path: Path) -> str:
        """Вычисление хэша изображения для поиска дубликатов"""
        try:
            with open(image_path, 'rb') as f:
                return hashlib.md5(f.read()).hexdigest()
        except Exception:
            return ""

    def calculate_phash(self, image_path: Path) -> Optional[str]:
        """Вычисление perceptual hash для поиска похожих изображений"""
        try:
            img = Image.open(image_path).convert('L')
            # Изменяем размер до 8x8
            img = img.resize((8, 8), Image.Resampling.LANCZOS)
            # Вычисляем среднее значение пикселей
            pixels = np.array(img.getdata()).reshape((8, 8))
            avg = pixels.mean()
            # Создаем битовую строку
            bits = 1 << np.arange(64, dtype=np.uint64)
            vals = (pixels.reshape(64) > avg) * bits
            phash = np.bitwise_or.reduce(vals)
            return format(phash, '016x')
        except Exception:
            return None

    def analyze_image_quality(self, image_path: Path) -> Dict:
        """Анализ качества изображения"""
        try:
            # Чтение изображения
            img = cv.imread(str(image_path))
            if img is None:
                return {'error': 'cannot_read'}

            h, w = img.shape[:2]
            gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)

            # Оценка размытости (Laplacian variance)
            blur_score = variance_of_laplacian(gray)

            # Оценка экспозиции (доля пересвеченных пикселей)
            overexposed = overexposed_ratio(img)

            # Оценка контраста (стандартное отклонение градиента)
            grad_x = cv.Sobel(gray, cv.CV_64F, 1, 0, ksize=3)
            grad_y = cv.Sobel(gray, cv.CV_64F, 0, 1, ksize=3)
            gradient_magnitude = np.sqrt(grad_x**2 + grad_y**2)
            contrast_score = np.std(gradient_magnitude)

            # Оценка яркости
            brightness = np.mean(gray)

            # Оценка насыщенности
            hsv = cv.cvtColor(img, cv.COLOR_BGR2HSV)
            saturation = np.mean(hsv[:,:,1])

            # Попытка детекции детали с помощью YOLO
            results = self.yolo_model(
                img,
                conf=self.confidence_threshold,
                device=self.device,
                verbose=False
            )

            detection_success = len(results) > 0 and results[0].boxes is not None and len(results[0].boxes) > 0
            detection_confidence = 0.0

            if detection_success:
                confidences = results[0].boxes.conf.cpu().numpy()
                if len(confidences) > 0:
                    detection_confidence = np.max(confidences)

            # Оценка видимости детали (поиск наибольшего переднего плана)
            bbox = find_largest_foreground_bbox(img)
            if bbox is not None:
                x1, y1, x2, y2 = bbox
                bbox_area = (x2 - x1) * (y2 - y1)
                total_area = w * h
                foreground_ratio = bbox_area / total_area
            else:
                foreground_ratio = 0.0

            return {
                'blur_score': blur_score,
                'overexposed_ratio': overexposed,
                'contrast_score': contrast_score,
                'brightness': brightness,
                'saturation': saturation,
                'detection_success': detection_success,
                'detection_confidence': detection_confidence,
                'foreground_ratio': foreground_ratio,
                'width': w,
                'height': h,
                'aspect_ratio': w / h if h != 0 else 0,
                'size_mb': image_path.stat().st_size / (1024 * 1024)
            }
        except Exception as e:
            return {'error': str(e)}

    def analyze_part_directory(self, part_dir: Path) -> Dict:
        """Анализ каталога одной детали"""
        images = [f for f in part_dir.iterdir()
                 if f.suffix.lower() in ['.jpg', '.jpeg', '.png']]

        if not images:
            return {
                'count': 0,
                'quality_scores': [],
                'hashes': {},
                'phashes': {},
                'stats': {}
            }

        quality_scores = []
        hashes = {}
        phashes = {}
        image_stats = {}

        for img_path in tqdm(images, desc=f"Анализ {part_dir.name}", leave=False):
            # Вычисление хэшей
            img_hash = self.calculate_image_hash(img_path)
            if img_hash:
                hashes[img_hash] = hashes.get(img_hash, []) + [img_path]

            phash = self.calculate_phash(img_path)
            if phash:
                phashes[phash] = phashes.get(phash, []) + [img_path]

            # Анализ качества
            quality = self.analyze_image_quality(img_path)
            image_stats[img_path.name] = quality

            if 'error' not in quality:
                quality_scores.append(quality)

        # Статистика по качеству
        if quality_scores:
            blur_scores = [q['blur_score'] for q in quality_scores if 'blur_score' in q]
            contrast_scores = [q['contrast_score'] for q in quality_scores if 'contrast_score' in q]
            brightness_scores = [q['brightness'] for q in quality_scores if 'brightness' in q]
            detection_success_rate = sum(1 for q in quality_scores if q.get('detection_success', False)) / len(quality_scores)

            stats = {
                'avg_blur': np.mean(blur_scores) if blur_scores else 0,
                'std_blur': np.std(blur_scores) if blur_scores else 0,
                'avg_contrast': np.mean(contrast_scores) if contrast_scores else 0,
                'avg_brightness': np.mean(brightness_scores) if brightness_scores else 0,
                'detection_success_rate': detection_success_rate,
                'avg_foreground_ratio': np.mean([q['foreground_ratio'] for q in quality_scores if 'foreground_ratio' in q]) if quality_scores else 0
            }
        else:
            stats = {
                'avg_blur': 0,
                'std_blur': 0,
                'avg_contrast': 0,
                'avg_brightness': 0,
                'detection_success_rate': 0,
                'avg_foreground_ratio': 0
            }

        return {
            'count': len(images),
            'quality_scores': quality_scores,
            'hashes': hashes,
            'phashes': phashes,
            'stats': stats,
            'image_stats': image_stats
        }

    def find_duplicates(self, all_part_stats: Dict) -> Dict:
        """Поиск дубликатов среди всех изображений"""
        all_hashes = {}
        all_phashes = {}

        for part_id, part_data in all_part_stats.items():
            for hash_val, paths in part_data['hashes'].items():
                if hash_val in all_hashes:
                    all_hashes[hash_val].extend(paths)
                else:
                    all_hashes[hash_val] = paths

            for phash_val, paths in part_data['phashes'].items():
                if phash_val in all_phashes:
                    all_phashes[phash_val].extend(paths)
                else:
                    all_phashes[phash_val] = paths

        # Фильтрация дубликатов (только если > 1 изображения с одинаковым хэшем)
        exact_duplicates = {k: v for k, v in all_hashes.items() if len(v) > 1}
        similar_images = {k: v for k, v in all_phashes.items() if len(v) > 1}

        return {
            'exact_duplicates': exact_duplicates,
            'similar_images': similar_images
        }

    def detect_mixed_categories(self, all_part_stats: Dict) -> List[str]:
        """Обнаружение каталогов с изображениями разных деталей (предположительно)"""
        mixed_categories = []

        for part_id, part_data in all_part_stats.items():
            if not part_data['quality_scores']:
                continue

            # Проверяем разнообразие характеристик изображений
            blur_scores = [q['blur_score'] for q in part_data['quality_scores'] if 'blur_score' in q]
            contrast_scores = [q['contrast_score'] for q in part_data['quality_scores'] if 'contrast_score' in q]

            if blur_scores and len(blur_scores) > 1:
                blur_std = np.std(blur_scores)
                blur_mean = np.mean(blur_scores)
                blur_cv = blur_std / blur_mean if blur_mean != 0 else 0

                # Если коэффициент вариации слишком высокий, возможно разные типы изображений
                if blur_cv > 2.0:  # Порог для определения неоднородности
                    mixed_categories.append(part_id)

        return mixed_categories

    def identify_problematic_directories(self, all_part_stats: Dict) -> List[str]:
        """Определение проблемных каталогов"""
        problematic_dirs = []

        for part_id, part_data in all_part_stats.items():
            issues = []

            # Проверка на слишком много дубликатов
            unique_images = len(part_data['hashes'])
            total_images = part_data['count']
            if total_images > 0 and unique_images / total_images < 0.5:  # Более 50% дубликатов
                issues.append(f"Слишком много дубликатов: {total_images - unique_images}/{total_images}")

            # Проверка на слишком маленькое количество изображений
            if total_images < 3:
                issues.append(f"Слишком мало изображений: {total_images}")

            # Проверка на низкий успех детекции
            if part_data['stats']['detection_success_rate'] < 0.3:  # Менее 30% детекций успешны
                issues.append(f"Низкий успех детекции: {part_data['stats']['detection_success_rate']:.2f}")

            if issues:
                problematic_dirs.append({
                    'part_id': part_id,
                    'issues': issues,
                    'total_images': total_images,
                    'unique_images': unique_images
                })

        return problematic_dirs

    def identify_low_quality_images(self, all_part_stats: Dict) -> List[Dict]:
        """Определение изображений низкого качества"""
        low_quality = []

        for part_id, part_data in all_part_stats.items():
            for img_name, img_stats in part_data['image_stats'].items():
                if 'error' in img_stats:
                    low_quality.append({
                        'part_id': part_id,
                        'image': img_name,
                        'issues': [f"Ошибка чтения: {img_stats['error']}"]
                    })
                    continue

                issues = []

                # Проверка на размытость
                if img_stats['blur_score'] < 100:  # Порог для размытости
                    issues.append(f"Слишком размыто: {img_stats['blur_score']:.2f}")

                # Проверка на низкий контраст
                if img_stats['contrast_score'] < 20:  # Порог для контраста
                    issues.append(f"Слишком низкий контраст: {img_stats['contrast_score']:.2f}")

                # Проверка на пересвет
                if img_stats['overexposed_ratio'] > 0.3:  # Более 30% пересвеченных пикселей
                    issues.append(f"Слишком много пересвета: {img_stats['overexposed_ratio']:.2f}")

                # Проверка на плохую детекцию
                if not img_stats['detection_success']:
                    issues.append("Не удалось детектировать деталь")

                # Проверка на малую видимость детали
                if img_stats['foreground_ratio'] < 0.1:  # Деталь занимает менее 10% изображения
                    issues.append(f"Деталь плохо видна: {img_stats['foreground_ratio']:.2f}")

                if issues:
                    low_quality.append({
                        'part_id': part_id,
                        'image': img_name,
                        'issues': issues,
                        'blur_score': img_stats['blur_score'],
                        'contrast_score': img_stats['contrast_score'],
                        'detection_success': img_stats['detection_success']
                    })

        return low_quality

    def identify_background_issues(self, all_part_stats: Dict) -> List[Dict]:
        """Определение проблем с фоном"""
        background_issues = []

        for part_id, part_data in all_part_stats.items():
            for img_name, img_stats in part_data['image_stats'].items():
                if 'error' in img_stats:
                    continue

                issues = []

                # Проверка на сливание фона и детали (низкая контрастность)
                if img_stats['contrast_score'] < 30:
                    issues.append(f"Слияние фона и детали (низкий контраст): {img_stats['contrast_score']:.2f}")

                # Проверка на слишком низкую видимость детали (мало переднего плана)
                if img_stats['foreground_ratio'] < 0.15:  # Деталь занимает менее 15% изображения
                    issues.append(f"Фон занимает слишком большую часть: {1 - img_stats['foreground_ratio']:.2f}")

                # Проверка на равномерный фон (слишком низкая вариативность)
                if img_stats['blur_score'] > 500 and img_stats['contrast_score'] < 15:
                    issues.append(f"Вероятно равномерный фон: размытость={img_stats['blur_score']:.2f}, контраст={img_stats['contrast_score']:.2f}")

                if issues:
                    background_issues.append({
                        'part_id': part_id,
                        'image': img_name,
                        'issues': issues,
                        'contrast_score': img_stats['contrast_score'],
                        'foreground_ratio': img_stats['foreground_ratio'],
                        'blur_score': img_stats['blur_score']
                    })

        return background_issues

    def run_analysis(self):
        """Запуск полного анализа датасета"""
        print("Начало анализа датасета...")
        print(f"Директория датасета: {self.raw_dir}")

        if not self.raw_dir.exists():
            print(f"Ошибка: директория {self.raw_dir} не найдена!")
            return

        # Получение списка каталогов деталей
        part_dirs = [d for d in self.raw_dir.iterdir() if d.is_dir()]
        print(f"Найдено {len(part_dirs)} каталогов деталей")

        # Анализ каждого каталога
        all_part_stats = {}
        for part_dir in tqdm(part_dirs, desc="Анализ каталогов"):
            part_id = part_dir.name
            print(f"Анализ каталога: {part_id}")
            part_stats = self.analyze_part_directory(part_dir)
            all_part_stats[part_id] = part_stats

        # Поиск дубликатов
        print("Поиск дубликатов...")
        duplicates = self.find_duplicates(all_part_stats)
        self.analysis_results['duplicates'] = duplicates

        # Обнаружение смешанных категорий
        print("Обнаружение смешанных категорий...")
        mixed_categories = self.detect_mixed_categories(all_part_stats)
        self.analysis_results['mixed_categories'] = mixed_categories

        # Определение проблемных каталогов
        print("Определение проблемных каталогов...")
        problematic_dirs = self.identify_problematic_directories(all_part_stats)
        self.analysis_results['problematic_dirs'] = problematic_dirs

        # Определение изображений низкого качества
        print("Определение изображений низкого качества...")
        low_quality_images = self.identify_low_quality_images(all_part_stats)
        self.analysis_results['low_quality_images'] = low_quality_images

        # Определение проблем с фоном
        print("Определение проблем с фоном...")
        background_issues = self.identify_background_issues(all_part_stats)
        self.analysis_results['background_issues'] = background_issues

        # Сохранение статистики по каждой детали
        self.analysis_results['part_stats'] = all_part_stats

        print("Анализ завершен!")

        return self.analysis_results

    def generate_report(self) -> str:
        """Генерация текстового отчета"""
        report = []
        report.append("=" * 80)
        report.append("АНАЛИЗ КАЧЕСТВА ДАТАСЕТА СЕЛЬХОЗ-ЗАПЧАСТЕЙ")
        report.append("=" * 80)
        report.append("")

        # Общая статистика
        total_parts = len(self.analysis_results['part_stats'])
        total_images = sum(part_data['count'] for part_data in self.analysis_results['part_stats'].values())
        report.append(f"Всего деталей: {total_parts}")
        report.append(f"Всего изображений: {total_images}")
        report.append("")

        # Дубликаты
        exact_dups = sum(len(v) for v in self.analysis_results['duplicates']['exact_duplicates'].values())
        similar_imgs = sum(len(v) for v in self.analysis_results['duplicates']['similar_images'].values())
        report.append(f"Точные дубликаты: {exact_dups} изображений")
        report.append(f"Похожие изображения: {similar_imgs} изображений")
        report.append("")

        # Смешанные категории
        report.append(f"Каталоги с потенциально смешанными категориями: {len(self.analysis_results['mixed_categories'])}")
        for cat in self.analysis_results['mixed_categories'][:10]:  # Показываем первые 10
            report.append(f"  - {cat}")
        if len(self.analysis_results['mixed_categories']) > 10:
            report.append(f"  ... и еще {len(self.analysis_results['mixed_categories']) - 10}")
        report.append("")

        # Проблемные каталоги
        report.append(f"Проблемные каталоги: {len(self.analysis_results['problematic_dirs'])}")
        for problem_dir in self.analysis_results['problematic_dirs']:
            report.append(f"  - {problem_dir['part_id']}: {problem_dir['total_images']} изображений ({problem_dir['unique_images']} уникальных)")
            for issue in problem_dir['issues']:
                report.append(f"    • {issue}")
        report.append("")

        # Изображения низкого качества
        report.append(f"Изображения низкого качества: {len(self.analysis_results['low_quality_images'])}")
        for i, img in enumerate(self.analysis_results['low_quality_images'][:10]):  # Показываем первые 10
            report.append(f"  - {img['part_id']}/{img['image']}")
            for issue in img['issues']:
                report.append(f"    • {issue}")
        if len(self.analysis_results['low_quality_images']) > 10:
            report.append(f"  ... и еще {len(self.analysis_results['low_quality_images']) - 10}")
        report.append("")

        # Проблемы с фоном
        report.append(f"Изображения с проблемами фона: {len(self.analysis_results['background_issues'])}")
        for i, img in enumerate(self.analysis_results['background_issues'][:10]):  # Показываем первые 10
            report.append(f"  - {img['part_id']}/{img['image']}")
            for issue in img['issues']:
                report.append(f"    • {issue}")
        if len(self.analysis_results['background_issues']) > 10:
            report.append(f"  ... и еще {len(self.analysis_results['background_issues']) - 10}")
        report.append("")

        # Рекомендации
        report.append("РЕКОМЕНДАЦИИ:")
        report.append("")

        if self.analysis_results['duplicates']['exact_duplicates']:
            report.append("1. Удалить точные дубликаты из датасета")

        if self.analysis_results['duplicates']['similar_images']:
            report.append("2. Рассмотреть удаление похожих изображений (>95% схожести)")

        if self.analysis_results['problematic_dirs']:
            report.append("3. Проверить и, возможно, удалить следующие каталоги:")
            for problem_dir in self.analysis_results['problematic_dirs'][:5]:
                report.append(f"   - {problem_dir['part_id']}")

        if self.analysis_results['low_quality_images']:
            report.append("4. Удалить изображения низкого качества (размытые, темные, пересвеченные)")

        if self.analysis_results['background_issues']:
            report.append("5. Улучшить фон на проблемных изображениях или удалить их")
            report.append("   - Рекомендуется использовать фон, контрастирующий с деталью")
            report.append("   - Деталь должна занимать значительную часть изображения")

        if self.analysis_results['mixed_categories']:
            report.append("6. Проверить каталоги с потенциально смешанными категориями:")
            for cat in self.analysis_results['mixed_categories'][:5]:
                report.append(f"   - {cat}")

        report.append("")
        report.append("Дополнительные рекомендации:")
        report.append("- Использовать разнообразные ракурсы для каждой детали")
        report.append("- Обеспечить равномерное освещение без бликов")
        report.append("- Сделать деталь центральным элементом изображения")
        report.append("- Поддерживать разрешение не менее 300x300 пикселей")
        report.append("- Использовать однородный фон, контрастирующий с деталью")

        return "\n".join(report)

    def save_detailed_report(self, output_file: str = "dataset_analysis_report.txt"):
        """Сохранение подробного отчета в файл"""
        report = self.generate_report()

        with open(output_file, 'w', encoding='utf-8') as f:
            f.write(report)

        print(f"Подробный отчет сохранен в: {output_file}")


def main():
    parser = argparse.ArgumentParser(
        description="Анализ качества датасета сельхоз-запчастей",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Примеры:
  %(prog)s --raw data/raw
  %(prog)s --raw data/raw --output analysis_report.txt
        """
    )

    parser.add_argument("--raw", type=str, default="data/raw",
                       help="Путь к исходным изображениям (default: data/raw)")
    parser.add_argument("--output", type=str, default="dataset_analysis_report.txt",
                       help="Файл для сохранения отчета (default: dataset_analysis_report.txt)")
    parser.add_argument("--model", type=str, default='yolov8n.pt',
                       help="YOLO модель для детекции (default: yolov8n.pt)")
    parser.add_argument("--conf", type=float, default=0.25,
                       help="Порог уверенности для YOLO (default: 0.25)")

    args = parser.parse_args()

    print("Инициализация анализатора датасета...")
    analyzer = DatasetAnalyzer(
        raw_dir=args.raw,
        yolo_model=args.model,
        confidence_threshold=args.conf
    )

    # Запуск анализа
    results = analyzer.run_analysis()

    # Вывод отчета в консоль
    report = analyzer.generate_report()
    print(report)

    # Сохранение отчета в файл
    analyzer.save_detailed_report(args.output)

    return 0


if __name__ == "__main__":
    sys.exit(main())