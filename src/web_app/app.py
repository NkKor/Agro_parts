# src/web_app/app.py
"""
Веб-приложение для поиска сельхоз-деталей по фото

"""

import os
import sys
from pathlib import Path
import logging
import json
import time
import numpy as np
import cv2 as cv
import torch
import torchvision.transforms as T
from flask import Flask, request, render_template, redirect, url_for, send_from_directory, send_file, jsonify
from werkzeug.utils import secure_filename
from datetime import datetime

# --- Настройка путей ---
# Определяем корень проекта
current_file = Path(__file__).resolve()
project_root = current_file.parent.parent.parent
utils_path = project_root / "utils"
src_path = project_root / "src"

# Добавляем пути в sys.path
sys.path.insert(0, str(project_root))
sys.path.insert(0, str(utils_path))
sys.path.insert(0, str(src_path))

# --- Создание директорий для логов ---
logs_dir = project_root / "logs"
logs_dir.mkdir(parents=True, exist_ok=True)

# --- Настройка TORCH_HOME ---
# Правильный путь к моделям относительно корня проекта
torch_home = project_root / 'data' / 'models'
torch_home.mkdir(parents=True, exist_ok=True)
os.environ['TORCH_HOME'] = str(torch_home)

# --- Настройка логирования ---
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s %(name)-12s %(levelname)-8s %(message)s',
    datefmt='%m-%d %H:%M',
    handlers=[
        logging.FileHandler(logs_dir / 'web_app.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

# --- Импорт модулей ---
try:
    import utils.config as config
    # from utils.utils_cv import find_largest_foreground_bbox, pad_bbox, center_square_crop, resize_high_quality
    # from src.models.encoder import ResNet50Encoder # Не используется напрямую в app.py, но нужна для SearchEngine
    from src.search.searcher import SearchEngine
except ImportError as e:
    logger.error(f" Ошибка импорта: {e}")
    # Убираем sys.exit(1), чтобы дать шанс отработать глобальной логике Gunicorn
    # sys.exit(1)


# Правильные пути к данным (относительные -> абсолютные относительно project_root)
TEMP_DIR =           project_root / getattr(config, 'TEMP_DIR',        Path('data/uploads'))
DATA_PROCESSED_DIR = project_root / getattr(config, 'PROC_DIR',        Path('data/processed'))
EMBED_DIR =          project_root / getattr(config, 'EMB_DIR',         Path('data/embeddings'))

# Приводим к абсолютным путям для работы
TEMP_DIR = TEMP_DIR.resolve()
DATA_PROCESSED_DIR = DATA_PROCESSED_DIR.resolve()
EMBED_DIR = EMBED_DIR.resolve()

# Создаем директории если не существуют
TEMP_DIR.mkdir(parents=True, exist_ok=True)
DATA_PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
EMBED_DIR.mkdir(parents=True, exist_ok=True)

logger.info(f" project_root: {project_root}")
logger.info(f" TEMP_DIR: {TEMP_DIR}")
logger.info(f" DATA_PROCESSED_DIR: {DATA_PROCESSED_DIR}")
logger.info(f" EMBED_DIR: {EMBED_DIR}")
logger.info(f" TEMP_DIR существует: {TEMP_DIR.exists()}")

# --- Глобальные переменные ---
ALLOWED_EXT = {".jpg", ".jpeg", ".png"}
search_engine = None

# --- Flask приложение ---
app = Flask(__name__,
           template_folder='templates')
app.config['MAX_CONTENT_LENGTH'] = 50 * 1024 * 1024  # 50MB

# --- Трансформации ---
transform = T.Compose([
    T.ToTensor(),
    T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

# --- Вспомогательные функции ---
def get_device(device_str: str = "auto") -> str:
    """Определение устройства для вычислений"""
    if device_str == "auto":
        if torch.cuda.is_available():
            return "cuda"
        elif torch.backends.mps.is_available() and torch.backends.mps.is_built():
            return "mps"
        else:
            return "cpu"
    return device_str

def allowed_file(filename: str) -> bool:
    """Проверка допустимого расширения файла"""
    return Path(filename).suffix.lower() in ALLOWED_EXT

def sample_example_for_part(part_id: str) -> str:
    """
    Возвращает путь к первому файлу в data/processed/<part_id> для отображения.
    Путь возвращается относительно DATA_PROCESSED_DIR.
    """
    try:
        logger.debug(f" Поиск примера для части: {part_id}")
        
        if not DATA_PROCESSED_DIR.exists():
            logger.error(f" Директория data/processed не найдена: {DATA_PROCESSED_DIR}")
            return None
        
        # Ищем директорию части
        part_dir = DATA_PROCESSED_DIR / str(part_id)
        
        # Если прямая директория не найдена, ищем рекурсивно
        if not part_dir.exists():
            for item in DATA_PROCESSED_DIR.iterdir():
                if item.is_dir() and item.name == str(part_id):
                    part_dir = item
                    break
            else:
                logger.warning(f" Директория для части {part_id} не найдена")
                return None
        
        if part_dir.exists() and part_dir.is_dir():
            # Ищем изображения
            image_files = []
            for ext in ['*.jpg', '*.jpeg', '*.png']:
                image_files.extend(part_dir.glob(ext))
            
            if image_files:
                first_image = image_files[0]
                # Возвращаем относительный путь от DATA_PROCESSED_DIR
                try:
                    relative_path = first_image.relative_to(DATA_PROCESSED_DIR)
                    path_str = str(relative_path).replace('\\', '/')
                    logger.debug(f" Найден пример: {path_str}")
                    return path_str
                except ValueError:
                    return f"{part_id}/{first_image.name}"
        else:
            logger.warning(f" Директория не существует или не является директорией: {part_dir}")
            return None
            
    except Exception as e:
        logger.error(f" Ошибка поиска примера для {part_id}: {e}")
        return None

def init_search_engine():
    """Инициализация поисковой системы"""
    global search_engine
    if search_engine is None:
        try:
            logger.info(" Инициализация поисковой системы...")
            search_engine = SearchEngine()
            logger.info(" Поисковая система инициализирована")
            return True
        except Exception as e:
            logger.error(f" Ошибка инициализации поисковой системы: {e}", exc_info=True)
            search_engine = None
            return False
    return True

# --- Инициализация модели на уровне модуля (для Gunicorn --preload) ---
# Эта часть выполняется один раз в родительском процессе Gunicorn
logger.info(" Gunicorn Preload: Инициализация поисковой системы на уровне модуля (WEB)...")
if not init_search_engine():
    logger.critical(" CRITICAL: Поисковая система WEB не была инициализирована.")
# ----------------------------------------------------------------------


# --- Маршруты ---
@app.route("/", methods=["GET"])
def index():
    """Главная страница - форма поиска"""
    return render_template("index.html")

@app.route("/search", methods=["POST"])
def search_route():
    """Маршрут для поиска по загруженным изображениям"""
    try:
        logger.info(" Получен запрос на поиск")
        
        # 1. Получение файлов
        files = request.files.getlist("images")
        files = [f for f in files if f and f.filename and allowed_file(f.filename)]
        
        if len(files) == 0:
            logger.warning(" В запросе не найдены изображения")
            return render_template("index.html", error="Не найдены изображения для поиска")
        
        logger.info(f" Получено {len(files)} изображений")
        
        # 2. Сохранение и чтение изображений
        saved_query_paths = []
        images = []
        
        # Убедимся, что TEMP_DIR существует
        TEMP_DIR.mkdir(parents=True, exist_ok=True)
        
        for file_storage in files[:10]:  # Ограничение на 10 изображений
            try:
                filename = secure_filename(file_storage.filename)
                if not filename:
                    continue
                
                tmp_path = TEMP_DIR / filename
                
                # Сохраняем файл
                file_storage.save(tmp_path)
                
                # Проверяем, что файл сохранился
                if tmp_path.exists():
                    saved_query_paths.append(str(tmp_path))
                else:
                    logger.error(f" Файл не сохранился: {tmp_path}")
                    continue
                
                # Чтение изображения
                img = cv.imread(str(tmp_path), cv.IMREAD_COLOR)
                if img is not None:
                    images.append(img)
                else:
                    logger.warning(f" Не удалось прочитать изображение {tmp_path}")
                    
            except Exception as e:
                logger.warning(f" Ошибка обработки файла {file_storage.filename}: {e}")
                continue
        
        if len(images) == 0:
            logger.error(" Не удалось обработать ни одно изображение")
            return render_template("index.html", error="Не удалось обработать ни одно изображение")
        
        logger.info(f" Обработано {len(images)} изображений, сохранено путей: {len(saved_query_paths)}")
        
        # 3. Поиск с помощью SearchEngine
        global search_engine # Убеждаемся, что используем глобальную переменную
        if search_engine is None:
            logger.error(" Поисковая система не инициализирована")
            return render_template("index.html", error="Поисковая система не готова")
        
        # Поиск топ-5 результатов
        results_dict = search_engine.predict_search(images, top_k=5)
        similarities = results_dict.get('similarities', {})
        
        logger.info(f" Получены результаты поиска: {len(similarities)} элементов")
        
        # 4. Подготовка данных для шаблона
        template_results = []
        # Преобразование словаря в отсортированный список, если searcher вернул не отсортированный
        if isinstance(similarities, dict):
             # Сортируем по значению схожести (по убыванию)
            sorted_similarities = sorted(similarities.items(), key=lambda item: item[1], reverse=True)
        else:
            sorted_similarities = similarities # Если уже список (part_id, similarity)
            
        for i, (part_id, similarity) in enumerate(sorted_similarities):
            sample_path = sample_example_for_part(part_id)
            template_results.append({
                "pid": str(part_id),
                "similarity": f"{similarity:.1f}%",
                "sample": sample_path,
                "is_best": i == 0,  # Первый в списке - лучший
                "rank": i + 1
            })
        
        # 5. Рендер шаблона с результатами
        # Заменяем абсолютные пути на URL для отображения
        query_img_urls = [
            url_for('uploaded_file', filename=Path(p).name) 
            for p in saved_query_paths
        ]
        
        return render_template(
            "index.html",
            query_imgs=query_img_urls,
            results=template_results,
            search_performed=True
        )
        
    except Exception as e:
        logger.error(f" Критическая ошибка в search_route: {e}", exc_info=True)
        return render_template("index.html", error="Внутренняя ошибка сервера")

@app.route("/data/uploads/<path:filename>")
def uploaded_file(filename):
    """Отдача временно загруженных файлов"""
    try:
        import urllib.parse
        decoded_filename = urllib.parse.unquote(filename)
        
        safe_filename = secure_filename(decoded_filename)
        if not safe_filename:
            return ("Invalid filename", 400)
        
        file_path = TEMP_DIR / safe_filename
        
        if file_path.exists() and file_path.is_file():
            return send_from_directory(str(TEMP_DIR), safe_filename)
        else:
            return ("File not found", 404)
            
    except Exception as e:
        logger.error(f" Ошибка отдачи загруженного файла {filename}: {e}", exc_info=True)
        return ("Server error", 500)

@app.route("/processed/<path:filepath>")
def processed_img(filepath):
    """Отдача обработанных изображений из data/processed"""
    try:
        if not DATA_PROCESSED_DIR.exists():
            logger.error(f" Директория data/processed не найдена: {DATA_PROCESSED_DIR}")
            return ("Data directory not found", 500)
        
        # Нормализация пути
        decoded_filepath = filepath.replace('\\', '/').strip('/')
        if not decoded_filepath:
            return ("Invalid path", 400)
        
        full_path = DATA_PROCESSED_DIR / decoded_filepath
        
        if full_path.exists() and full_path.is_file():
            if full_path.suffix.lower() in ['.jpg', '.jpeg', '.png', '.gif']:
                return send_from_directory(str(full_path.parent), full_path.name)
            else:
                return ("Not an image", 400)
        else:
            return ("File not found", 404)
            
    except Exception as e:
        logger.error(f" Ошибка отдачи processed файла {filepath}: {e}")
        return ("Server error", 500)

@app.route("/health")
def health_check():
    """Health check endpoint"""
    return jsonify({
        "status": "healthy",
        "search_engine_ready": search_engine is not None,
        "data_processed_dir": str(DATA_PROCESSED_DIR),
        "data_processed_dir_exists": DATA_PROCESSED_DIR.exists(),
        "tmp_dir": str(TEMP_DIR),
        "tmp_dir_exists": TEMP_DIR.exists()
    })


# --- Инициализация и запуск ---
if __name__ == "__main__":
    # Инициализация поисковой системы (для запуска без Gunicorn)
    if not init_search_engine():
        logger.error(" Критическая ошибка: не удалось инициализировать поисковую систему")
        sys.exit(1)
    
    # Запуск Flask приложения
    import utils.config as config_module
    host = getattr(config_module, 'WEB_HOST', '0.0.0.0')
    port = getattr(config_module, 'WEB_PORT', 5000)
    debug = getattr(config_module, 'WEB_DEBUG', False)
    
    logger.info(f" Запуск Flask веб-приложения на http://{host}:{port} (debug={debug})")
    app.run(host=host, port=port, debug=debug)