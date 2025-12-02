# src/api/api_app.py
"""Flask API с Flask-RESTX для документации Swagger"""
import os
import sys
from pathlib import Path
import logging
import json
import numpy as np
import cv2 as cv
from flask import Flask, request, send_from_directory
from flask_restx import Api, Resource, fields
from werkzeug.utils import secure_filename
import urllib.parse

# --- Настройка путей ---
current_file = Path(__file__).resolve()
project_root = current_file.parent.parent.parent
utils_path = project_root / "utils"
src_path = project_root / "src"

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
        logging.FileHandler(logs_dir / 'api.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

# --- Импорт конфигурации и модулей ---
try:
    import utils.config as config
    from src.search.searcher import SearchEngine
except ImportError as e:
    logger.error(f" Ошибка импорта: {e}")
    # Важно: Не используйте raise здесь, если init_search_engine() может отработать
    # в глобальной области видимости. Для чистоты:
    # raise 

# --- Глобальные переменные ---
search_engine = None
TEMP_DIR =              project_root / getattr(config, 'TEMP_DIR', Path('data/uploads'))
DATA_PROCESSED_DIR =    project_root / getattr(config, 'PROC_DIR', Path('data/processed'))
ALLOWED_EXT = {".jpg", ".jpeg", ".png"}

# Приводим к абсолютным путям
TEMP_DIR = TEMP_DIR.resolve()
DATA_PROCESSED_DIR = DATA_PROCESSED_DIR.resolve()

# Создаем директории если не существуют
TEMP_DIR.mkdir(parents=True, exist_ok=True)
DATA_PROCESSED_DIR.mkdir(parents=True, exist_ok=True)

logger.info(f" TEMP_DIR: {TEMP_DIR}")
logger.info(f" DATA_PROCESSED_DIR: {DATA_PROCESSED_DIR}")

# --- Вспомогательные функции ---
def init_search_engine():
    """Инициализация поисковой системы (общей для всего проекта)"""
    global search_engine
    if search_engine is None:
        try:
            logger.info(" Инициализация общей поисковой системы...")
            search_engine = SearchEngine()
            logger.info(" Поисковая система инициализирована")
            return True
        except Exception as e:
            logger.error(f" Ошибка инициализации поисковой системы: {e}", exc_info=True)
            search_engine = None
            return False
    return True

def sample_example_for_part(part_id: str) -> str:
    """Поиск примера изображения для части"""
    try:
        if not DATA_PROCESSED_DIR.exists():
            logger.warning(f" Директория processed не найдена: {DATA_PROCESSED_DIR}")
            return None
        
        part_dir = DATA_PROCESSED_DIR / str(part_id)
        if not part_dir.exists():
            for item in DATA_PROCESSED_DIR.iterdir():
                if item.is_dir() and item.name == str(part_id):
                    part_dir = item
                    break
            else:
                logger.warning(f" Директория для части {part_id} не найдена")
                return None
        
        if part_dir.exists() and part_dir.is_dir():
            for ext in ['*.jpg', '*.jpeg', '*.png']:
                image_files = list(part_dir.glob(ext))
                if image_files:
                    first_image = image_files[0]
                    try:
                        relative_path = first_image.relative_to(DATA_PROCESSED_DIR)
                        return str(relative_path).replace('\\', '/')
                    except ValueError:
                        return f"{part_id}/{first_image.name}"
        
        return None
    except Exception as e:
        logger.error(f" Ошибка поиска примера для {part_id}: {e}")
        return None

def allowed_file(filename: str) -> bool:
    """Проверка допустимого расширения файла"""
    return Path(filename).suffix.lower() in ALLOWED_EXT

def process_uploaded_files(files) -> list:
    """Обработка загруженных файлов и преобразование в изображения"""
    images = []
    for file_storage in files[:10]:
        try:
            if not file_storage.filename or not allowed_file(file_storage.filename):
                continue
                
            file_bytes = file_storage.read()
            image_array = np.frombuffer(file_bytes, np.uint8)
            img_bgr = cv.imdecode(image_array, cv.IMREAD_COLOR)
            
            if img_bgr is not None:
                images.append(img_bgr)
                logger.debug(f" Изображение {file_storage.filename} успешно декодировано")
            else:
                logger.warning(f" Не удалось декодировать изображение из {file_storage.filename}")
        except Exception as e:
            logger.warning(f" Ошибка обработки файла {file_storage.filename}: {e}")
            continue
    
    return images

# --- Инициализация модели на уровне модуля (для Gunicorn --preload) ---
# Эта часть выполняется один раз в родительском процессе Gunicorn.
logger.info(" Gunicorn Preload: Инициализация поисковой системы на уровне модуля...")
if not init_search_engine():
    logger.critical(" CRITICAL: Поисковая система не была инициализирована. API будет недоступен.")
# ----------------------------------------------------------------------


# --- Создание Flask приложения и API ---
app = Flask(__name__)
app.config['MAX_CONTENT_LENGTH'] = 50 * 1024 * 1024  # 50MB

# Настройка Flask-RESTX
api = Api(
    app,
    version='1.0',
    title='Agricultural Parts Recognition API',
    description='API для поиска сельхоз-деталей по фотографиям',
    doc='/swagger-ui/',
    prefix='/api'
)

# Модели для документации
predict_model = api.model('PredictResponse', {
    'pred_classes': fields.List(fields.String, description='Предсказанные классы'),
    'rank': fields.Raw(description='Ранги классов'),
    'more_possible_classes': fields.List(fields.String, description='Более вероятные классы'),
    'another_possible_classes': fields.List(fields.String, description='Другие возможные классы'),
    'similarities': fields.Raw(description='Проценты схожести'),
    'results': fields.List(fields.Raw, description='Детальные результаты')
})

check_model = api.model('CheckResponse', {
    'message': fields.Raw(description='Результат проверки')
})

info_model = api.model('InfoResponse', {
    'message': fields.Raw(description='Информация о системе')
})

error_model = api.model('Error', {
    'error': fields.String(description='Сообщение об ошибке')
})

# Namespace для API
ns = api.namespace('', description='Операции поиска деталей')

@ns.route('/predict')
class Predict(Resource):
    @ns.doc('predict')
    @ns.param('top_k', 'Количество возвращаемых результатов (1-1000)', _in='query')
    @ns.response(200, 'Успешный поиск', predict_model)
    @ns.response(400, 'Ошибка в запросе', error_model)
    @ns.response(500, 'Внутренняя ошибка сервера', error_model)
    def post(self):
        """Поиск деталей по загруженным изображениям"""
        try:
            logger.info(f' Получен запрос /api/predict от {request.remote_addr}')
            
            # Получение параметра top_k
            top_k_param = request.args.get('top_k', type=int, default=5)
            if not (1 <= top_k_param <= 1000):
                logger.warning(f"Недопустимое значение top_k: {top_k_param}")
                return {'error': 'Parameter top_k must be between 1 and 1000'}, 400
            
            # Получение файлов
            files = request.files.getlist("images")
            files = [f for f in files if f and f.filename and allowed_file(f.filename)]
            
            if len(files) == 0:
                logger.warning('В запросе не найдены изображения')
                return {'error': 'Not Found Image in request'}, 400

            logger.info(f"Получено {len(files)} изображений.")

            # Обработка изображений
            images = []
            saved_query_paths = []
            
            for file_storage in files[:10]:  # Ограничение на 10 изображений
                try:
                    filename = secure_filename(file_storage.filename)
                    if not filename:
                        continue
                    
                    tmp_path = TEMP_DIR / filename
                    file_storage.save(tmp_path)
                    saved_query_paths.append(str(tmp_path))
                    
                    # Чтение изображения
                    img = cv.imread(str(tmp_path), cv.IMREAD_COLOR)
                    if img is not None:
                        images.append(img)
                        logger.debug(f" Изображение {filename} успешно декодировано.")
                    else:
                        logger.warning(f" Не удалось декодировать изображение из {tmp_path}")
                except Exception as e:
                    logger.warning(f" Ошибка обработки файла {file_storage.filename}: {e}")
                    continue

            if len(images) == 0:
                logger.error("Не удалось обработать ни одно изображение.")
                return {'error': 'Could not process any images'}, 400

            # Проверка поисковой системы
            global search_engine
            if search_engine is None:
                logger.error("Поисковая система не инициализирована.")
                return {'error': 'Search engine is not ready'}, 500

            # Вызов логики поиска
            try:
                results_dict = search_engine.predict_search(images, top_k=top_k_param)
                logger.info(" Поиск завершен успешно.")
            except Exception as e:
                logger.error(f" Ошибка поиска в predict_search: {e}", exc_info=True)
                return {'error': f'Search failed: {str(e)}'}, 500

            # --- ИСПРАВЛЕНИЕ ФОРМАТА ОТВЕТА БЕЗ ИЗМЕНЕНИЯ SEARCHER ---
            
            # Получаем pred_classes и similarities из results_dict
            pred_classes = results_dict.get('pred_classes', [])
            similarities_raw = results_dict.get('similarities', {})
            
            # Формируем словарь с процентами похожести
            similarities = {}
            if isinstance(similarities_raw, dict):
                # searcher возвращает {part_id: similarity_value}
                for part_id, similarity_value in similarities_raw.items():
                    if isinstance(similarity_value, (int, float)):
                        # Преобразуем в строку с правильной кодировкой
                        part_id_str = str(part_id)
                        similarities[part_id_str] = float(similarity_value)
                    else:
                        part_id_str = str(part_id)
                        similarities[part_id_str] = 0.0
            elif isinstance(similarities_raw, list):
                # searcher может возвращать список [(part_id, similarity_value), ...]
                for item in similarities_raw:
                    if isinstance(item, (tuple, list)) and len(item) >= 2:
                        part_id, similarity_value = item[0], item[1]
                        if isinstance(similarity_value, (int, float)):
                            part_id_str = str(part_id)
                            similarities[part_id_str] = float(similarity_value)
                        else:
                            part_id_str = str(part_id)
                            similarities[part_id_str] = 0.0
            
            # Убедимся, что pred_classes - это список строк с правильной кодировкой
            if not isinstance(pred_classes, list):
                pred_classes = []
            pred_classes = [str(cls) for cls in pred_classes]
            
            # Формируем ответ в нужном формате
            response_data = {
                'pred_classes': pred_classes,
                'similarities': similarities
            }
            
            logger.info(f" Отправка ответа с {len(pred_classes)} результатами")
            
            # Возвращаем ответ с правильной кодировкой
            import json
            response_json = json.dumps(response_data, ensure_ascii=False)
            from flask import Response
            return Response(
                response=response_json,
                status=200,
                mimetype="application/json"
            )

        except Exception as e:
            logger.error(f" Критическая ошибка в /predict: {e}", exc_info=True)
            return {'error': 'Internal server error'}, 500
@ns.route('/check')
class Check(Resource):
    @ns.doc('check_class')
    @ns.param('class', 'ID класса для проверки', required=True)
    @ns.response(200, 'Результат проверки')
    @ns.response(400, 'Отсутствует параметр class')
    @ns.response(500, 'Внутренняя ошибка сервера')
    def get(self):
        """Проверка существования класса в базе"""
        try:
            class_arg = request.args.get('class')
            if not class_arg:
                logger.warning('Параметр "class" обязателен для /api/check')
                return {'error': 'Missing required parameter: class'}, 400
            
            logger.info(f' Получен запрос /api/check для класса: {class_arg}')
            
            global search_engine
            if search_engine is None:
                logger.error('Поисковая система не инициализирована')
                return {'error': 'Search engine is not ready'}, 500
            
            # Проверка существования класса через centroid_ids
            if hasattr(search_engine, 'centroid_ids'):
                known_classes = search_engine.centroid_ids
                if hasattr(known_classes, 'tolist'):
                    known_classes = known_classes.tolist()
                
                if class_arg in known_classes:
                    message = {class_arg: "Class found"}
                else:
                    message = {class_arg: "Class not found"}
            else:
                message = {class_arg: "Unable to verify - database structure unknown"}
            
            # Исправленный формат ответа с правильной кодировкой
            import json
            response_data = {'message': message}
            response_json = json.dumps(response_data, ensure_ascii=False)
            from flask import Response
            return Response(
                response=response_json,
                status=200,
                mimetype="application/json"
            )
            
        except Exception as e:
            logger.error(f' Ошибка в /api/check: {e}', exc_info=True)
            # Исправленный формат ошибки с правильной кодировкой
            import json
            error_data = {'error': 'Internal server error'}
            error_json = json.dumps(error_data, ensure_ascii=False)
            from flask import Response
            return Response(
                response=error_json,
                status=500,
                mimetype="application/json"
            )

@ns.route('/get_info')
class GetInfo(Resource):
    @ns.doc('get_info')
    @ns.response(200, 'Информация о системе', info_model)
    @ns.response(500, 'Внутренняя ошибка сервера', error_model)
    def get(self):
        """Получение информации о доступных классах"""
        try:
            logger.info(' Получен запрос /api/get_info')
            
            global search_engine
            if search_engine is None:
                logger.error('Поисковая система не инициализирована')
                return {'error': 'Search engine is not ready'}, 500
            
            # Получение информации через centroid_ids
            info_data = {}
            if hasattr(search_engine, 'centroid_ids'):
                all_known_classes = search_engine.centroid_ids
                if hasattr(all_known_classes, 'tolist'):
                    all_known_classes = all_known_classes.tolist()
                info_data['available_classes'] = all_known_classes
                info_data['total_classes'] = len(all_known_classes)
            else:
                info_data['error'] = "Database structure unknown"
                info_data['available_classes'] = []
            
            return {'message': info_data}
            
        except Exception as e:
            logger.error(f' Ошибка в /api/get_info: {e}', exc_info=True)
            return {'error': 'Internal server error'}, 500

# --- Базовые маршруты Flask (вне RESTX) ---

@app.route('/data/uploads/<path:filename>')
def uploaded_file(filename):
    """Отдача временно загруженных файлов"""
    try:
        decoded_filename = urllib.parse.unquote(filename)
        safe_filename = secure_filename(decoded_filename)
        
        if not safe_filename:
            return {'error': 'Invalid filename'}, 400
        
        file_path = TEMP_DIR / safe_filename
        if file_path.exists() and file_path.is_file():
            return send_from_directory(str(TEMP_DIR), safe_filename)
        else:
            return {'error': 'File not found'}, 404
    except Exception as e:
        logger.error(f" Ошибка отдачи файла {filename}: {e}")
        return {'error': 'Server error'}, 500

@app.route('/processed/<path:filepath>')
def processed_img(filepath):
    """Отдача обработанных изображений из data/processed"""
    try:
        if not DATA_PROCESSED_DIR.exists():
            logger.error(f" Директория processed не найдена: {DATA_PROCESSED_DIR}")
            return {'error': 'Data directory not found'}, 500
        
        decoded_filepath = urllib.parse.unquote(filepath)
        normalized_path = decoded_filepath.replace('\\', '/').strip('/')
        
        if not normalized_path:
            return {'error': 'Invalid path'}, 400
        
        full_path = DATA_PROCESSED_DIR / normalized_path
        
        if full_path.exists() and full_path.is_file():
            if full_path.suffix.lower() in ['.jpg', '.jpeg', '.png', '.gif']:
                return send_from_directory(str(full_path.parent), full_path.name)
            else:
                return {'error': 'Not an image'}, 400
        else:
            return {'error': 'File not found'}, 404
    except Exception as e:
        logger.error(f" Ошибка отдачи processed файла {filepath}: {e}")
        return {'error': 'Server error'}, 500

@app.route('/health')
def health_check():
    """Health check endpoint"""
    global search_engine
    return {
        'status': 'healthy',
        'search_engine_ready': search_engine is not None,
        'data_processed_dir': str(DATA_PROCESSED_DIR),
        'data_processed_dir_exists': DATA_PROCESSED_DIR.exists(),
        'total_classes': len(search_engine.centroid_ids) if search_engine and hasattr(search_engine, 'centroid_ids') else 0
    }

# --- Обработчики ошибок ---
@app.errorhandler(404)
def not_found(error):
    return {'error': 'Not found'}, 404

@app.errorhandler(500)
def internal_error(error):
    return {'error': 'Internal server error'}, 500

# --- Инициализация и запуск ---
if __name__ == '__main__':
    
    # Инициализация поисковой системы (для запуска без Gunicorn)
    if not init_search_engine():
        logger.error(" Критическая ошибка: не удалось инициализировать поисковую систему")
        sys.exit(1)
    
    # Запуск Flask-сервера
    import utils.config as config_module
    host = getattr(config_module, 'API_HOST', '127.0.0.1')
    port = getattr(config_module, 'API_PORT', 3887)
    debug = getattr(config_module, 'DEBUG', False)
    
    logger.info(f" Запуск Flask API на http://{host}:{port} (debug={debug})")
    logger.info(f" Swagger документация доступна по адресу: http://{host}:{port}/swagger-ui/")
    app.run(host=host, port=port, debug=debug)