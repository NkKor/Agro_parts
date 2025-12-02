# src/api/routes.py
"""Маршруты (endpoints) для API"""
from flask import request, Response
from flask_restx import Namespace, Resource
import json
import numpy as np
import cv2 as cv
import logging
from pathlib import Path
import sys

# Настройка путей
project_root = Path(__file__).parent.parent.parent
utils_path = project_root / "utils"
sys.path.insert(0, str(project_root))
sys.path.insert(0, str(utils_path))

# Настройка логирования
logger = logging.getLogger(__name__)

# Создаем namespace для группировки endpoints
api_ns = Namespace('api', description='API methods for Details Recognition Service')

def get_search_engine():
    """Получение экземпляра поисковой системы"""
    try:
        from src.api.api_app import search_engine
        return search_engine
    except Exception as e:
        logger.error(f"Ошибка получения search_engine: {e}")
        return None

@api_ns.route('/predict')
@api_ns.doc(description="Method for class prediction based on embedding extraction")
class Predict(Resource):
    def post(self):
        """
        Endpoint для предсказания классов по загруженным изображениям.
        Ожидает multipart/form-data с полями файлов, например 'images'.
        """
        try:
            logger.info(f'Получен запрос /predict от {request.remote_addr}')

            # Получение файлов из запроса
            files = []
            if 'images' in request.files:
                files = request.files.getlist('images')
            else:
                for key in request.files:
                    files.extend(request.files.getlist(key))
            
            if not files:
                logger.warning('В запросе не найдены изображения')
                return Response(
                    response=json.dumps({'error': 'Not Found Image in request'}),
                    status=400,
                    mimetype="application/json"
                )

            logger.info(f'Получено {len(files)} изображений.')

            # Преобразование файлов в массивы NumPy
            images = []
            for file_storage in files[:10]:  # Ограничение на 10 изображений
                try:
                    if file_storage.filename == '':
                        logger.debug("Пропущен файл без имени.")
                        continue
                    file_bytes = file_storage.read()
                    image_array = np.frombuffer(file_bytes, np.uint8)
                    img_bgr = cv.imdecode(image_array, cv.IMREAD_COLOR)
                    
                    if img_bgr is not None:
                        images.append(img_bgr)
                        logger.debug(f"Изображение {file_storage.filename} успешно декодировано.")
                    else:
                        logger.warning(f"Не удалось декодировать изображение из {file_storage.filename}")
                except Exception as e:
                    logger.warning(f"Ошибка обработки файла {file_storage.filename}: {e}")
                    continue

            if len(images) == 0:
                logger.warning('Не удалось обработать ни одно изображение.')
                return Response(
                    response=json.dumps({'error': 'Could not process any images'}),
                    status=400,
                    mimetype="application/json"
                )

            # Вызов логики поиска
            search_engine = get_search_engine()
            if search_engine is None:
                logger.error('Поисковая система не инициализирована.')
                return Response(
                    response=json.dumps({'error': 'Search engine is not ready'}),
                    status=500,
                    mimetype="application/json"
                )

            result = search_engine.predict_search(images)

            # Формирование и отправка ответа
            response_json = json.dumps(result)
            logger.info('Отправка ответа.')
            return Response(
                response=response_json,
                status=200,
                mimetype="application/json"
            )

        except Exception as e:
            logger.error(f'Критическая ошибка в /predict: {e}', exc_info=True)
            return Response(
                response=json.dumps({'error': 'Internal server error'}),
                status=500,
                mimetype="application/json"
            )

@api_ns.route('/check')
@api_ns.doc(description="Check if a specific class exists")
class Check(Resource):
    @api_ns.param('class', 'The class ID to check', required=True)
    def get(self):
        """
        Endpoint для проверки существования класса.
        Ожидает обязательный параметр ?class=<class_id> в URL.
        """
        try:
            class_arg = request.args.get('class')
            if not class_arg:
                logger.warning('Параметр "class" обязателен для /check')
                return Response(
                    response=json.dumps({'error': 'Missing required parameter: class'}),
                    status=400,
                    mimetype="application/json"
                )
            
            logger.info(f'Получен запрос /check для класса: {class_arg}')

            # Получение поисковой системы
            search_engine = get_search_engine()
            if search_engine is None:
                return Response(
                    response=json.dumps({'error': 'Search engine is not ready'}),
                    status=500,
                    mimetype="application/json"
                )

            # Логика проверки
            if hasattr(search_engine, 'centroid_ids'):
                known_classes = search_engine.centroid_ids
                if hasattr(known_classes, 'tolist'):
                    known_classes = known_classes.tolist()
                
                if class_arg in known_classes:
                    message = {class_arg: "Class found in the database"}
                else:
                    message = {class_arg: "Class not found"}
            else:
                logger.warning("search_engine не имеет атрибута centroid_ids")
                message = {class_arg: "Unable to verify - database structure unknown"}

            response_message = json.dumps({'message': message})
            return Response(
                response=response_message,
                status=200,
                mimetype="application/json"
            )
        except Exception as e:
            logger.error(f'Ошибка в /check: {e}', exc_info=True)
            return Response(
                response=json.dumps({'error': 'Internal server error'}),
                status=500,
                mimetype="application/json"
            )

@api_ns.route('/get_info')
@api_ns.doc(description="Get general information about the service or available classes")
class GetInfo(Resource):
    def get(self):
        """
        Endpoint для получения общей информации.
        Возвращает список доступных классов.
        """
        try:
            logger.info('Получен запрос /get_info')

            # Получение поисковой системы
            search_engine = get_search_engine()
            if search_engine is None:
                return Response(
                    response=json.dumps({'error': 'Search engine is not ready'}),
                    status=500,
                    mimetype="application/json"
                )

            # Логика получения информации
            info_data = {}
            if hasattr(search_engine, 'centroid_ids'):
                all_known_classes = search_engine.centroid_ids
                if hasattr(all_known_classes, 'tolist'):
                    all_known_classes = all_known_classes.tolist()
                info_data['available_classes'] = all_known_classes
                info_data['total_classes'] = len(all_known_classes)
            else:
                logger.warning("search_engine не имеет атрибута centroid_ids")
                info_data['error'] = "Database structure unknown"
                info_data['available_classes'] = []

            response_message = json.dumps({'message': info_data})
            return Response(
                response=response_message,
                status=200,
                mimetype="application/json"
            )
        except Exception as e:
            logger.error(f'Ошибка в /get_info: {e}', exc_info=True)
            return Response(
                response=json.dumps({'error': 'Internal server error'}),
                status=500,
                mimetype="application/json"
            )