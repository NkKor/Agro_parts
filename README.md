# 🌾 AgroParts Recognition Service

Сервис распознавания и поиска сельскохозяйственных запчастей по фотографиям.

## Описание

Этот проект предоставляет систему для идентификации и поиска сельхоз-запчастей на основе изображений. Пользователи могут загрузить фотографии сломанной детали, и система предложит наиболее похожие запчасти из каталога.

Проект состоит из:
- **Offline Pipeline:** Предобработка изображений, извлечение эмбеддингов, построение центроидов и FAISS-индексов.
- **Веб-приложение:** Интерфейс для ручного тестирования и демонстрации.
- **REST API:** Программный интерфейс для интеграции с другими системами.

### Требования

- Python 3.10 (мин 3.9)
- Docker & Docker-compose
- CUDA 11.8+ (опционально, для ускорения на GPU) - при работе с CPU необходимо внести изменения в конфигурацию.
- ОЗУ 18 ГБ+ (для загрузки индексов поиска необходимо порядка 18 ГБ) - для каждого контейнера
- Место на диске: не менее 200 ГБ (данные + вектора + индексы)

### 📁 Структура проекта
project/
├── data/
│   ├── uploads/           # Загруженные в веб-сервисе фотографии
│   ├── raw/               # Исходные необработанные изображения
│   ├── processed/         # Обработанные изображения (после preprocess.py)
│   ├── embeddings/        # Извлеченные эмбеддинги изображений
│   │   ├── per_image.npy
│   │   ├── part_ids.npy
│   │   └── metadata.json
│   ├── centroids/         # Центроиды деталей
│   │   ├── per_part.npy
│   │   ├── part_names.npy
│   │   └── metadata.json
│   └── indexes/           # FAISS индексы
│       ├── image_index.faiss
│       ├── centroid_index.faiss
│       └── metadata.json
├── src/ # Исходный код
│   ├── models/ # Модель ResNet50Encoder
│   ├── search/ # Логика поиска (SearchEngine)
│   │   └── searcher.py # Поисковая логика (используется API)
│   ├── api/ # API
│   │   ├── Dockerfile      # файл docker для сборки образа API 
│   │   └── api_app.py      # Flask приложение + API инициализация + Маршруты + Документация
│   ├── web_app/ # Веб-интерфейс (Flask)
│   │   ├── Dockerfile      # файл docker для сборки образа веб-сервиса 
│   │   ├── app.py
│   │   └── templates
│   │       └── index.html
│   ├── dataset.py # Датасет PartsDataset
│   │   # Скрипты для запуска этапов обработки изображеий и извлечения векторов
│   ├── preprocess.py # Предобработка изображений (YOLO + обрезка)  # - 1 этап обработки
│   ├── extract_embeddings.py # Извлечение эмбеддингов              # - 2 этап обработки
│   ├── build_centroids.py # Построение центроидов                  # - 3 этап обработки
│   └── build_index.py # Построение FAISS-индексов                  # - 4 этап обработки
├── utils/ # Конфигурации и общие утилиты
│   ├── config.py # Основной конфигурационный файл
│   └── utils_cv.py # Утилиты компьютерного зрения
├── logs/ # Логи приложения
│   ├── searcher.log
│   ├── appi.log
│   └── web_app.log
├── docker-compose.yml # Docker Compose конфигурация
├── requirements.txt # Зависимости Python
└── README.md # Этот файл


## Этапы обработки данных

> **Примечание:** Все пути к данным и конфигурации определяются в `utils/config.py`. Убедитесь, что они установлены правильно.


### 1 - Препроцессинг датасета, детекция деталей на фото, центрирование кадра, приведение к единому размеру, при необходимости повышение контрастности

Обрезает лишний фон, центрирует деталь и приводит изображения к единому формату.

1. Обычная обработка (пропуск существующих):
python src/preprocess.py --src data/raw --dst data/processed --size 512

2. Принудительная обработка всех файлов:
python src/preprocess.py --src data/raw --dst data/processed --size 512 --force

3. Тестовая обработка новых файлов:
python src/preprocess.py --src data/raw --dst data/processed --size 512 --test --limit 50

--skip-existing (параметр по умолчанию) - пропускать существующие фото в датасете, позволяет быстро добавлять новые данные
--force - обработать все заново, включая все существующие фото в data/processed

Все эти размеры поддерживаются ResNet50:
sizes = [224, 256, 288, 320, 384, 448, 512, 640, 768]

Производительность vs Качество:
224×224 - быстрее, но может терять детали
384×384 - оптимальный баланс (рекомендуется)
512×512 - максимальное качество


### 2 - Извлечение признаков изображений

-- Извлечение эмбеддингов

python src/extract_embeddings.py --src data/processed --out data/embeddings по умолчанию обрабатывает только новые изображения
python src/extract_embeddings.py --src data/processed --out data/embeddings --force принудительно обновляет все вектора
python src/extract_embeddings.py --src data/processed --out data/embeddings --device cuda:0 принудительный выбор CUDA

### 3. Построение центроидов для каждой детали

python src/build_centroids.py --embeddings data/embeddings --out data/centroids по умолчанию не пересоздаёт, если файл уже есть
python src/build_centroids.py --embeddings data/embeddings --out data/centroids --force  для принудительного пересоздания

### 4. Построение базы индексов для поисковой машины FAISS

python src/build_index.py --embeddings data/embeddings --centroids data/centroids --out data/indexes по умолчанию не пересоздаёт, если файл уже есть
python src/build_index.py --embeddings data/embeddings --centroids data/centroids --out data/indexes --force  для принудительного пересоздания


## При добавлении новых фото выполнить последовательно (в режиме обновления, без --force):
1 - Препроцессинг фото
2 - Извлечение эмдеддингов
3 - Построение центроидов
4 - Созданиие индексов



## 3 - Поиск (тест / прод), веб приложение, Работа с API 
### Вариант 1: Использование Docker (Рекомендуется)

#### Сборка образа 
- Сборка образа
Рекомендуется выполнять сборку без использования кэша
docker-compose build --no-cache 

- Проверка состояния
docker-compose ps

#### Запуск API и веб-приложения
- Оба контейнера
docker-compose up -d

- Только API
docker-compose up -d api

- Только веб-приложение
docker-compose up -d web


#### Просмотр логов
- Логи всех сервисов
docker-compose logs -f

- Логи конкретного сервиса
docker-compose logs -f api
docker-compose logs -f web

#### Тестирование работы контейнеров

- Health check API
curl http://localhost:3887/health

- Проверка существования класса, в данном случае "1151"
curl -X GET "http://127.0.0.1:3887/api/check?class=1151"

- Swagger UI API
http://localhost:3887/swagger-ui/ Там ты найдешь интерактивную документацию API со всеми endpoint'ами и можешь тестировать их прямо в браузере.

- Predict API
curl -X POST "http://localhost:3887/api/predict" -F "images=@test.jpg"

Доступ к сервисам:
API (Swagger UI): http://localhost:3887/swagger-ui
Веб-приложение: http://localhost:5000
Health check: http://localhost:3887/health

Поиск по одному или нескольким изображениям.
curl -X POST "http://127.0.0.1:3887/api/predict?top_k=5" -F "images=@<path-to-project>\data\uploads\test.jpg"

например:

-------win---------
curl -X POST "http://127.0.0.1:3887/api/predict?top_k=5" -F "images=@S:\code\vscode\agro_parts\Agro_parts\data\uploads\test1.jpg"

-------nix---------
curl -X POST "http://127.0.0.1:3887/api/predict?top_k=5" -F "images=@/media/koraki/Data/code/Agro-parts/data/uploads/test1.jpg"


Основной поиск (predict) - несколько файлов:
curl -X POST "http://127.0.0.1:3887/api/predict" \
     -F "images=@/path/to/your/test_image1.jpg" \
     -F "images=@/path/to/your/test_image2.jpg" \
     -F "images=@/path/to/your/test_image3.jpg"


Поиск с указанием количества результатов:
curl -X POST "http://localhost:3887/api/predict?top_k=10" \
     -F "images=@/path/to/your/test_image1.jpg"


Формат ответа API:

{
    "pred_classes": ["BБА000017255", "CБА000004731", "C8162", "C2884", "CБА000030417", "BБА000018447", "CБА000013480", "BБА000004882", "CН0000518", "CБА000001998"], 
    "similarities": {"BБА000017255": 90.82548156900766, "CБА000004731": 88.85158251852575, "C8162": 88.61630876836881, "C2884": 88.51690886108962, "CБА000030417": 88.47644313435119, "BБА000018447": 88.47227906044473, "CБА000013480": 88.40876448860327, "BБА000004882": 88.21466763086941, "CН0000518": 88.0293601576541, "CБА000001998": 87.96806554734347
    }
}

Остановка сервисов:
docker-compose down

Полная очистка контейнеров с удалением образов

docker-compose down -v --remove-orphans
docker rmi $(docker images -q) -f
docker system prune -a --volumes -f

пересборка
docker-compose build --no-cache
docker-compose build api web --no-cache - сборка контейнеров поименно
docker-compose up -d api - только апи и документация, без веб
docker-compose logs -f api
docker-compose up -d web - только веб поиск, без апи
docker-compose logs -f web

### Вариант 2: Запуск вручную

Запуск API:
#### Используя Gunicorn (для продакшена)
CMD gunicorn --bind 0.0.0.0:3887 --workers 1 --timeout ${GUNICORN_TIMEOUT} --keep-alive 5 --log-level info src.api.api_app:app

Запуск Web:
#### Используя Gunicorn (для продакшена)
CMD gunicorn --bind 0.0.0.0:5000 --workers 1 --timeout ${GUNICORN_TIMEOUT} --log-level info src.web_app.app:app

#### Или из консоли, используя встроенный Flask-сервер (для разработки и тестирования)

- Перейди в корень проекта
синтаксис linux:
cd /media/vscode/Agro_parts 
пример для Windows:
cd S:\code\vscode\Agro_parts


Запуск api
python src/api/api_app.py
API будет доступен по адресу: http://127.0.0.1:3887

Запуск веб-приложения:
python src/web_app/app.py
Web приложение будет доступно по адресу: http://127.0.0.1:5000

Команды запросов к API не зависят от способа его запуска и работают идентично.

# Если что то не работает

- Проект настроен на работу с GPU, поэтому его корректная работа на CPU может быть достигнута только при условии внесения изменений в requirements.txt 
Данный файл используется конфигуратором контейнеров при их сборке, поэтому библиотеки необходимые для работы на CPU в него просто не попадут, если не указаны в файле.

- Работа всего проекта тестировалась как в среде win так и ubuntu linux, но использование в продуктовой среде НАСТОЯТЕЛЬНО рекомендуется в Linux Ubuntu (desctop 25 или же live server 22.4 - не принципиально).

## Python и библиотеки
Минимально для запуска проекта необходимо установить:
python3.10
pithon3-pip
docker-compose
В зависимости от операционной системы могут потребоваться иные пакеты, но в большинстве они нужны только при работе и тестировании без использования docker.

## Проверьте, что Docker видит GPU
docker run --rm --gpus all nvidia/cuda:12.4.1-base-ubuntu22.04 nvidia-smi
Для корректной работы может понадобиться установка пакета nvidia-container-toolkit:
sudo apt install nvidia-container-toolkit
