
### 1 - Препроцессинг датасета, детекция деталей на фото, центрирование кадра, приведение к единому размеру, при необходимости повышение контрастности

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

### 2 - Извлечение признаков, создание базы индексов для поисковой машины FAISS

-- 1. Извлечение эмбеддингов

python src/extract_embeddings.py --src data/processed --out data/embeddings по умолчанию обрабатывает только новые изображения
python src/extract_embeddings.py --src data/processed --out data/embeddings --force принудительно обновляет все вектора
python src/extract_embeddings.py --src data/processed --out data/embeddings --device cuda:0 принудительный выбор CUDA

-- 2. Построение центроидов

python src/build_centroids.py --embeddings data/embeddings --out data/centroids по умолчанию не пересоздаёт, если файл уже есть
python src/build_centroids.py --embeddings data/embeddings --out data/centroids --force  для принудительного пересоздания

-- 3. Построение индекса

python src/build_index.py --embeddings data/embeddings --centroids data/centroids --out data/indexes по умолчанию не пересоздаёт, если файл уже есть
python src/build_index.py --embeddings data/embeddings --centroids data/centroids --out data/indexes --force  для принудительного пересоздания

Конвертация старого формата
python src/convert_embeddings.py --input data/embeddings.pkl --output data/embeddings
Затем продолжаем с новым extract_embeddings.py
python src/extract_embeddings.py --src data/processed --out data/embeddings

#### При добавлении новых фото выполнить последовательно:

python src/preprocess.py --src data/raw --dst data/processed --size 512
Извлечение эмбеддингов (инкрементально)
python src/extract_embeddings.py --src data/processed --out data/embeddings
Построение центроидов
python src/build_centroids.py --embeddings data/embeddings --out data/centroids
Построение индексов
python src/build_index.py --embeddings data/embeddings --centroids data/centroids --out data/indexes



### 3 - Поиск (тест / прод), веб приложение

python src/web_app/app.py






python src/search_prod.py --query user_input.jpg --topk 5