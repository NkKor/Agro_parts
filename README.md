
### 1 - Препроцессинг датасета, детекция деталей на фото, центрирование кадра, приведение к единому размеру, при необходимости повышение контрастности

1. Обычная обработка (пропуск существующих):
python src/preprocess.py --src data/raw --dst data/processed --size 384

2. Принудительная обработка всех файлов:
python src/preprocess.py --src data/raw --dst data/processed --size 384 --force

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

python src/extract_embeddings.py --src processed --out embeddings
python src/build_index.py --src embeddings --out index.faiss

при добавлении новых фото:
python src/extract_embeddings.py --update (добавятся только новые эмбеддинги).
python src/build_centroids.py --update.
python src/build_index.py --update.



### 3 - 


python src/search_prod.py --query user_input.jpg --topk 5
