# src/web_app/Dockerfile
FROM nvidia/cuda:12.4.1-runtime-ubuntu22.04

# переменные окружения
ENV DEBIAN_FRONTEND=noninteractive
ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1
ENV PIP_NO_CACHE_DIR=1
ENV HOME=/app
ENV TORCH_HOME=/app/data/models
ENV PYTHONPATH=/app

# системные зависимости (нужны для OpenCV, FAISS, сборки некоторых wheel'ов)
RUN apt-get update && \
    apt-get install -y --no-install-recommends \
    python3.10 \
    python3.10-venv \
    python3-pip \
    build-essential \
    libgomp1 \
    libglib2.0-0 \
    libsm6 \
    libxext6 \
    libgl1 \
    curl \
    ca-certificates && \
    rm -rf /var/lib/apt/lists/*

WORKDIR /app
COPY whl_cache /tmp/whl_cache/

# Устанавливаем зависимости
COPY requirements.txt /app/requirements.txt
RUN python3.10 -m pip install --no-cache-dir -r requirements.txt \
    --find-links /tmp/whl_cache/ \
    --timeout 60

# Копируем код веб-приложения и общие модули
COPY src/web_app/ /app/src/web_app/
COPY src/ /app/src/
COPY utils/ /app/utils/

# Создаём пользователя и даём права
RUN groupadd -r app && useradd -m -r -g app -d /app appuser && chown -R appuser:app /app

# Устанавливаем правильные права на директории данных
RUN mkdir -p /app/data/processed /app/data/embeddings /app/data/centroids /app/data/indexes /app/data/uploads /app/logs /app/src/models && \
    chown -R appuser:app /app/data /app/logs /app/src/models


# Создаём файл лога заранее и устанавливаем права
RUN touch /app/logs/web_app.log && chown appuser:app /app/logs/web_app.log
RUN touch /app/logs/searcher.log && chown appuser:app /app/logs/searcher.log

# Устанавливаем права на домашнюю директорию моделей и загрузки  для пользователя appuser
RUN mkdir -p /app/data/models && chown -R appuser:app /app/data/models
RUN mkdir -p /app/data/uploads && chown -R appuser:app /app/data/uploads

USER appuser

EXPOSE 5000

# Запуск Gunicorn с --preload для устранения проблем с памятью
# --workers 1-2 должно быть достаточно для веб-интерфейса.
CMD gunicorn --bind 0.0.0.0:5000 --workers 1 --timeout ${GUNICORN_TIMEOUT} --log-level info src.web_app.app:app
