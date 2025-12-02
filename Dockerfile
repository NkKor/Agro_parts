# Dockerfile
# Используем официальный образ Python slim
FROM python:3.9-slim

# Установка системных зависимостей, необходимых для работы OpenCV и других библиотек
# libgl1-mesa-glx для GUI функций OpenCV (иногда требуется)
# libglib2.0-0 для glib
RUN apt-get update && apt-get install -y \
    build-essential \
    libgl1-mesa-glx \
    libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

# Установка рабочей директории внутри контейнера
WORKDIR /app

# Копирование файла зависимостей
COPY requirements.txt .

# Установка Python зависимостей
# --no-cache-dir экономит место, не сохраняя кэш пакетов
RUN pip install --no-cache-dir -r requirements.txt

# Копирование всего исходного кода проекта в контейнер
# .dockerignore должен исключать ненужные файлы (например, .git, __pycache__)
COPY . .

# Создание директорий для данных и логов (на случай, если они будут примонтированы)
RUN mkdir -p data/embeddings data/processed logs tmp_uploads

# Открытие порта, на котором будет работать API
EXPOSE 3887

# Команда по умолчанию для запуска контейнера
# Используем Gunicorn как WSGI-сервер для продакшена

# СТАРАЯ КОМАНДА (с app.py):
# CMD ["gunicorn", "--bind", "0.0.0.0:3887", "--workers", "4", "--timeout", "120", "--keep-alive", "5", "src.api.app:create_app()"]

# НОВАЯ КОМАНДА (с api_app.py):
CMD ["gunicorn", "--bind", "0.0.0.0:3887", "--workers", "4", "--timeout", "120", "--keep-alive", "5", "src.api.api_app:create_app()"]
