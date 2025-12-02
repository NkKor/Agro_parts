# utils/config.py
"""Конфигурационный файл проекта"""

from pathlib import Path

# --- Пути к данным ---
# Все пути относительно корня проекта
PROJECT_ROOT = Path(__file__).parent.parent  # Поднимаемся из utils/ в корень проекта

RAW_DIR = PROJECT_ROOT / "data" / "raw"
PROC_DIR = PROJECT_ROOT / "data" / "processed"
EMB_DIR = PROJECT_ROOT / "data" / "embeddings"
CENTROIDS_DIR = PROJECT_ROOT / "data" / "centroids"
INDEXES_DIR = PROJECT_ROOT / "data" / "indexes"
MODELS_DIR = PROJECT_ROOT / "src" / "models"
TEMP_DIR = PROJECT_ROOT / "data" / "uploads"

# --- Препроцессинг ---
TARGET_SIZE = 384          # входной размер (ResNet50 обычно 224, но можно 384)
BATCH_SIZE = 32            # Размер батча для обработки
PAD_RATIO = 0.06           # расширение бокса (6%)
MIN_OBJ_AREA = 0.12        # минимальная доля кадра, считаем валидным контуром
BLUR_VAR_THR = 60.0        # порог размытия (Laplacian variance)
OVEREXPO_THR = 0.96        # доля пикселей ~белых -> переэкспонирование
NUM_WORKERS = 4            # Количество worker'ов для DataLoader
DEVICE = "cuda"            # Устройство для обработки
USE_CUDA = True

# --- Эмбеддинги ---
EMB_DIM = 2048             # у ResNet50 после avgpool
EMB_BATCH_SIZE = 128       # Размер батча для извлечения эмбеддингов
EMB_NUM_WORKERS = 4        # Worker'ы для DataLoader эмбеддингов

# --- FAISS ---
FAISS_HNSW_M = 32
TOPK_DEFAULT = 20

# --- API настройки ---
API_HOST = "127.0.0.1"
API_PORT = 3887
DEBUG = False

# --- Веб-приложение настройки ---
WEB_HOST = "127.0.0.1"
WEB_PORT = 5000
WEB_DEBUG = False

# --- Логирование ---
LOG_LEVEL = "INFO"
LOG_DIR = PROJECT_ROOT / "logs"

# Создаем директории если не существуют
LOG_DIR.mkdir(parents=True, exist_ok=True)
TEMP_DIR.mkdir(parents=True, exist_ok=True)

print(f" PROJECT_ROOT: {PROJECT_ROOT}")
print(f" RAW_DIR: {RAW_DIR}")
print(f" PROC_DIR: {PROC_DIR}")
print(f" EMB_DIR: {EMB_DIR}")
print(f" CENTROIDS_DIR: {CENTROIDS_DIR}")
print(f" INDEXES_DIR: {INDEXES_DIR}")
print(f" TEMP_DIR: {TEMP_DIR}")