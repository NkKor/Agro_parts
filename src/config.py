from pathlib import Path

# Пути
RAW_DIR = Path("data/raw")
PROC_DIR = Path("data/processed")
EMB_DIR = Path("data/embeddings")
IDX_DIR = Path("data/index")

# Препроцессинг
TARGET_SIZE = 336          # входной размер (ResNet50 обычно 224, но можно 256/336/512)
PAD_RATIO = 0.06           # расширение бокса (6%)
MIN_OBJ_AREA = 0.12        # минимальная доля кадра, считаем валидным контуром
BLUR_VAR_THR = 60.0        # порог размытия (Laplacian variance)
OVEREXPO_THR = 0.96        # доля пикселей ~белых -> переэкспонирование

# Эмбеддинги
EMB_DIM = 2048             # у ResNet50 после avgpool
BATCH_SIZE = 128
NUM_WORKERS = 6
DEVICE = "cuda"

# FAISS
FAISS_HNSW_M = 32
TOPK_DEFAULT = 20
