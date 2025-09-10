import argparse
import numpy as np
import faiss
from tqdm import tqdm
from config import EMB_DIR, IDX_DIR

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--update", action="store_true", help="пересоздать индексы")
    args = ap.parse_args()

    perimg = np.load(EMB_DIR / "per_image.npy")
    centroids = np.load(EMB_DIR / "centroids.npy")

    IDX_DIR.mkdir(parents=True, exist_ok=True)

    # FAISS index по per-image
    dim = perimg.shape[1]
    print("🔄 Строим индекс для per-image эмбеддингов...")
    idx_img = faiss.IndexFlatIP(dim)
    idx_img.add(perimg.astype("float32"))
    faiss.write_index(idx_img, str(IDX_DIR / "faiss_img.bin"))

    # FAISS index по центроидам
    dim_c = centroids.shape[1]
    print("🔄 Строим индекс для центроидов...")
    idx_c = faiss.IndexFlatIP(dim_c)
    idx_c.add(centroids.astype("float32"))
    faiss.write_index(idx_c, str(IDX_DIR / "faiss_centroid.bin"))

    print("✅ Индексы пересозданы.")

if __name__ == "__main__":
    main()
