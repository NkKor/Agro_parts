import numpy as np, faiss
from pathlib import Path
from config import EMB_DIR, IDX_DIR, FAISS_HNSW_M

def build_hnsw(vectors: np.ndarray, m=32):
    dim = vectors.shape[1]
    index = faiss.IndexHNSWFlat(dim, m)
    index.hnsw.efConstruction = 200
    index.add(vectors.astype("float32"))
    return index

def main():
    IDX_DIR.mkdir(parents=True, exist_ok=True)

    # центроидный индекс (основной)
    C = np.load(EMB_DIR/"centroids.npy")
    idx_c = build_hnsw(C, m=FAISS_HNSW_M)
    faiss.write_index(idx_c, str(IDX_DIR/"faiss_centroid.bin"))

    # per-image индекс (для вторичного rerank; опционально)
    X = np.load(EMB_DIR/"per_image.npy")
    idx_x = build_hnsw(X, m=FAISS_HNSW_M)
    faiss.write_index(idx_x, str(IDX_DIR/"faiss_img.bin"))

    print("✅ FAISS indices saved.")

if __name__ == "__main__":
    main()
