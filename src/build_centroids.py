import argparse
import numpy as np
from pathlib import Path
from collections import defaultdict
from tqdm import tqdm
from config import EMB_DIR

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--update", action="store_true", help="пересоздать центроиды")
    args = ap.parse_args()

    emb_file = EMB_DIR / "per_image.npy"
    ids_file = EMB_DIR / "part_ids.npy"
    if not emb_file.exists() or not ids_file.exists():
        raise FileNotFoundError("Сначала запусти extract_embeddings.py")

    embs = np.load(emb_file)
    ids = np.load(ids_file, allow_pickle=True)

    # сгруппируем по part_id
    part_to_embs = defaultdict(list)
    print("🔄 Группировка эмбеддингов по деталям...")
    for emb, rel_id in zip(tqdm(embs), ids):
        part_id = rel_id.split("/")[0]
        part_to_embs[part_id].append(emb)

    centroids, centroid_ids = [], []
    for pid, vecs in tqdm(part_to_embs.items(), desc="Счёт центроидов"):
        arr = np.stack(vecs, axis=0)
        centroids.append(arr.mean(axis=0))
        centroid_ids.append(pid)

    centroids = np.stack(centroids, axis=0)
    centroid_ids = np.array(centroid_ids, dtype=object)

    np.save(EMB_DIR / "centroids.npy", centroids)
    np.save(EMB_DIR / "centroid_ids.npy", centroid_ids)

    print(f"✅ Пересчитано {len(centroids)} центроидов.")

if __name__ == "__main__":
    main()
