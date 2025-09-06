import numpy as np
from collections import defaultdict
from pathlib import Path
from config import EMB_DIR

def main():
    feats = np.load(EMB_DIR/"per_image.npy")          # [N, D] (уже L2)
    ids   = np.load(EMB_DIR/"part_ids.npy", allow_pickle=True)

    buckets = defaultdict(list)
    for f, pid in zip(feats, ids):
        buckets[pid].append(f)

    centroids, centroid_ids = [], []
    for pid, arr in buckets.items():
        c = np.mean(np.vstack(arr), axis=0)
        # важно: нормализуем ещё раз после усреднения
        c = c / (np.linalg.norm(c) + 1e-12)
        centroids.append(c)
        centroid_ids.append(pid)

    centroids = np.vstack(centroids)
    centroid_ids = np.array(centroid_ids, dtype=object)
    np.save(EMB_DIR/"centroids.npy", centroids)
    np.save(EMB_DIR/"centroid_ids.npy", centroid_ids)
    print("✅ saved centroids:", centroids.shape)

if __name__ == "__main__":
    main()
