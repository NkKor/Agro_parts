import argparse, cv2 as cv, numpy as np, faiss, torch, torchvision.transforms as T
from pathlib import Path
from models.encoder import ResNet50Encoder
from utils.config import (IDX_DIR, EMB_DIR, DEVICE, TARGET_SIZE, PAD_RATIO, MIN_OBJ_AREA, TOPK_DEFAULT)
from utils.utils_cv import find_largest_foreground_bbox, pad_bbox, center_square_crop, resize_high_quality

# --- загрузка индексов и id ---
CENTROIDS = np.load(EMB_DIR/"centroids.npy")
CENTROID_IDS = np.load(EMB_DIR/"centroid_ids.npy", allow_pickle=True)
IDX_C = faiss.read_index(str(IDX_DIR/"faiss_centroid.bin"))

# per-image для re-rank (необязательно)
PERIMG = np.load(EMB_DIR/"per_image.npy")
PERIMG_IDS = np.load(EMB_DIR/"part_ids.npy", allow_pickle=True)
IDX_X = faiss.read_index(str(IDX_DIR/"faiss_img.bin"))

transform = T.Compose([
    T.ToTensor(),
    T.Normalize([0.485,0.456,0.406],[0.229,0.224,0.225])
])

@torch.no_grad()
def embed_bgr(img_bgr, model):
    """Детект детали, кроп, resize и извлечение эмбеддинга"""
    bbox = find_largest_foreground_bbox(img_bgr, min_area_ratio=MIN_OBJ_AREA)
    if bbox is not None:
        bbox = pad_bbox(bbox, img_bgr.shape, pad_ratio=PAD_RATIO)
        x1,y1,x2,y2 = bbox
        img_bgr = img_bgr[y1:y2, x1:x2]
    else:
        img_bgr = center_square_crop(img_bgr)

    img_bgr = resize_high_quality(img_bgr, TARGET_SIZE)

    import torchvision.transforms.functional as F
    img_rgb = cv.cvtColor(img_bgr, cv.COLOR_BGR2RGB)
    pil = F.to_pil_image(img_rgb)
    x = transform(pil).unsqueeze(0).to(DEVICE)
    emb = model(x).cpu().numpy()  # L2-норма уже применена в encoder
    return emb


def search(query_path: str, topk=TOPK_DEFAULT, rerank_per_image=True, per_image_k=200):
    """Поиск по базе"""
    model = ResNet50Encoder().to(DEVICE).eval()

    img = cv.imread(query_path, cv.IMREAD_COLOR)
    if img is None:
        raise FileNotFoundError(f"Не удалось открыть файл {query_path}")

    emb = embed_bgr(img, model)  # [1, D]

    # 1) быстрый поиск по центроидам
    D, I = IDX_C.search(emb.astype("float32"), topk if not rerank_per_image else per_image_k)
    cand_part_ids = [CENTROID_IDS[i] for i in I[0]]

    if not rerank_per_image:
        return list(zip(cand_part_ids, D[0].tolist()))

    # 2) переранжирование по per-image эмбеддингам кандидатов
    cand_mask = np.isin(PERIMG_IDS, np.array(cand_part_ids, dtype=object))
    cand_vectors = PERIMG[cand_mask]
    sims = (emb @ cand_vectors.T).ravel()  # косинусная близость
    cand_ids = PERIMG_IDS[cand_mask]

    best = {}
    for sim, pid in zip(sims, cand_ids):
        best[pid] = max(best.get(pid, -1.0), float(sim))

    ranked = sorted(best.items(), key=lambda x: x[1], reverse=True)[:topk]
    return ranked


def main_cli():
    """Запуск через командную строку"""
    ap = argparse.ArgumentParser()
    ap.add_argument("--query", required=True, type=str)
    ap.add_argument("--topk", type=int, default=TOPK_DEFAULT)
    ap.add_argument("--no_rerank", action="store_true")
    args = ap.parse_args()

    results = search(args.query, topk=args.topk, rerank_per_image=not args.no_rerank)
    print("🔎 Результаты:")
    for pid, score in results:
        print(f"{pid}\t{score:.4f}")


def main_test():
    """Тестовый запуск прямо из VSCode (без аргументов)"""
    query = "data/test/1.jpg"  # пример
    results = search(query, topk=5, rerank_per_image=True)
    print("🔎 Тестовые результаты:")
    for pid, score in results:
        print(f"{pid}\t{score:.4f}")


if __name__ == "__main__":
    import sys
    if len(sys.argv) > 1:
        main_cli()
    else:
        main_test()
