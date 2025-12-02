import argparse, cv2 as cv, numpy as np, faiss, torch, torchvision.transforms as T
import matplotlib.pyplot as plt
from pathlib import Path

from models.encoder import ResNet50Encoder
from utils.config import (IDX_DIR, EMB_DIR, DEVICE, TARGET_SIZE, PAD_RATIO, MIN_OBJ_AREA, TOPK_DEFAULT)
from utils.utils_cv import find_largest_foreground_bbox, pad_bbox, center_square_crop, resize_high_quality

# --- загрузка индексов и id ---
CENTROIDS = np.load(EMB_DIR/"centroids.npy")
CENTROID_IDS = np.load(EMB_DIR/"centroid_ids.npy", allow_pickle=True)
IDX_C = faiss.read_index(str(IDX_DIR/"faiss_centroid.bin"))

# per-image для re-rank
PERIMG = np.load(EMB_DIR/"per_image.npy")
PERIMG_IDS = np.load(EMB_DIR/"part_ids.npy", allow_pickle=True)
IDX_X = faiss.read_index(str(IDX_DIR/"faiss_img.bin"))

transform = T.Compose([
    T.ToTensor(),
    T.Normalize([0.485,0.456,0.406],[0.229,0.224,0.225])
])


@torch.no_grad()
def embed_bgr(img_bgr, model):
    """Детект детали, кроп, resize и эмбеддинг"""
    bbox = find_largest_foreground_bbox(img_bgr, min_area_ratio=MIN_OBJ_AREA)
    if bbox is not None:
        bbox = pad_bbox(bbox, img_bgr.shape, pad_ratio=PAD_RATIO)
        x1, y1, x2, y2 = bbox
        img_bgr = img_bgr[y1:y2, x1:x2]
    else:
        img_bgr = center_square_crop(img_bgr)

    img_bgr = resize_high_quality(img_bgr, TARGET_SIZE)

    import torchvision.transforms.functional as F
    img_rgb = cv.cvtColor(img_bgr, cv.COLOR_BGR2RGB)
    pil = F.to_pil_image(img_rgb)
    x = transform(pil).unsqueeze(0).to(DEVICE)
    emb = model(x).cpu().numpy()
    return emb


def search_and_visualize(query_path: str, topk=TOPK_DEFAULT):
    """Поиск + визуализация результатов"""
    model = ResNet50Encoder().to(DEVICE).eval()
    img = cv.imread(query_path, cv.IMREAD_COLOR)
    if img is None:
        raise FileNotFoundError(f"Не удалось открыть файл {query_path}")

    emb = embed_bgr(img, model)

    # 1) быстрый поиск по центроидам
    D, I = IDX_C.search(emb.astype("float32"), topk)
    cand_part_ids = [CENTROID_IDS[i] for i in I[0]]

    # 2) визуализация
    fig, axes = plt.subplots(1, topk + 1, figsize=(16, 6))

    # показываем query
    img_rgb = cv.cvtColor(img, cv.COLOR_BGR2RGB)
    axes[0].imshow(img_rgb)
    axes[0].set_title("Query")
    axes[0].axis("off")

    # показываем кандидатов
    for j, pid in enumerate(cand_part_ids, start=1):
        cand_img_path = Path("data") / str(pid) / "img1.jpg"  # пример: первая фотка детали
        if cand_img_path.exists():
            cand_img = cv.imread(str(cand_img_path))
            cand_img = cv.cvtColor(cand_img, cv.COLOR_BGR2RGB)
            axes[j].imshow(cand_img)
            axes[j].set_title(f"{pid}")
            axes[j].axis("off")
        else:
            axes[j].axis("off")
            axes[j].set_title(f"{pid}\n(no img)")

    plt.tight_layout()
    plt.show()

    return cand_part_ids


def main_cli():
    """Запуск через CLI"""
    ap = argparse.ArgumentParser()
    ap.add_argument("--query", required=True, type=str, help="путь к изображению для поиска")
    ap.add_argument("--topk", type=int, default=TOPK_DEFAULT)
    args = ap.parse_args()

    results = search_and_visualize(args.query, args.topk)
    print(" Результаты:", results)


def main_test():
    """Тест прямо в VSCode (без аргументов)"""
    query = "data/12345/img1.jpg"  # пример
    results = search_and_visualize(query, topk=5)
    print(" Тестовые результаты:", results)


if __name__ == "__main__":
    import sys
    if len(sys.argv) > 1:
        main_cli()
    else:
        main_test()
