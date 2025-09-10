import argparse
import numpy as np
import cv2 as cv
import torch
import torchvision.transforms as T
from tqdm import tqdm
from pathlib import Path

from encoder import ResNet50Encoder
from utils_cv import find_largest_foreground_bbox, pad_bbox, center_square_crop, resize_high_quality
from config import (RAW_DIR, PROC_DIR, EMB_DIR, DEVICE, TARGET_SIZE, PAD_RATIO, MIN_OBJ_AREA)

transform = T.Compose([
    T.ToTensor(),
    T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

@torch.no_grad()
def embed_image(img_bgr, model):
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
    emb = model(x).cpu().numpy()[0]  # (D,)
    return emb

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--update", action="store_true", help="добавить только новые изображения")
    args = ap.parse_args()

    EMB_DIR.mkdir(parents=True, exist_ok=True)
    emb_file = EMB_DIR / "per_image.npy"
    ids_file = EMB_DIR / "part_ids.npy"

    existing_embs, existing_ids, seen_files = None, None, set()
    if args.update and emb_file.exists() and ids_file.exists():
        existing_embs = np.load(emb_file)
        existing_ids = np.load(ids_file, allow_pickle=True)
        seen_files = set(existing_ids.tolist())

    model = ResNet50Encoder().to(DEVICE).eval()

    all_imgs, all_ids = [], []
    print("🔄 Извлечение эмбеддингов...")
    for part_dir in Path(PROC_DIR).iterdir():
        if not part_dir.is_dir():
            continue
        part_id = part_dir.name
        for img_path in part_dir.glob("*.jpg"):
            rel_id = f"{part_id}/{img_path.name}"
            if rel_id in seen_files:
                continue
            img = cv.imread(str(img_path), cv.IMREAD_COLOR)
            if img is None:
                continue
            emb = embed_image(img, model)
            all_imgs.append(emb)
            all_ids.append(rel_id)

    if len(all_imgs) == 0:
        print("✅ Новых изображений нет.")
        return

    all_imgs = np.stack(all_imgs, axis=0)

    if existing_embs is not None:
        final_embs = np.concatenate([existing_embs, all_imgs], axis=0)
        final_ids = np.concatenate([existing_ids, np.array(all_ids, dtype=object)], axis=0)
    else:
        final_embs = all_imgs
        final_ids = np.array(all_ids, dtype=object)

    np.save(emb_file, final_embs)
    np.save(ids_file, final_ids)
    print(f"✅ Сохранено {len(all_ids)} новых эмбеддингов, всего {len(final_ids)}.")

if __name__ == "__main__":
    main()
