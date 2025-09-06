import argparse, os
from pathlib import Path
import cv2 as cv
from tqdm import tqdm
from utils_cv import (variance_of_laplacian, overexposed_ratio,
                      find_largest_foreground_bbox, pad_bbox,
                      center_square_crop, resize_high_quality)
from config import (TARGET_SIZE, PAD_RATIO, MIN_OBJ_AREA, BLUR_VAR_THR, OVEREXPO_THR)

def process_image(src_path: Path, dst_path: Path):
    img = cv.imread(str(src_path), cv.IMREAD_COLOR)
    if img is None:
        return False, "read_fail"

    # QC: экспозиция и размытость (логируем, но не отбрасываем жёстко)
    gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
    blur_score = variance_of_laplacian(gray)
    overexp = overexposed_ratio(gray)

    # детект крупного объекта и кроп
    bbox = find_largest_foreground_bbox(img, min_area_ratio=MIN_OBJ_AREA)
    if bbox is not None:
        bbox = pad_bbox(bbox, img.shape, pad_ratio=PAD_RATIO)
        x1,y1,x2,y2 = bbox
        crop = img[y1:y2, x1:x2]
    else:
        # запасной план: центр-квадрат
        crop = center_square_crop(img)

    # приведение к квадрату без «потери» ключевых деталей — просто high-quality resize
    out = resize_high_quality(crop, TARGET_SIZE)

    dst_path.parent.mkdir(parents=True, exist_ok=True)
    cv.imwrite(str(dst_path), out, [cv.IMWRITE_JPEG_QUALITY, 95])

    # Можно писать CSV-лог качества при желании
    return True, {"blur": blur_score, "overexp": float(overexp)}

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", type=str, required=True)
    ap.add_argument("--dst", type=str, required=True)
    args = ap.parse_args()

    src_root = Path(args.src)
    dst_root = Path(args.dst)

    files = list(src_root.rglob("*"))
    images = [p for p in files if p.suffix.lower() in [".jpg",".jpeg",".png"]]

    for p in tqdm(images, desc="preprocess"):
        rel = p.relative_to(src_root)
        dst = dst_root / rel.with_suffix(".jpg")
        ok, _ = process_image(p, dst)

if __name__ == "__main__":
    main()
