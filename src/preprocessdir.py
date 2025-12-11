# src/preprocessdir.py
"""
Инструмент анализа директорий и выделения проблемных каталогов.

Запуск:
  python src/preprocessdir.py --src data/raw --out data/problematic --score-threshold 50 --sample-n 20 --detect-sample 8 --dry-run

Описание:
 - Для каждой поддиректории (класса) берём sample (sample-n) изображений
 - Считаем метрики: laplacian (размытость), contrast (std of gray), exposure (mean luminance), saturation
 - Берём долю точных дубликатов (md5) в sample
 - Запускаем U2NET сегментацию на detect-sample изображениях (быстрая проверка) и считаем долю успешных сегментаций (mask area > min_area_ratio)
 - Если одна из метрик попадает в зону риска (score < score_threshold OR duplicates_ratio > dup_threshold OR detect_success < detect_success_threshold)
   — каталог помечается как проблемный и копируется/перемещается в out (в dry-run только логируем)
 - Логи в JSON (utf-8), поддержка кириллицы, устойчивое копирование с retry.

Примечание: требует src/u2net.py и модель u2net.pth в src/models (скрипт акуратно попытается скачать).
"""

import argparse
import json
import time
import random
from pathlib import Path
import shutil
import hashlib
import sys
import math

import cv2
import numpy as np
import torch
from PIL import Image
from typing import List

# --- U2Net загрузка (локально в src/models/u2net.pth) ---
U2NET_URL = "https://github.com/xuebinqin/U-2-Net/releases/download/v1.0/u2net.pth"
MODEL_DIR = Path("src/models")
MODEL_PATH = MODEL_DIR / "u2net.pth"

def download_u2net_if_missing():
    if MODEL_PATH.exists():
        return
    try:
        import requests, shutil
        MODEL_DIR.mkdir(parents=True, exist_ok=True)
        print("Downloading U2Net model (this may take a while)...")
        r = requests.get(U2NET_URL, stream=True, timeout=30)
        r.raise_for_status()
        with open(MODEL_PATH, "wb") as f:
            shutil.copyfileobj(r.raw, f)
        print("U2Net model downloaded to", MODEL_PATH)
    except Exception as e:
        print("Warning: could not download U2Net model automatically:", e)
        print("Make sure to place the model at:", MODEL_PATH)

# import model class
from src.u2net import U2NET  # requires src/u2net.py present

# ----------------------
# Utilities
# ----------------------
def safe_read_image(path: Path):
    """Try multiple ways to read an image; returns BGR numpy or None"""
    try:
        img = cv2.imread(str(path))
        if img is not None:
            return img
    except Exception:
        pass
    try:
        # fallback via PIL
        with Image.open(path) as pil:
            pil = pil.convert("RGB")
            arr = np.asarray(pil)[:, :, ::-1].copy()  # RGB->BGR
            return arr
    except Exception:
        return None

def md5_of_file(path: Path):
    h = hashlib.md5()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(8192), b""):
            h.update(chunk)
    return h.hexdigest()

def compute_image_metrics(img: np.ndarray):
    """
    Returns dict with laplacian (var), contrast (std gray), exposure (mean luminance), saturation (std S)
    Input BGR numpy
    """
    metrics = {"lap_var": 0.0, "contrast": 0.0, "exposure": 0.0, "saturation": 0.0}
    if img is None:
        return metrics
    try:
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        metrics["lap_var"] = float(cv2.Laplacian(gray, cv2.CV_64F).var())
        metrics["contrast"] = float(gray.std())
        # exposure: mean luminance (0-255)
        metrics["exposure"] = float(gray.mean())
        hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
        s = hsv[:, :, 1]
        metrics["saturation"] = float(s.std())
    except Exception:
        pass
    return metrics

def safe_copy(src: Path, dst: Path, retries: int = 3, overwrite: bool = False):
    """Copy file robustly, handle PermissionError and unicode filenames."""
    dst.parent.mkdir(parents=True, exist_ok=True)
    for i in range(retries):
        try:
            if dst.exists() and not overwrite:
                return True
            # stream copy
            with open(src, "rb") as fr, open(dst, "wb") as fw:
                shutil.copyfileobj(fr, fw)
            shutil.copystat(src, dst, follow_symlinks=True)
            return True
        except PermissionError as e:
            time.sleep(0.2 + 0.2 * i)
            continue
        except Exception as e:
            # fallback to shutil.copy2 which sometimes handles metadata better
            try:
                shutil.copy2(src, dst)
                return True
            except Exception:
                time.sleep(0.1)
                continue
    return False

# ----------------------
# U2Net quick detect helper
# ----------------------
def load_u2net_for_detection(device):
    download_u2net_if_missing()
    model = U2NET(3,1)
    map_loc = torch.device(device if (device.startswith("cuda") and torch.cuda.is_available()) else "cpu")
    try:
        state = torch.load(MODEL_PATH, map_location=map_loc)
        model.load_state_dict(state)
    except Exception as e:
        # try flexible load
        model.load_state_dict({k.replace("module.", ""): v for k,v in state.items()})
    model.to(map_loc)
    model.eval()
    return model

def u2net_quick_mask_area(img: np.ndarray, model, device, resize_to=320):
    """
    Run U2Net on a single image and return mask area ratio (0..1) and boolean success.
    For speed: resize to resize_to for inference.
    """
    if img is None:
        return 0.0, False
    try:
        h, w = img.shape[:2]
        import torch.nn.functional as F
        from torchvision import transforms
        pil = Image.fromarray(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
        transform = transforms.Compose([
            transforms.Resize((resize_to, int(resize_to * w / h) if h !=0 else resize_to)),
            transforms.Resize((resize_to, resize_to)),
            transforms.ToTensor(),
        ])
        tensor = transform(pil).unsqueeze(0).to(device)
        with torch.no_grad():
            d1, *_ = model(tensor)
            pred = d1[:,0,:,:]
            pred = (pred - pred.min()) / (pred.max() - pred.min() + 1e-9)
            pred = F.interpolate(pred.unsqueeze(0), size=(h,w), mode='bilinear', align_corners=False)
            mask = (pred.squeeze().cpu().numpy() * 255).astype(np.uint8)
            area_ratio = (mask>0).sum() / float(h*w)
            success = area_ratio > 0.001  # very small object threshold
            return float(area_ratio), bool(success)
    except Exception:
        return 0.0, False

# ----------------------
# Directory scoring
# ----------------------
def dir_quality_score(dirpath: Path, sample_n: int = 20, detect_sample: int = 6, device: str = "cpu",
                      min_mask_area_ratio=0.005, verbose=False) -> dict:
    """
    Compute a collection of metrics for a directory.
    Returns a dict:
      {
         'n_images': int,
         'sampled': int,
         'avg_lap': float,
         'avg_contrast': float,
         'avg_exposure': float,
         'avg_saturation': float,
         'duplicates_ratio': float,
         'detect_success_ratio': float,
         'mean_mask_area': float
      }
    """
    files = sorted([p for p in dirpath.iterdir() if p.is_file() and p.suffix.lower() in (".jpg",".jpeg",".png")])
    n = len(files)
    if n == 0:
        return {
            "n_images": 0,
            "sampled": 0,
            "avg_lap": 0.0,
            "avg_contrast": 0.0,
            "avg_exposure": 0.0,
            "avg_saturation": 0.0,
            "duplicates_ratio": 0.0,
            "detect_success_ratio": 0.0,
            "mean_mask_area": 0.0
        }

    # sampling
    rng = random.Random(0)  # deterministic
    sample = files if n <= sample_n else rng.sample(files, sample_n)

    laps, contrasts, exposures, sats = [], [], [], []
    md5s = {}
    for p in sample:
        img = safe_read_image(p)
        if img is None:
            continue
        m = compute_image_metrics(img)
        laps.append(m["lap_var"])
        contrasts.append(m["contrast"])
        exposures.append(m["exposure"])
        sats.append(m["saturation"])
        md5s.setdefault(md5_of_file(p), 0)
        md5s[md5_of_file(p)] += 1

    avg_lap = float(np.mean(laps)) if laps else 0.0
    avg_contrast = float(np.mean(contrasts)) if contrasts else 0.0
    avg_exposure = float(np.mean(exposures)) if exposures else 0.0
    avg_saturation = float(np.mean(sats)) if sats else 0.0
    duplicates_ratio = float(1.0 - len(md5s)/float(len(sample))) if sample else 0.0

    # detection sample via U2Net
    # Load U2Net lazily if needed
    detect_success = 0
    mean_mask_area = 0.0
    det_count = 0
    try:
        u2 = load_u2net_for_detection(device)
        det_sample = sample if len(sample)<=detect_sample else sample[:detect_sample]
        areas = []
        successes = 0
        for p in det_sample:
            img = safe_read_image(p)
            area, ok = u2net_quick_mask_area(img, u2, device)
            areas.append(area)
            if ok and area >= min_mask_area_ratio:
                successes += 1
            det_count += 1
        detect_success = float(successes / det_count) if det_count>0 else 0.0
        mean_mask_area = float(np.mean(areas)) if areas else 0.0
    except Exception:
        detect_success = 0.0
        mean_mask_area = 0.0

    return {
        "n_images": n,
        "sampled": len(sample),
        "avg_lap": avg_lap,
        "avg_contrast": avg_contrast,
        "avg_exposure": avg_exposure,
        "avg_saturation": avg_saturation,
        "duplicates_ratio": duplicates_ratio,
        "detect_success_ratio": detect_success,
        "mean_mask_area": mean_mask_area
    }

# ----------------------
# Main
# ----------------------
def main():
    parser = argparse.ArgumentParser(description="Analyze directories and move problematic ones.")
    parser.add_argument("--src", required=True, help="Root source directory with class subdirs")
    parser.add_argument("--out", required=True, help="Output directory for problematic classes")
    parser.add_argument("--score-threshold", type=float, default=50.0,
                        help="Minimal aggregated score (higher is better). See scoring in code.")
    parser.add_argument("--dup-threshold", type=float, default=0.6,
                        help="If duplicates_ratio > dup-threshold => mark problematic")
    parser.add_argument("--detect-success-threshold", type=float, default=0.25,
                        help="If detect_success_ratio < threshold => problematic")
    parser.add_argument("--sample-n", type=int, default=20, help="Number of images to sample per dir")
    parser.add_argument("--detect-sample", type=int, default=6, help="Number images for quick detection by U2Net")
    parser.add_argument("--min-mask-area", type=float, default=0.005, help="Min mask area ratio to consider detection ok")
    parser.add_argument("--device", type=str, default="cuda", help="device for U2Net: 'cuda' or 'cpu'")
    parser.add_argument("--dry-run", action="store_true", help="Do not move/copy files, only log")
    parser.add_argument("--move", action="store_true", help="Move problematic dirs instead of copying")
    parser.add_argument("--log", default="preprocessdir_log.json", help="Output JSON log file")
    args = parser.parse_args()

    src = Path(args.src)
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    # load U2Net if we will run detection (lazy inside function)
    device = args.device if args.device else "cpu"
    # deterministic
    random.seed(0)

    overall_log = {}
    class_dirs = [p for p in sorted(src.iterdir()) if p.is_dir()]

    total = len(class_dirs)
    print(f"Found {total} class dirs to analyze.")

    for idx, cls in enumerate(class_dirs, 1):
        try:
            print(f"[{idx}/{total}] Analyzing: {cls.name}")
            metrics = dir_quality_score(cls, sample_n=args.sample_n, detect_sample=args.detect_sample,
                                        device=device, min_mask_area_ratio=args.min_mask_area)
            # compute a combined score: normalize metrics heuristically
            # lap_var, contrast, exposure, saturation — combine with weights
            # We normalize by simple divisors to keep score in meaningful range
            lap = metrics["avg_lap"] / (200.0 + 1e-9)  # typical laplacian variance scaling
            cont = metrics["avg_contrast"] / (50.0 + 1e-9)
            exp = (metrics["avg_exposure"] / 255.0)
            sat = metrics["avg_saturation"] / (60.0 + 1e-9)
            detect = metrics["detect_success_ratio"]
            dup = metrics["duplicates_ratio"]

            combined = (0.3*lap + 0.3*cont + 0.15*exp + 0.1*sat + 0.15*detect) * 100.0

            is_problem = (combined < args.score_threshold) or (dup > args.dup_threshold) or (detect < args.detect_success_threshold)

            overall_log[cls.name] = {
                "metrics": metrics,
                "combined_score": combined,
                "is_problem": bool(is_problem)
            }

            print(f"  combined_score={combined:.1f}, duplicates={dup:.2f}, detect_success={detect:.2f} => problem={is_problem}")

            if is_problem and not args.dry_run:
                target = out / cls.name
                if args.move:
                    # try to move dir
                    try:
                        shutil.move(str(cls), str(target))
                        print(f"  moved {cls} -> {target}")
                    except Exception as e:
                        print(f"  move failed: {e}. Attempting copy instead.")
                        target.mkdir(parents=True, exist_ok=True)
                        for f in cls.iterdir():
                            if f.is_file():
                                safe_copy(f, target / f.name)
                else:
                    # copy files
                    target.mkdir(parents=True, exist_ok=True)
                    for f in cls.iterdir():
                        if f.is_file():
                            ok = safe_copy(f, target / f.name)
                            if not ok:
                                print(f"  WARNING: failed to copy {f}")
            elif is_problem and args.dry_run:
                print(f"  dry-run: would mark {cls.name} problematic (no files moved/copied).")
        except Exception as e:
            print(f"Error analyzing {cls}: {e}", file=sys.stderr)
            overall_log[cls.name] = {"error": str(e)}

    # write log
    with open(args.log, "w", encoding="utf-8") as fo:
        json.dump(overall_log, fo, ensure_ascii=False, indent=2)

    print("Done. Log saved to", args.log)

if __name__ == "__main__":
    main()
