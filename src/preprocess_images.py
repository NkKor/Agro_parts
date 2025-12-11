# src/preprocess_images.py
"""
Унифицированный препроцессор изображений.

Запуск примера:
  python src/preprocess_images.py --src data/raw --dst data/raw-clean --max-per-dir 20 --min-lap 50 --min-contrast 10 --phash-threshold 0.10 --device auto --delete-bad --workers 4

Функции:
 - md5-дубликаты удаляются
 - похожие изображения по phash (imagehash.phash) объединяются: оставляем лучший по качеству
 - фильтруем низкое качество по laplacian, contrast, exposure, saturation (адаптивно по каталогу)
 - выбираем до max_per_dir изображений, обеспечивая разнообразие (greedy по HSV-гистограмме)
 - сегментация U2Net (кэш масок), мягкая маска (feather), заливка белым фоном
 - сохраняем метаданные в dst/<class>/_meta.json и маски в dst/_masks/<class>/
 - логируем JSON общий лог
 - поддержка кириллицы и устойчивого копирования
 - кеширование результатов анализа (hashes, metrics) в src/.cache/preprocess_images/
"""

import argparse
import json
import math
import os
import random
import shutil
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import List, Dict, Tuple

import cv2
import imagehash
import numpy as np
import torch
from PIL import Image
from tqdm import tqdm

# U2Net import
from src.u2net import U2NET

# model path
U2NET_URL = "https://github.com/xuebinqin/U-2-Net/releases/download/v1.0/u2net.pth"
MODEL_DIR = Path("src/models")
MODEL_PATH = MODEL_DIR / "u2net.pth"

def download_u2net_if_missing():
    if MODEL_PATH.exists():
        return
    try:
        import requests
        MODEL_DIR.mkdir(parents=True, exist_ok=True)
        print("Downloading U2Net model...")
        r = requests.get(U2NET_URL, stream=True, timeout=30)
        r.raise_for_status()
        with open(MODEL_PATH, "wb") as f:
            shutil.copyfileobj(r.raw, f)
        print("Downloaded U2Net model.")
    except Exception as e:
        print("Warning: could not download U2Net automatically:", e)

# -----------------------
# Utilities
# -----------------------
def safe_read_image(path: Path):
    """Robust image read returning BGR numpy or None"""
    try:
        img = cv2.imdecode(np.fromfile(str(path), dtype=np.uint8), cv2.IMREAD_COLOR)
        if img is not None:
            return img
    except Exception:
        pass
    try:
        img = cv2.imread(str(path))
        if img is not None:
            return img
    except Exception:
        pass
    try:
        with Image.open(path) as pil:
            pil = pil.convert("RGB")
            arr = np.asarray(pil)[:, :, ::-1].copy()
            return arr
    except Exception:
        return None

def md5_of_file(path: Path) -> str:
    import hashlib
    h = hashlib.md5()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(8192), b""):
            h.update(chunk)
    return h.hexdigest()

def phash_of_image(path: Path):
    try:
        with Image.open(path) as im:
            return imagehash.phash(im)
    except Exception:
        return None

def compute_image_metrics(img: np.ndarray) -> Dict[str, float]:
    """Compute laplacian, contrast, exposure, saturation"""
    out = {"lap_var": 0.0, "contrast": 0.0, "exposure": 0.0, "saturation": 0.0}
    if img is None:
        return out
    try:
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        out["lap_var"] = float(cv2.Laplacian(gray, cv2.CV_64F).var())
        out["contrast"] = float(gray.std())
        out["exposure"] = float(gray.mean())
        hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
        out["saturation"] = float(hsv[:, :, 1].std())
    except Exception:
        pass
    return out

def safe_copy(src: Path, dst: Path, retries: int = 3, overwrite: bool = False) -> bool:
    dst.parent.mkdir(parents=True, exist_ok=True)
    for i in range(retries):
        try:
            if dst.exists() and not overwrite:
                return True
            # use numpy-based copy for unicode-safe path on windows
            with open(src, "rb") as fr, open(dst, "wb") as fw:
                shutil.copyfileobj(fr, fw)
            try:
                shutil.copystat(src, dst)
            except Exception:
                pass
            return True
        except PermissionError:
            time.sleep(0.2 + 0.1 * i)
            continue
        except Exception:
            try:
                shutil.copy2(src, dst)
                return True
            except Exception:
                time.sleep(0.1)
                continue
    return False

# -----------------------
# U2Net segmentation helpers
# -----------------------
def load_u2net(device: str):
    download_u2net_if_missing()
    model = U2NET(3,1)
    map_loc = torch.device(device if (device.startswith("cuda") and torch.cuda.is_available()) else "cpu")
    state = torch.load(MODEL_PATH, map_location=map_loc)
    # flexible state dict
    try:
        model.load_state_dict(state)
    except Exception:
        model.load_state_dict({k.replace("module.", ""): v for k,v in state.items()})
    model.to(map_loc)
    model.eval()
    return model, map_loc

import torch.nn.functional as F
from torchvision import transforms

def u2net_segment_mask(img_bgr: np.ndarray, model, device, resize_short=320) -> np.ndarray:
    """
    Returns mask uint8 (0/255) same HxW as input.
    Uses model on device, does smoothing and returns binary mask.
    """
    if img_bgr is None:
        return None
    h, w = img_bgr.shape[:2]
    pil = Image.fromarray(cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB))
    # maintain aspect ratio: resize shorter side to resize_short, then to square resize_short x resize_short
    transform = transforms.Compose([
        transforms.Resize((resize_short, resize_short)),
        transforms.ToTensor()
    ])
    tensor = transform(pil).unsqueeze(0).to(device)
    with torch.no_grad():
        d1, *_ = model(tensor)
        pred = d1[:,0,:,:]
        pred = (pred - pred.min()) / (pred.max() - pred.min() + 1e-9)
        pred_up = F.interpolate(pred.unsqueeze(0), size=(h,w), mode='bilinear', align_corners=False)
        mask = (pred_up.squeeze().cpu().numpy() * 255.0).astype(np.uint8)
        # smooth + threshold with soft edge
        mask = cv2.GaussianBlur(mask, (7,7), 0)
        _, thr = cv2.threshold(mask, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        # create soft alpha using normalized mask to blend
        return (mask > (thr//2)).astype(np.uint8) * 255

def apply_mask_white_bg(img_bgr: np.ndarray, mask_255: np.ndarray) -> np.ndarray:
    """Apply mask but with soft edges: create alpha from mask and composite on white."""
    if img_bgr is None or mask_255 is None:
        return None
    mask_f = (mask_255.astype(np.float32) / 255.0)[:,:,None]
    white = np.ones_like(img_bgr, dtype=np.uint8) * 255
    out = (img_bgr.astype(np.float32) * mask_f + white.astype(np.float32) * (1.0 - mask_f)).astype(np.uint8)
    return out

# -----------------------
# Dedup with phash
# -----------------------
def find_phash_groups(paths: List[Path], threshold: float = 0.10) -> List[List[Path]]:
    """
    Group images by phash similarity.
    threshold: normalized hamming distance (0..1) e.g. 0.10 means <= 10% bits different.
    Returns list of groups where first element is canonical.
    """
    hashes = {}
    for p in paths:
        ph = phash_of_image(p)
        if ph is None:
            continue
        hashes[p] = ph

    used = set()
    groups = []
    items = list(hashes.items())
    for i, (p, h) in enumerate(items):
        if p in used:
            continue
        group = [p]
        used.add(p)
        for q, hq in items[i+1:]:
            if q in used:
                continue
            # normalized hamming distance
            dist = (h - hq)  # integer
            norm = dist / (h.hash.size)  # h.hash is numpy array; size = bits
            if norm <= threshold:
                group.append(q)
                used.add(q)
        groups.append(group)
    # include any paths without hash as singleton groups
    for p in paths:
        if p not in hashes:
            groups.append([p])
    return groups

# -----------------------
# Diversity selection (greedy on HSV hist distance)
# -----------------------
def hsv_hist_descriptor(img_bgr: np.ndarray, bins=(16,8)):
    """Return normalized histogram of H and S channels (flattened)."""
    if img_bgr is None:
        return np.zeros(bins[0]*bins[1], dtype=np.float32)
    hsv = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2HSV)
    h = hsv[:,:,0]; s = hsv[:,:,1]
    hist = cv2.calcHist([h,s], [0,1], None, [bins[0], bins[1]], [0,180, 0,256])
    hist = hist.flatten()
    if hist.sum() > 0:
        hist = hist / (hist.sum()+1e-9)
    return hist

def select_diverse_by_hist(path_list: List[Path], k: int) -> List[Path]:
    """Greedy selection: pick highest-quality first externally, then iteratively pick item with max distance."""
    if len(path_list) <= k:
        return path_list.copy()
    # compute hist descriptors
    descs = {}
    for p in path_list:
        img = safe_read_image(p)
        descs[p] = hsv_hist_descriptor(img)
    # start with first
    selected = []
    remaining = set(path_list)
    # pick image with largest sum of metrics (we don't have metrics here), so pick arbitrarily first: use mean lap as proxy by loading metrics externally.
    # For now choose first by filename sorted (deterministic)
    first = sorted(path_list)[0]
    selected.append(first)
    remaining.remove(first)
    while len(selected) < k and remaining:
        best_p = None
        best_dist = -1.0
        for p in list(remaining):
            dists = [np.linalg.norm(descs[p] - descs[s]) for s in selected]
            mind = min(dists) if dists else 1.0
            if mind > best_dist:
                best_dist = mind
                best_p = p
        if best_p is None:
            break
        selected.append(best_p)
        remaining.remove(best_p)
    # if still less than k, fill from remaining sorted
    if len(selected) < k:
        remaining_sorted = sorted(list(remaining))
        for p in remaining_sorted[:(k-len(selected))]:
            selected.append(p)
    return selected

# -----------------------
# Process single class dir
# -----------------------
CACHE_DIR = Path("src/.cache/preprocess_images")
CACHE_DIR.mkdir(parents=True, exist_ok=True)

def analyze_class_dir(class_dir: Path, force_recalc: bool = False) -> Dict:
    """
    Analyze images in a class dir: compute md5, phash, metrics.
    Returns dict with lists and maps.
    Caches result to CACHE_DIR/<class>.json for speed.
    """
    cache_file = CACHE_DIR / f"{class_dir.name}.json"
    if cache_file.exists() and not force_recalc:
        try:
            with open(cache_file, "r", encoding="utf-8") as f:
                return json.load(f)
        except Exception:
            pass

    files = sorted([p for p in class_dir.iterdir() if p.is_file() and p.suffix.lower() in (".jpg",".jpeg",".png")])
    data = {
        "files": [str(p) for p in files],
        "md5": {},
        "phash": {},
        "metrics": {}
    }
    for p in files:
        try:
            md = md5_of_file(p)
            data["md5"][str(p)] = md
        except Exception:
            data["md5"][str(p)] = None
        ph = phash_of_image(p)
        data["phash"][str(p)] = str(ph) if ph is not None else None
        img = safe_read_image(p)
        data["metrics"][str(p)] = compute_image_metrics(img)
    try:
        with open(cache_file, "w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False, indent=2)
    except Exception:
        pass
    return data

def choose_best_in_phash_group(group: List[Path], metrics_map: Dict[str, Dict]) -> Tuple[Path, List[Path]]:
    """
    Given a group of similar images (phash), select best by combined quality metric, return (best, removed_list)
    Combined metric: weighted sum of lap_var, contrast, exposure (centered), saturation
    """
    best = None
    best_score = -1e9
    for p in group:
        m = metrics_map.get(str(p), None)
        if m is None:
            score = 0.0
        else:
            # combine; normalize roughly
            lap = m.get("lap_var",0.0)/200.0
            cont = m.get("contrast",0.0)/50.0
            exp = (m.get("exposure",0.0)/255.0)
            sat = m.get("saturation",0.0)/60.0
            score = 0.45*lap + 0.35*cont + 0.1*exp + 0.1*sat
        if score > best_score:
            best_score = score
            best = p
    removed = [p for p in group if p != best]
    return best, removed

def process_class(class_dir: Path, out_root: Path, u2_model, device, args) -> Dict:
    """
    Full pipeline for one class.
    Steps:
      - analyze (cached)
      - remove exact md5 duplicates (keep first occurrence)
      - group by phash and keep best in group (threshold)
      - compute per-class adaptive threshold: mean_quality * factor (args.adaptive_factor)
      - filter low-quality images
      - if > max_images: select by diversity + quality
      - segment chosen images and save to out
      - save metadata
      - optionally delete bad originals if args.delete_bad
    """
    res = {"class": class_dir.name, "initial": 0, "kept": [], "removed_exact": [], "removed_similar": [], "removed_low_quality": [], "segmented": [], "errors": []}
    try:
        analysis = analyze_class_dir(class_dir, force_recalc=args.force_recalc)
        files = [Path(p) for p in analysis.get("files",[])]
        res["initial"] = len(files)
        if len(files) == 0:
            return res

        # 1) exact md5 dedup
        md5_map = {}
        kept = []
        for p in files:
            md = analysis["md5"].get(str(p))
            if md is None:
                # keep unknown
                kept.append(p)
                continue
            if md in md5_map:
                res["removed_exact"].append(str(p))
            else:
                md5_map[md] = p
                kept.append(p)

        # 2) phash grouping and removal of similar (threshold normalized hamming)
        phash_map = {}
        valid_for_phash = []
        for p in kept:
            phs = analysis["phash"].get(str(p))
            if phs is None:
                continue
            try:
                ph = imagehash.hex_to_hash(phs)
                phash_map[p] = ph
                valid_for_phash.append(p)
            except Exception:
                continue

        # build groups
        groups = []
        used = set()
        item_list = list(phash_map.items())
        for i,(p,h) in enumerate(item_list):
            if p in used:
                continue
            grp = [p]
            used.add(p)
            for q,hq in item_list[i+1:]:
                if q in used:
                    continue
                dist = (h - hq)
                norm = dist / (h.hash.size)
                if norm <= args.phash_threshold:
                    grp.append(q)
                    used.add(q)
            groups.append(grp)
        # include singletons that had no phash or were omitted
        singletons = [p for p in kept if p not in phash_map]
        for s in singletons:
            groups.append([s])

        # For each group keep best
        after_sim = []
        for grp in groups:
            if len(grp) == 1:
                after_sim.append(grp[0])
            else:
                best, removed = choose_best_in_phash_group(grp, analysis["metrics"])
                after_sim.append(best)
                res["removed_similar"].extend([str(x) for x in removed])

        # 3) Quality filter (adaptive): compute per-class mean of combined metric and use factor
        # Combined metric per img
        combined_map = {}
        for p in after_sim:
            m = analysis["metrics"].get(str(p), {})
            lap = m.get("lap_var",0.0)/200.0
            cont = m.get("contrast",0.0)/50.0
            exp = (m.get("exposure",0.0)/255.0)
            sat = m.get("saturation",0.0)/60.0
            combined = 0.5*lap + 0.35*cont + 0.1*exp + 0.05*sat
            combined_map[p] = combined

        mean_combined = float(np.mean(list(combined_map.values()))) if combined_map else 0.0
        adaptive_thresh = max(args.min_combined, mean_combined * args.adaptive_factor)

        filtered = []
        for p,score in combined_map.items():
            if score >= adaptive_thresh:
                filtered.append(p)
            else:
                res["removed_low_quality"].append(str(p))

        # If less than 1 remain, relax threshold to min_combined
        if len(filtered) == 0:
            filtered = [p for p,sc in combined_map.items() if sc >= args.min_combined/10]  # very lenient fallback
            if not filtered:
                # keep top-1 by score
                if combined_map:
                    bestp = max(combined_map.keys(), key=lambda x: combined_map[x])
                    filtered = [bestp]

        # 4) If more than max_per_dir, select by diversity+quality
        if len(filtered) > args.max_per_dir:
            # sort by combined score descending
            sorted_by_score = sorted(filtered, key=lambda x: combined_map[x], reverse=True)
            # take top 2*max as pool
            pool = sorted_by_score[: min(len(sorted_by_score), args.max_per_dir * 3)]
            # select diverse by HSV hist greedily
            selected = select_diverse_by_hist(pool, args.max_per_dir)
            final_selection = selected
        else:
            final_selection = filtered

        # 5) Segmentation (sequential to avoid GPU contention). Cache masks per image
        masks_cache_dir = out_root / "_masks" / class_dir.name
        masks_cache_dir.mkdir(parents=True, exist_ok=True)
        out_class_dir = out_root / class_dir.name
        out_class_dir.mkdir(parents=True, exist_ok=True)

        for p in final_selection:
            try:
                img = safe_read_image(p)
                # mask cache key
                mask_path = masks_cache_dir / (p.name + ".png")
                if mask_path.exists() and not args.force_recalc:
                    mask = cv2.imread(str(mask_path), cv2.IMREAD_GRAYSCALE)
                else:
                    mask = u2net_segment_mask(img, u2_model, device)
                    if mask is None:
                        raise RuntimeError("Segmentation returned None")
                    cv2.imwrite(str(mask_path), mask)

                out_img = apply_mask_white_bg(img, mask)
                # resize result to target_size with padding while preserving aspect
                if args.target_size:
                    th = args.target_size
                    h,w = out_img.shape[:2]
                    scale = th / max(h,w)
                    new_w = int(w*scale); new_h = int(h*scale)
                    resized = cv2.resize(out_img, (new_w, new_h), interpolation=cv2.INTER_AREA)
                    canvas = np.ones((th,th,3), dtype=np.uint8)*255
                    sx = (th - new_w)//2
                    sy = (th - new_h)//2
                    canvas[sy:sy+new_h, sx:sx+new_w] = resized
                    out_img = canvas

                out_path = out_class_dir / p.name
                # save using numpy->file to handle unicode names robustly on Windows
                ext_ok = cv2.imencode('.jpg', out_img, [int(cv2.IMWRITE_JPEG_QUALITY), 95])[1].tobytes()
                with open(out_path, "wb") as fo:
                    fo.write(ext_ok)
                res["segmented"].append(str(out_path))
            except Exception as e:
                res["errors"].append({"file": str(p), "error": str(e)})

        # metadata and log
        meta = {
            "class": class_dir.name,
            "initial_count": res["initial"],
            "kept_count": len(final_selection),
            "removed_exact": res["removed_exact"],
            "removed_similar": res["removed_similar"],
            "removed_low_quality": res["removed_low_quality"],
            "segmented_files": res["segmented"],
            "errors": res["errors"]
        }
        # write meta json
        try:
            with open(out_class_dir / "_meta.json", "w", encoding="utf-8") as f:
                json.dump(meta, f, ensure_ascii=False, indent=2)
        except Exception:
            pass

        # optionally delete poor originals
        if args.delete_bad:
            for pstr in (res["removed_exact"] + res["removed_similar"] + res["removed_low_quality"]):
                try:
                    p = Path(pstr)
                    if p.exists():
                        p.unlink()
                except Exception:
                    pass

        res["kept"] = [str(x) for x in final_selection]
        return res

    except Exception as e:
        return {"class": class_dir.name, "error": str(e)}

# -----------------------
# Main CLI
# -----------------------
def main():
    parser = argparse.ArgumentParser(description="Preprocess images: dedup, quality filter, U2Net segmentation.")
    parser.add_argument("--src", required=True, help="Source root with class subdirs")
    parser.add_argument("--dst", required=True, help="Destination root for cleaned images")
    parser.add_argument("--max-per-dir", type=int, default=20, help="Max images to keep per class")
    parser.add_argument("--phash-threshold", type=float, default=0.10,
                        help="Normalized phash hamming distance threshold for similarity (0..1)")
    parser.add_argument("--min-combined", type=float, default=0.05,
                        help="Absolute minimal combined quality (0..1)")
    parser.add_argument("--adaptive-factor", type=float, default=0.7,
                        help="Adaptive thresh = mean_combined * adaptive_factor")
    parser.add_argument("--delete-bad", action="store_true", help="Delete bad originals (irreversible)")
    parser.add_argument("--device", default="auto", help="Device for segmentation: 'auto'|'cpu'|'cuda'")
    parser.add_argument("--workers", type=int, default=4, help="Number of worker threads for initial analysis")
    parser.add_argument("--force-recalc", action="store_true", help="Force recalc of caches")
    parser.add_argument("--target-size", type=int, default=512, help="Final image size (square) after padding")
    parser.add_argument("--log", default="preprocess_images_log.json", help="Output JSON log")
    args = parser.parse_args()

    src = Path(args.src)
    dst = Path(args.dst)
    dst.mkdir(parents=True, exist_ok=True)

    # device selection
    if args.device == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"
    else:
        device = args.device
    print("Using device:", device)

    # prepare u2net model
    download_u2net_if_missing()
    u2_model, map_loc = load_u2net(device)

    # class dirs
    class_dirs = sorted([p for p in src.iterdir() if p.is_dir()])

    overall_log = {"total_classes": len(class_dirs), "classes": {}}

    # Stage 1: parallel analysis (md5, phash, metrics) using ThreadPoolExecutor
    print("Stage 1: analyzing classes (parallel)...")
    with ThreadPoolExecutor(max_workers=max(1, args.workers)) as exe:
        futures = {exe.submit(analyze_class_dir, cd, args.force_recalc): cd for cd in class_dirs}
        analysis_results = {}
        for fut in tqdm(as_completed(futures), total=len(futures)):
            cls = futures[fut]
            try:
                analysis_results[cls] = fut.result()
            except Exception as e:
                analysis_results[cls] = {"error": str(e)}

    # Stage 2: processing classes sequentially with segmentation (GPU safe)
    print("Stage 2: processing classes (segmentation on device)...")
    for cls in tqdm(class_dirs):
        try:
            result = process_class(cls, dst, u2_model, map_loc, args)
            overall_log["classes"][cls.name] = result
        except Exception as e:
            overall_log["classes"][cls.name] = {"error": str(e)}

    # save overall log
    try:
        with open(args.log, "w", encoding="utf-8") as fo:
            json.dump(overall_log, fo, ensure_ascii=False, indent=2)
        print("Log written to", args.log)
    except Exception as e:
        print("Could not write log:", e)

if __name__ == "__main__":
    main()
