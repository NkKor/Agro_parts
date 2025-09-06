# src/web_app/app.py
import os
import io
from pathlib import Path
from flask import Flask, request, render_template, redirect, url_for, send_from_directory
import cv2 as cv
import numpy as np
import torch
import torchvision.transforms as T
import faiss
from werkzeug.utils import secure_filename

# --- Импорт наших модулей (предполагаем, что PYTHONPATH включает project_root/src) ---
from config import DEVICE, TARGET_SIZE, PAD_RATIO, MIN_OBJ_AREA, EMB_DIR, IDX_DIR, PROC_DIR
from utils_cv import find_largest_foreground_bbox, pad_bbox, center_square_crop, resize_high_quality
from encoder import ResNet50Encoder

# Путь для временных загрузок
TMP_DIR = Path("tmp_uploads")
TMP_DIR.mkdir(exist_ok=True)

ALLOWED_EXT = {".jpg", ".jpeg", ".png"}

# Flask приложение
app = Flask(__name__)
app.config['MAX_CONTENT_LENGTH'] = 20 * 1024 * 1024  # 20 MB per request guard

# --- Загрузка модели и индексов на старте (один раз) ---
print("Загружаем модель и индексы...")
DEVICE = DEVICE if torch.cuda.is_available() else "cpu"

model = ResNet50Encoder().to(DEVICE).eval()

# загрузка эмбеддингов/индексов и id mapping
CENTROIDS = np.load(EMB_DIR/"centroids.npy")
CENTROID_IDS = np.load(EMB_DIR/"centroid_ids.npy", allow_pickle=True)

PERIMG = np.load(EMB_DIR/"per_image.npy")
PERIMG_IDS = np.load(EMB_DIR/"part_ids.npy", allow_pickle=True)

IDX_C = faiss.read_index(str(IDX_DIR/"faiss_centroid.bin"))
try:
    IDX_X = faiss.read_index(str(IDX_DIR/"faiss_img.bin"))
except Exception:
    IDX_X = None

print("Готово.")

# трансформ для модели (тот же, что использовался оффлайн)
transform = T.Compose([
    T.ToTensor(),
    T.Normalize(mean=[0.485,0.456,0.406], std=[0.229,0.224,0.225])
])

def allowed_file(filename):
    return Path(filename).suffix.lower() in ALLOWED_EXT

def preprocess_bgr_save(img_bgr):
    """Используем ту же логику, что в оффлайн preprocess: ROI->pad->resize."""
    bbox = find_largest_foreground_bbox(img_bgr, min_area_ratio=MIN_OBJ_AREA)
    if bbox is not None:
        bbox = pad_bbox(bbox, img_bgr.shape, pad_ratio=PAD_RATIO)
        x1,y1,x2,y2 = bbox
        crop = img_bgr[y1:y2, x1:x2]
    else:
        crop = center_square_crop(img_bgr)
    out = resize_high_quality(crop, TARGET_SIZE)
    return out

@torch.no_grad()
def embed_from_bgr(img_bgr):
    """Возвращает L2-нормированный эмбеддинг (1,D) numpy float32."""
    out = preprocess_bgr_save(img_bgr)
    # convert BGR->RGB and to PIL via torchvision.functional or to tensor directly
    import torchvision.transforms.functional as F
    img_rgb = cv.cvtColor(out, cv.COLOR_BGR2RGB)
    pil = F.to_pil_image(img_rgb)
    x = transform(pil).unsqueeze(0).to(DEVICE)
    emb = model(x).cpu().numpy()
    # ensure float32
    return emb.astype('float32')

def search_by_emb(emb, topk=5, rerank_per_image=True, per_image_k=200):
    """Поиск: сначала по центроидам, затем re-rank по per-image (max-aggregation)."""
    # Нормируем (на всякий случай)
    emb = emb / (np.linalg.norm(emb, axis=1, keepdims=True) + 1e-12)
    # быстрый поиск по центроидам (возвращаем per_image_k кандидатов для re-rank)
    k_for_search = per_image_k if rerank_per_image else topk
    D, I = IDX_C.search(emb.astype('float32'), k_for_search)
    cand_part_ids = [CENTROID_IDS[i] for i in I[0]]

    if not rerank_per_image:
        return list(zip(cand_part_ids[:topk], D[0][:topk].tolist()))

    # re-rank: возьмём все per-image векторы, принадлежащие candidate part ids
    mask = np.isin(PERIMG_IDS, np.array(cand_part_ids, dtype=object))
    cand_vectors = PERIMG[mask]            # [M, D]
    cand_ids = PERIMG_IDS[mask]

    # similarity = dot, потому что L2-нормированные
    sims = (emb @ cand_vectors.T).ravel()
    best = {}
    for sim, pid in zip(sims, cand_ids):
        prev = best.get(pid)
        if prev is None or sim > prev:
            best[pid] = float(sim)
    ranked = sorted(best.items(), key=lambda x: x[1], reverse=True)[:topk]
    return ranked

def sample_example_for_part(pid):
    """Возвращает путь к первому файлу в processed/<pid> для показа."""
    part_dir = Path(PROC_DIR) / str(pid)
    if not part_dir.exists():
        # fallback: ищем в raw
        raw_dir = Path("data/raw") / str(pid)
        if raw_dir.exists():
            files = [p for p in raw_dir.iterdir() if p.suffix.lower() in ALLOWED_EXT]
            return str(files[0]) if files else None
        return None
    files = [p for p in part_dir.iterdir() if p.suffix.lower() in ALLOWED_EXT]
    return str(files[0]) if files else None

# --- Маршруты ---
@app.route("/", methods=["GET"])
def index():
    return render_template("index.html")

@app.route("/search", methods=["POST"])
def search_route():
    # ожидаем up to 3 файлов из поля 'images'
    files = request.files.getlist("images")
    # фильтруем пустые
    files = [f for f in files if f and f.filename and allowed_file(f.filename)]
    if len(files) == 0:
        return redirect(url_for("index"))

    embeddings = []
    saved_query_paths = []
    for f in files[:3]:
        filename = secure_filename(f.filename)
        tmp_path = TMP_DIR / filename
        f.save(tmp_path)
        saved_query_paths.append(str(tmp_path))
        # читаем и embed
        img = cv.imread(str(tmp_path), cv.IMREAD_COLOR)
        if img is None:
            continue
        emb = embed_from_bgr(img)   # (1, D)
        embeddings.append(emb)

    if len(embeddings) == 0:
        return redirect(url_for("index"))

    # усредняем эмбеддинги (по строкам)
    stacked = np.vstack(embeddings)   # [k, D]
    query_emb = np.mean(stacked, axis=0, keepdims=True)
    query_emb = query_emb / (np.linalg.norm(query_emb, axis=1, keepdims=True) + 1e-12)

    results = search_by_emb(query_emb, topk=5, rerank_per_image=True)

    # подготовим отображаемые примеры
    out_results = []
    for pid, score in results:
        sample = sample_example_for_part(pid)
        out_results.append({"pid": str(pid), "score": float(score), "sample": sample})

    # рендер
    return render_template("results.html", query_imgs=saved_query_paths, results=out_results)

# служим временные загруженные файлы (для отображения)
@app.route("/uploads/<path:filename>")
def uploaded_file(filename):
    return send_from_directory(str(TMP_DIR), filename)

# служим примеры из processed (если нужно)
@app.route("/processed_img/<path:imgpath>")
def processed_img(imgpath):
    # imgpath будет относительным: например "0001234/img1.jpg"
    full = Path(PROC_DIR) / Path(imgpath)
    if full.exists():
        return send_from_directory(str(full.parent), full.name)
    return ("Not found", 404)

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=True)
