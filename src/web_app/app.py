# src/web_app/app.py
import os
import io
from pathlib import Path
import sys
from flask import Flask, request, render_template, redirect, url_for, send_from_directory, jsonify
import cv2 as cv
import numpy as np
import torch
import torchvision.transforms as T
import faiss
from werkzeug.utils import secure_filename

# Добавляем пути к проекту для корректных импортов
project_root = Path(__file__).parent.parent.parent
utils_path = project_root / "utils"
sys.path.insert(0, str(project_root))
sys.path.insert(0, str(utils_path))

# Импорт наших модулей с правильными путями
try:
    # Пробуем разные варианты импорта
    import utils.config as config
    from utils.utils_cv import find_largest_foreground_bbox, pad_bbox, center_square_crop, resize_high_quality
    from src.models.encoder import ResNet50Encoder
    print("✅ Импорты загружены успешно (вариант 1)")
except ImportError:
    try:
        # Альтернативные пути
        import config as config
        from utils_cv import find_largest_foreground_bbox, pad_bbox, center_square_crop, resize_high_quality
        from models.encoder import ResNet50Encoder
        print("✅ Импорты загружены успешно (вариант 2)")
    except ImportError:
        try:
            # Еще один вариант
            sys.path.append(str(Path(__file__).parent.parent))
            import utils.config as config
            from utils.utils_cv import find_largest_foreground_bbox, pad_bbox, center_square_crop, resize_high_quality
            from src.models.encoder import ResNet50Encoder
            print("✅ Импорты загружены успешно (вариант 3)")
        except ImportError as e:
            print(f"❌ Ошибка импорта: {e}")
            print("💡 Проверьте структуру проекта:")
            print("   project/")
            print("   ├── src/")
            print("   │   ├── models/encoder.py")
            print("   │   └── web_app/app.py")
            print("   └── utils/")
            print("       ├── config.py")
            print("       └── utils_cv.py")
            sys.exit(1)

# Путь для временных загрузок
TMP_DIR = Path("tmp_uploads")
TMP_DIR.mkdir(exist_ok=True)

ALLOWED_EXT = {".jpg", ".jpeg", ".png"}

# Flask приложение
app = Flask(__name__, 
           template_folder='templates',
           static_folder='static')
app.config['MAX_CONTENT_LENGTH'] = 20 * 1024 * 1024  # 20 MB per request guard

# Глобальные переменные для модели и индексов
model = None
CENTROIDS = None
CENTROID_IDS = None
PERIMG = None
PERIMG_IDS = None
IDX_C = None
IDX_X = None

# трансформ для модели
transform = T.Compose([
    T.ToTensor(),
    T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

def init_model_and_indexes():
    """Инициализация модели и индексов при запуске"""
    global model, CENTROIDS, CENTROID_IDS, PERIMG, PERIMG_IDS, IDX_C, IDX_X
    
    print("🔄 Загружаем модель и индексы...")
    
    # Определяем устройство
    device = "cpu"
    if torch.cuda.is_available():
        try:
            x = torch.zeros(1).cuda()
            del x
            device = "cuda"
            print("✅ Используется CUDA")
        except:
            print("⚠️  CUDA доступна, но не поддерживается")
    print(f"🔧 Устройство: {device}")
    
    # Загрузка модели
    try:
        model = ResNet50Encoder(out_dim=2048, pretrained=True).to(device).eval()
        print("✅ Модель загружена")
    except Exception as e:
        print(f"❌ Ошибка загрузки модели: {e}")
        return False
    
    # Проверка существования файлов
    emb_dir = Path(getattr(config, 'EMB_DIR', 'data/embeddings'))
    if not emb_dir.exists():
        print(f"❌ Директория эмбеддингов не найдена: {emb_dir}")
        return False
    
    # Проверяем разные возможные имена файлов
    centroid_files = [
        emb_dir / "per_part.npy",
        emb_dir / "centroids.npy",
        emb_dir / "centroid_vectors.npy"
    ]
    
    centroid_id_files = [
        emb_dir / "part_names.npy",
        emb_dir / "centroid_ids.npy",
        emb_dir / "centroid_names.npy"
    ]
    
    perimg_files = [
        emb_dir / "per_image.npy",
        emb_dir / "embeddings.npy"
    ]
    
    perimg_id_files = [
        emb_dir / "part_ids.npy",
        emb_dir / "image_ids.npy"
    ]
    
    # Ищем существующие файлы
    centroid_file = None
    centroid_ids_file = None
    perimg_file = None
    perimg_ids_file = None
    
    for f in centroid_files:
        if f.exists():
            centroid_file = f
            break
    
    for f in centroid_id_files:
        if f.exists():
            centroid_ids_file = f
            break
            
    for f in perimg_files:
        if f.exists():
            perimg_file = f
            break
            
    for f in perimg_id_files:
        if f.exists():
            perimg_ids_file = f
            break
    
    # Проверяем, что все необходимые файлы найдены
    missing_files = []
    if not centroid_file:
        missing_files.append("файл центроидов")
    if not centroid_ids_file:
        missing_files.append("файл ID центроидов")
    if not perimg_file:
        missing_files.append("файл изображений")
    if not perimg_ids_file:
        missing_files.append("файл ID изображений")
    
    if missing_files:
        print("❌ Не найдены необходимые файлы:")
        for f in missing_files:
            print(f"   - {f}")
        print("\n💡 Выполните команды:")
        print("   python src/build_centroids.py --embeddings data/embeddings --out data/embeddings")
        return False
    
    print(f"✅ Найдены файлы:")
    print(f"   Центроиды: {centroid_file}")
    print(f"   ID центроидов: {centroid_ids_file}")
    print(f"   Изображения: {perimg_file}")
    print(f"   ID изображений: {perimg_ids_file}")
    
    # Загрузка эмбеддингов и индексов
    try:
        CENTROIDS = np.load(centroid_file).astype(np.float32)
        CENTROID_IDS = np.load(centroid_ids_file, allow_pickle=True)
        PERIMG = np.load(perimg_file).astype(np.float32)
        PERIMG_IDS = np.load(perimg_ids_file, allow_pickle=True)
        
        print(f"✅ Загружено центроидов: {len(CENTROIDS)}")
        print(f"✅ Загружено изображений: {len(PERIMG)}")
    except Exception as e:
        print(f"❌ Ошибка загрузки эмбеддингов: {e}")
        return False
    
    # Загрузка индексов FAISS (ищем в той же директории)
    idx_files = [
        emb_dir / "centroid_index.faiss",
        emb_dir / "faiss_centroid.bin",
        emb_dir / "centroids.index"
    ]
    
    idx_img_files = [
        emb_dir / "image_index.faiss",
        emb_dir / "faiss_img.bin",
        emb_dir / "images.index"
    ]
    
    idx_c_file = None
    idx_x_file = None
    
    for f in idx_files:
        if f.exists():
            idx_c_file = f
            break
    
    for f in idx_img_files:
        if f.exists():
            idx_x_file = f
            break
    
    if not idx_c_file:
        print("❌ Индекс центроидов не найден")
        print("💡 Выполните команду:")
        print("   python src/build_index.py --embeddings data/embeddings --centroids data/embeddings --out data/embeddings")
        return False
    
    try:
        IDX_C = faiss.read_index(str(idx_c_file))
        print("✅ Индекс центроидов загружен")
        
        if idx_x_file and idx_x_file.exists():
            IDX_X = faiss.read_index(str(idx_x_file))
            print("✅ Индекс изображений загружен")
        else:
            IDX_X = None
            print("⚠️  Индекс изображений не найден")
            
    except Exception as e:
        print(f"❌ Ошибка загрузки индексов: {e}")
        return False
    
    print("🎉 Все данные загружены успешно!")
    return True

def allowed_file(filename):
    return Path(filename).suffix.lower() in ALLOWED_EXT

def preprocess_bgr_save(img_bgr):
    """Используем ту же логику, что в оффлайн preprocess: ROI->pad->resize."""
    try:
        bbox = find_largest_foreground_bbox(img_bgr, min_area_ratio=getattr(config, 'MIN_OBJ_AREA', 0.01))
        if bbox is not None:
            bbox = pad_bbox(bbox, img_bgr.shape, pad_ratio=getattr(config, 'PAD_RATIO', 0.1))
            x1, y1, x2, y2 = bbox
            crop = img_bgr[y1:y2, x1:x2]
        else:
            crop = center_square_crop(img_bgr)
        out = resize_high_quality(crop, getattr(config, 'TARGET_SIZE', 384))
        return out
    except Exception as e:
        print(f"⚠️  Ошибка предобработки: {e}")
        # fallback на простую обрезку
        try:
            return center_square_crop(img_bgr)
        except:
            return img_bgr

@torch.no_grad()
def embed_from_bgr(img_bgr):
    """Возвращает L2-нормированный эмбеддинг (1,D) numpy float32."""
    try:
        out = preprocess_bgr_save(img_bgr)
        # convert BGR->RGB and to PIL via torchvision.functional or to tensor directly
        import torchvision.transforms.functional as F
        img_rgb = cv.cvtColor(out, cv.COLOR_BGR2RGB)
        pil = F.to_pil_image(img_rgb)
        x = transform(pil).unsqueeze(0).to(next(model.parameters()).device)
        emb = model(x).cpu().numpy()
        # ensure float32 и нормализация
        emb = emb.astype('float32')
        emb = emb / (np.linalg.norm(emb, axis=1, keepdims=True) + 1e-12)
        return emb
    except Exception as e:
        print(f"❌ Ошибка извлечения эмбеддинга: {e}")
        # Возвращаем нулевой вектор в случае ошибки
        return np.zeros((1, 2048), dtype=np.float32)

def search_by_emb(emb, topk=5, rerank_per_image=True, per_image_k=200):
    """Поиск: сначала по центроидам, затем re-rank по per-image (max-aggregation)."""
    try:
        # Нормируем (на всякий случай)
        emb = emb / (np.linalg.norm(emb, axis=1, keepdims=True) + 1e-12)
        
        # быстрый поиск по центроидам (возвращаем per_image_k кандидатов для re-rank)
        k_for_search = min(per_image_k, len(CENTROIDS)) if rerank_per_image else min(topk, len(CENTROIDS))
        if k_for_search == 0:
            return []
            
        D, I = IDX_C.search(emb.astype('float32'), k_for_search)
        cand_part_ids = [CENTROID_IDS[i] for i in I[0]]

        if not rerank_per_image:
            # Конвертируем расстояния в проценты похожести
            # Для L2 расстояний: чем меньше расстояние, тем больше похожесть
            # Используем экспоненциальное преобразование для лучшего масштабирования
            similarity_scores = []
            for distance in D[0][:topk]:
                # Преобразуем расстояние в процент похожести (0-100%)
                # Используем обратную экспоненту: чем меньше расстояние, тем больше похожесть
                similarity = max(0, min(100, 100 * np.exp(-distance/2)))
                similarity_scores.append(similarity)
            return list(zip(cand_part_ids[:topk], D[0][:topk].tolist(), similarity_scores))

        # re-rank: возьмём все per-image векторы, принадлежащие candidate part ids
        mask = np.isin(PERIMG_IDS, np.array(cand_part_ids, dtype=object))
        if np.sum(mask) == 0:
            # Если нет изображений, возвращаем центроиды с преобразованными расстояниями
            similarity_scores = []
            for distance in D[0][:topk]:
                similarity = max(0, min(100, 100 * np.exp(-distance/2)))
                similarity_scores.append(similarity)
            return list(zip(cand_part_ids[:topk], D[0][:topk].tolist(), similarity_scores))
            
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
        
        # Конвертируем cosine similarity в проценты ([-1,1] -> [0,100])
        similarity_scores = []
        for pid, cos_sim in ranked:
            # Преобразуем cosine similarity в процент похожести
            similarity_percent = max(0, min(100, (cos_sim + 1) * 50))
            similarity_scores.append(similarity_percent)
            
        # Возвращаем [(part_id, distance, similarity_percent), ...]
        return [(ranked[i][0], 1-ranked[i][1], similarity_scores[i]) for i in range(len(ranked))]
        
    except Exception as e:
        print(f"❌ Ошибка поиска: {e}")
        return []

def sample_example_for_part(pid):
    """Возвращает путь к первому файлу в processed/<pid> для показа."""
    try:
        processed_dir = Path(getattr(config, 'DATA_PROCESSED', 'data/processed'))
        part_dir = processed_dir / str(pid)
        if not part_dir.exists():
            # Ищем в подкаталогах
            for subdir in processed_dir.iterdir():
                if subdir.is_dir() and subdir.name == str(pid):
                    part_dir = subdir
                    break
            else:
                return None
        
        if part_dir.exists():
            files = [p for p in part_dir.iterdir() if p.suffix.lower() in ALLOWED_EXT]
            return str(files[0]) if files else None
        return None
    except Exception as e:
        print(f"⚠️  Ошибка поиска примера для {pid}: {e}")
        return None

# --- Маршруты ---
@app.route("/", methods=["GET"])
def index():
    return render_template("index.html")

@app.route("/search", methods=["POST"])
def search_route():
    try:
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
            if not filename:
                continue
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
        for i, result in enumerate(results):
            if len(result) == 3:
                pid, distance, similarity_percent = result
            else:
                # fallback для старого формата
                pid, distance = result
                # Преобразуем расстояние в процент похожести
                similarity_percent = max(0, min(100, 100 * np.exp(-distance/2)))
            
            sample = sample_example_for_part(pid)
            out_results.append({
                "pid": str(pid), 
                "distance": float(distance), 
                "similarity": f"{similarity_percent:.1f}%",
                "sample": sample,
                "is_best": i == 0  # Помечаем лучший результат
            })

        # рендер
        return render_template("results.html", query_imgs=saved_query_paths, results=out_results)
        
    except Exception as e:
        print(f"❌ Ошибка в search_route: {e}")
        import traceback
        traceback.print_exc()
        return redirect(url_for("index"))

# служим временные загруженные файлы (для отображения)
@app.route("/uploads/<path:filename>")
def uploaded_file(filename):
    try:
        # Нормализуем путь для Windows
        normalized_filename = filename.replace('\\', '/')
        file_path = TMP_DIR / Path(normalized_filename).name
        if file_path.exists():
            return send_from_directory(str(TMP_DIR), Path(normalized_filename).name)
        else:
            print(f"❌ Файл не найден: {file_path}")
            return ("Not found", 404)
    except Exception as e:
        print(f"❌ Ошибка отдачи файла: {e}")
        return ("Not found", 404)

# служим примеры из processed (если нужно)
@app.route("/processed/<path:filepath>")
def processed_img(filepath):
    try:
        # Нормализуем путь
        normalized_filepath = filepath.replace('\\', '/')
        processed_dir = Path(getattr(config, 'DATA_PROCESSED', 'data/processed'))
        full_path = processed_dir / normalized_filepath
        
        if full_path.exists():
            return send_from_directory(str(full_path.parent), full_path.name)
        else:
            print(f"❌ Processed файл не найден: {full_path}")
            return ("Not found", 404)
    except Exception as e:
        print(f"❌ Ошибка отдачи processed файла: {e}")
        return ("Not found", 404)

# API endpoint для программного использования
@app.route("/api/search", methods=["POST"])
def api_search():
    """API endpoint для поиска"""
    try:
        files = request.files.getlist("images")
        files = [f for f in files if f and f.filename and allowed_file(f.filename)]
        
        if len(files) == 0:
            return jsonify({"error": "No valid images provided"}), 400

        embeddings = []
        for f in files[:3]:
            # Читаем изображение в память
            img_bytes = f.read()
            img_array = np.frombuffer(img_bytes, np.uint8)
            img = cv.imdecode(img_array, cv.IMREAD_COLOR)
            
            if img is not None:
                emb = embed_from_bgr(img)
                embeddings.append(emb)

        if len(embeddings) == 0:
            return jsonify({"error": "Could not process any images"}), 400

        # Усредняем эмбеддинги
        stacked = np.vstack(embeddings)
        query_emb = np.mean(stacked, axis=0, keepdims=True)
        query_emb = query_emb / (np.linalg.norm(query_emb, axis=1, keepdims=True) + 1e-12)

        results = search_by_emb(query_emb, topk=5, rerank_per_image=True)
        
        # Форматируем результаты
        formatted_results = []
        for i, result in enumerate(results):
            if len(result) == 3:
                pid, distance, similarity_percent = result
            else:
                # fallback для старого формата
                pid, distance = result
                similarity_percent = max(0, min(100, 100 * np.exp(-distance/2)))
                
            formatted_results.append({
                "part_id": str(pid),
                "distance": float(distance),
                "similarity_percent": f"{similarity_percent:.1f}%",
                "is_best": i == 0
            })

        return jsonify({
            "success": True,
            "results": formatted_results
        })
        
    except Exception as e:
        print(f"❌ API ошибка: {e}")
        import traceback
        traceback.print_exc()
        return jsonify({"error": str(e)}), 500

@app.route("/health")
def health_check():
    """Health check endpoint"""
    return jsonify({
        "status": "healthy",
        "model_loaded": model is not None,
        "indexes_loaded": IDX_C is not None,
        "centroids_count": len(CENTROIDS) if CENTROIDS is not None else 0,
        "images_count": len(PERIMG) if PERIMG is not None else 0
    })

if __name__ == "__main__":
    # Инициализация модели и индексов
    if not init_model_and_indexes():
        print("❌ Не удалось инициализировать модель и индексы")
        sys.exit(1)
    
    print("🚀 Запуск Flask приложения...")
    app.run(host="0.0.0.0", port=5000, debug=True)