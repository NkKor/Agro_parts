# src/search/searcher.py
"""Поисковая система с поддержкой совместимости форматов"""

import numpy as np
import torch
import faiss
import pickle
import cv2 as cv
import torchvision.transforms as T
from pathlib import Path
from typing import List, Tuple, Dict, Set, Optional
from collections import Counter
import logging
import sys
from datetime import datetime
import os
# import time


# --- Настройка путей ---
current_file = Path(__file__).resolve()
project_root = current_file.parent.parent.parent
utils_path = project_root / "utils"
src_path = project_root / "src"

# --- Создание директорий для логов ---
logs_dir = project_root / "logs"
logs_dir.mkdir(parents=True, exist_ok=True)

sys.path.insert(0, str(project_root))
sys.path.insert(0, str(utils_path))
sys.path.insert(0, str(src_path))

# --- Настройка TORCH_HOME ---
torch_home = project_root / 'data' / 'models'
torch_home.mkdir(parents=True, exist_ok=True)
os.environ['TORCH_HOME'] = str(torch_home)

# --- Импорт конфигурации и утилит ---
try:
    import utils.config as config
    from utils.utils_cv import find_largest_foreground_bbox, pad_bbox, center_square_crop, resize_high_quality
    from src.models.encoder import ResNet50Encoder
except ImportError as e:
    print(f" Ошибка импорта в searcher.py: {e}")
    raise

# --- Настройка логирования ---
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s %(name)-12s %(levelname)-8s %(message)s',
    datefmt='%m-%d %H:%M',
    handlers=[
        logging.FileHandler(logs_dir / 'searcher.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

class SearchEngine:
    """Поисковая система с поддержкой совместимости форматов"""

    def __init__(self):
        """Инициализация поисковой системы с поддержкой совместимости"""
        logger.info(" Инициализация поисковой системы...")
        
        # Определение устройства
        self.device = "cuda" if torch.cuda.is_available() and getattr(config, 'USE_CUDA', False) else "cpu"
        logger.info(f" Используемое устройство: {self.device}")

        # Загрузка модели
        self.model = ResNet50Encoder(out_dim=getattr(config, 'EMB_DIM', 2048), pretrained=True)
        self.model = self.model.to(self.device).eval()
        logger.info(" Модель загружена.")

        # Загрузка данных с поддержкой совместимости
        self._load_compatible_data()
        
        # Трансформации для изображений
        self.transform = T.Compose([
            T.ToTensor(),
            T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])

    def _load_compatible_data(self):
        """Загрузка данных с поддержкой совместимости форматов"""
        emb_dir = Path(getattr(config, 'EMB_DIR', 'data/embeddings'))
        centroids_dir = Path(getattr(config, 'CENTROIDS_DIR', 'data/centroids'))
        indexes_dir = Path(getattr(config, 'INDEXES_DIR', 'data/indexes'))
        
        logger.info(f" EMB_DIR: {emb_dir}")
        logger.info(f" CENTROIDS_DIR: {centroids_dir}")
        logger.info(f" INDEXES_DIR: {indexes_dir}")
        
        # Попытка загрузки в новом формате
        success = self._load_new_format(emb_dir, centroids_dir, indexes_dir)
        
        # Если новый формат не работает, пробуем старый
        if not success:
            logger.warning("  Новый формат не найден, пробуем старый...")
            success = self._load_old_format(emb_dir)
        
        if not success:
            raise RuntimeError("Не удалось загрузить данные ни в новом, ни в старом формате")
        
        logger.info(" Данные загружены успешно")

    def _load_new_format(self, emb_dir: Path, centroids_dir: Path, indexes_dir: Path) -> bool:
        """Загрузка данных в новом формате (разделенные директории)"""
        try:
            # 1. Загрузка эмбеддингов изображений
            per_image_file = emb_dir / "per_image.npy"
            part_ids_file = emb_dir / "part_ids.npy"
            
            if not per_image_file.exists() or not part_ids_file.exists():
                logger.warning(" Новые файлы эмбеддингов не найдены")
                return False
            
            self.per_image_embeddings = np.load(per_image_file).astype(np.float32)
            self.per_image_ids = np.load(part_ids_file, allow_pickle=True)
            logger.info(f" Загружено {len(self.per_image_embeddings)} эмбеддингов изображений")
            
            # 2. Загрузка центроидов
            per_part_file = centroids_dir / "per_part.npy"
            part_names_file = centroids_dir / "part_names.npy"
            
            if not per_part_file.exists() or not part_names_file.exists():
                logger.warning(" Новые файлы центроидов не найдены")
                return False
            
            self.centroids = np.load(per_part_file).astype(np.float32)
            self.centroid_ids = np.load(part_names_file, allow_pickle=True)
            logger.info(f" Загружено {len(self.centroids)} центроидов")
            
            # 3. Загрузка FAISS индексов
            image_index_file = indexes_dir / "image_index.faiss"
            centroid_index_file = indexes_dir / "centroid_index.faiss"
            
            if image_index_file.exists():
                self.image_index = faiss.read_index(str(image_index_file))
                logger.info(f" FAISS индекс изображений загружен ({self.image_index.ntotal} векторов)")
            else:
                logger.warning(" FAISS индекс изображений не найден")
                return False
            
            if centroid_index_file.exists():
                self.centroid_index = faiss.read_index(str(centroid_index_file))
                logger.info(f" FAISS индекс центроидов загружен ({self.centroid_index.ntotal} векторов)")
            else:
                logger.warning(" FAISS индекс центроидов не найден")
                return False
            
            return True
            
        except Exception as e:
            logger.error(f" Ошибка загрузки нового формата: {e}")
            return False

    def _load_old_format(self, emb_dir: Path) -> bool:
        """Загрузка данных в старом формате (все в одной директории)"""
        try:
            # 1. Попытка загрузки embeddings.pkl
            pkl_file = emb_dir / "embeddings.pkl"
            if pkl_file.exists():
                logger.info(" Загрузка embeddings.pkl...")
                with open(pkl_file, 'rb') as f:
                    data = pickle.load(f)
                
                if isinstance(data, dict):
                    # Старый формат: {'embeddings': array, 'part_ids': list}
                    self.per_image_embeddings = data['embeddings'].astype(np.float32)
                    self.per_image_ids = np.array(data['part_ids'], dtype=object)
                    logger.info(f" Загружено {len(self.per_image_embeddings)} эмбеддингов из embeddings.pkl")
                elif isinstance(data, (list, tuple)) and len(data) >= 2:
                    # Формат: [embeddings, part_ids]
                    self.per_image_embeddings = data[0].astype(np.float32)
                    self.per_image_ids = np.array(data[1], dtype=object)
                    logger.info(f" Загружено {len(self.per_image_embeddings)} эмбеддингов из embeddings.pkl")
                else:
                    logger.error(" Неподдерживаемый формат embeddings.pkl")
                    return False
            
            # 2. Загрузка отдельных файлов (если pkl не найден)
            else:
                per_image_file = emb_dir / "per_image.npy"
                part_ids_file = emb_dir / "part_ids.npy"
                
                if not per_image_file.exists() or not part_ids_file.exists():
                    logger.error(" Старые файлы эмбеддингов не найдены")
                    return False
                
                self.per_image_embeddings = np.load(per_image_file).astype(np.float32)
                self.per_image_ids = np.load(part_ids_file, allow_pickle=True)
                logger.info(f" Загружено {len(self.per_image_embeddings)} эмбеддингов изображений")
            
            # 3. Загрузка центроидов
            per_part_file = emb_dir / "per_part.npy"
            part_names_file = emb_dir / "part_names.npy"
            
            if not per_part_file.exists() or not part_names_file.exists():
                logger.error(" Старые файлы центроидов не найдены")
                return False
            
            self.centroids = np.load(per_part_file).astype(np.float32)
            self.centroid_ids = np.load(part_names_file, allow_pickle=True)
            logger.info(f" Загружено {len(self.centroids)} центроидов")
            
            # 4. Загрузка FAISS индексов
            image_index_file = emb_dir / "image_index.faiss"
            centroid_index_file = emb_dir / "centroid_index.faiss"
            
            if not image_index_file.exists() or not centroid_index_file.exists():
                logger.error(" Старые FAISS индексы не найдены")
                return False
            
            self.image_index = faiss.read_index(str(image_index_file))
            self.centroid_index = faiss.read_index(str(centroid_index_file))
            logger.info(f" FAISS индексы загружены")
            
            return True
            
        except Exception as e:
            logger.error(f" Ошибка загрузки старого формата: {e}")
            return False

    def preprocess_image(self, img_bgr: np.ndarray) -> torch.Tensor:
        """Предобработка изображения перед подачей в модель"""
        try:
            bbox = find_largest_foreground_bbox(img_bgr, min_area_ratio=getattr(config, 'MIN_OBJ_AREA', 0.01))
            if bbox is not None:
                bbox = pad_bbox(bbox, img_bgr.shape, pad_ratio=getattr(config, 'PAD_RATIO', 0.1))
                x1, y1, x2, y2 = bbox
                crop = img_bgr[y1:y2, x1:x2]
            else:
                crop = center_square_crop(img_bgr)
            
            crop = resize_high_quality(crop, getattr(config, 'TARGET_SIZE', 384))
            
            img_rgb = cv.cvtColor(crop, cv.COLOR_BGR2RGB)
            import torchvision.transforms.functional as F
            pil = F.to_pil_image(img_rgb)
            tensor_img = self.transform(pil).unsqueeze(0).to(self.device)
            
            return tensor_img
        except Exception as e:
            logger.error(f" Ошибка предобработки изображения: {e}")
            raise

    @torch.no_grad()
    def extract_embeddings(self, images: List[np.ndarray]) -> np.ndarray:
        """Извлечение и усреднение эмбеддингов из списка изображений"""
        embeddings = []
        for img in images:
            try:
                tensor_img = self.preprocess_image(img)
                emb = self.model(tensor_img).cpu().numpy()
                emb = emb / (np.linalg.norm(emb, axis=1, keepdims=True) + 1e-12)
                embeddings.append(emb)
            except Exception as e:
                logger.warning(f"  Ошибка извлечения эмбеддинга: {e}")
                continue
        
        if not embeddings:
            raise ValueError("Не удалось извлечь ни один эмбеддинг")
        
        stacked_embs = np.vstack(embeddings)
        avg_emb = np.mean(stacked_embs, axis=0, keepdims=True)
        avg_emb = avg_emb / (np.linalg.norm(avg_emb, axis=1, keepdims=True) + 1e-12)
        
        return avg_emb.astype(np.float32)

    def search(self, query_emb: np.ndarray, top_k: int = 5) -> Dict[str, float]:
        """Поиск похожих деталей"""
        try:
            query_emb = query_emb / (np.linalg.norm(query_emb, axis=1, keepdims=True) + 1e-12)
            
            k_search = min(top_k * 10, len(self.centroids))
            D_centroids, I_centroids = self.centroid_index.search(query_emb, k_search)
            
            part_scores = {}
            for dist, idx in zip(D_centroids[0], I_centroids[0]):
                try:
                    part_id = self.centroid_ids[idx]
                    similarity = max(0, min(100, 100 * np.exp(-dist/2)))
                    
                    if part_id not in part_scores:
                        part_scores[part_id] = []
                    part_scores[part_id].append(similarity)
                except Exception as e:
                    continue

            aggregated_scores = {}
            for part_id, scores in part_scores.items():
                if scores:
                    aggregated_scores[part_id] = np.mean(scores)
            
            sorted_results = sorted(aggregated_scores.items(), key=lambda x: x[1], reverse=True)
            top_results = dict(sorted_results[:top_k])
            
            logger.info(f" Поиск завершен. Найдено {len(top_results)} результатов")
            return top_results
            
        except Exception as e:
            logger.error(f" Ошибка в search: {e}")
            raise

    def predict_search(self, images: List[np.ndarray], top_k: int = 5) -> Dict:
        """Основной метод предсказания"""
        try:
            logger.info(f" Начало предсказания для {len(images)} изображений")
            
            query_emb = self.extract_embeddings(images)
            logger.debug(" Эмбеддинг запроса извлечен")

            similarities = self.search(query_emb, top_k=top_k)
            logger.debug(" Поиск завершен")

            pred_classes = list(similarities.keys())
            rank = {part_id: idx + 1 for idx, part_id in enumerate(pred_classes)}
            
            split_idx = len(pred_classes) // 2
            more_possible_classes = pred_classes[:split_idx] if split_idx > 0 else pred_classes
            another_possible_classes = pred_classes[split_idx:] if split_idx < len(pred_classes) else []
            
            response_data = {
                'pred_classes': pred_classes,
                #'rank': rank,
                #'more_possible_classes': more_possible_classes,
                #'another_possible_classes': another_possible_classes,
                'similarities': similarities
            }
            
            logger.info(" Предсказание завершено")
            return response_data

        except Exception as e:
            logger.error(f" Ошибка в predict_search: {e}", exc_info=True)
            return {
                'pred_classes': [],
                #'rank': {},
                #'more_possible_classes': [],
                #'another_possible_classes': [],
                'similarities': {}
            }

# --- Обратная совместимость для старого API ---
def load_search_engine():
    """Загрузка поисковой системы с обратной совместимостью"""
    try:
        search_engine = SearchEngine()
        return search_engine
    except Exception as e:
        logger.error(f" Ошибка загрузки поисковой системы: {e}")
        return None

# --- Для использования в API ---
search_engine = None

def init_search_engine():
    """Инициализация поисковой системы при запуске"""
    global search_engine
    if search_engine is None:
        search_engine = load_search_engine()
    return search_engine is not None
