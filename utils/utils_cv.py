import cv2 as cv
import numpy as np

def variance_of_laplacian(gray: np.ndarray) -> float:
    return cv.Laplacian(gray, cv.CV_64F).var()

def overexposed_ratio(img: np.ndarray, thr: int = 245) -> float:
    # доля очень светлых пикселей
    return (img >= thr).mean()

def find_largest_foreground_bbox(img_bgr: np.ndarray, min_area_ratio=0.12):
    h, w = img_bgr.shape[:2]
    # усилить границы
    gray = cv.cvtColor(img_bgr, cv.COLOR_BGR2GRAY)
    blur = cv.GaussianBlur(gray, (5,5), 0)
    edges = cv.Canny(blur, 50, 150)
    # морфология
    kernel = cv.getStructuringElement(cv.MORPH_RECT, (5,5))
    edges = cv.morphologyEx(edges, cv.MORPH_CLOSE, kernel, iterations=2)
    # контуры
    cnts, _ = cv.findContours(edges, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)
    if not cnts:
        return None
    cnt = max(cnts, key=cv.contourArea)
    x,y,wc,hc = cv.boundingRect(cnt)
    area = wc*hc / (w*h)
    if area < min_area_ratio:
        return None
    return (x,y, x+wc, y+hc)

def pad_bbox(bbox, img_shape, pad_ratio=0.06):
    h, w = img_shape[:2]
    x1,y1,x2,y2 = bbox
    bw, bh = x2-x1, y2-y1
    pad_x, pad_y = int(bw*pad_ratio), int(bh*pad_ratio)
    nx1 = max(0, x1 - pad_x)
    ny1 = max(0, y1 - pad_y)
    nx2 = min(w, x2 + pad_x)
    ny2 = min(h, y2 + pad_y)
    return (nx1, ny1, nx2, ny2)

def center_square_crop(img_bgr: np.ndarray):
    h,w = img_bgr.shape[:2]
    side = min(h,w)
    x1 = (w - side)//2
    y1 = (h - side)//2
    return img_bgr[y1:y1+side, x1:x1+side]

def resize_high_quality(img_bgr: np.ndarray, size: int):
    return cv.resize(img_bgr, (size,size), interpolation=cv.INTER_LANCZOS4)
