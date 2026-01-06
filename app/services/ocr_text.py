import os
import tempfile
import numpy as np
from PIL import Image, ImageOps
import cv2
from rapidocr_onnxruntime import RapidOCR

_rapidocr = RapidOCR()

def _deskew_cv(gray: np.ndarray) -> np.ndarray:
    edges = cv2.Canny(gray, 50, 150)
    lines = cv2.HoughLines(edges, 1, np.pi/180, 120)
    if lines is None:
        return gray
    angles = []
    for l in lines[:50]:
        theta = l[0][1]
        ang = (theta * 180 / np.pi) - 90
        if -15 < ang < 15:
            angles.append(ang)
    if not angles:
        return gray
    angle = np.median(angles)
    (h, w) = gray.shape
    M = cv2.getRotationMatrix2D((w//2, h//2), angle, 1.0)
    return cv2.warpAffine(gray, M, (w, h), flags=cv2.INTER_CUBIC, borderMode=cv2.BORDER_REPLICATE)

def _preprocess_for_ocr(path: str) -> str:
    img = Image.open(path).convert("L")  # grayscale
    img = ImageOps.autocontrast(img, cutoff=2)
    arr = np.array(img)
    arr = _deskew_cv(arr)
    arr = cv2.resize(arr, None, fx=2.0, fy=2.0, interpolation=cv2.INTER_CUBIC)
    arr = cv2.bilateralFilter(arr, 7, 50, 50)
    arr = cv2.adaptiveThreshold(arr, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                cv2.THRESH_BINARY, 35, 15)
    out = Image.fromarray(arr)
    fd, tmp = tempfile.mkstemp(suffix=".png")
    os.close(fd)
    out.save(tmp)
    return tmp

def ocr_text_from_source(source: str) -> str:
    try:
        prep = _preprocess_for_ocr(source)
        result, _ = _rapidocr(prep)
        try: os.remove(prep)
        except: pass
        if not result:
            return ""
        return "\n".join([r[1] for r in result])
    except Exception:
        return ""

def ocr_text_from_upload(upload_bytes: bytes, filename: str) -> str:
    suffix = os.path.splitext(filename)[1] or ".bin"
    with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as f:
        f.write(upload_bytes)
        tmp = f.name
    try:
        return ocr_text_from_source(tmp)
    finally:
        try: os.remove(tmp)
        except: pass
