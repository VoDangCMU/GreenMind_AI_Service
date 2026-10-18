from fastapi import APIRouter, UploadFile, File
from ultralytics import YOLO
import numpy as np
import cv2
from collections import defaultdict
import os
import uuid
import cloudinary
import cloudinary.uploader
from dotenv import load_dotenv

# =========================
# Init router
# =========================
router = APIRouter()

# =========================
# Load ENV
# =========================
load_dotenv()

cloudinary.config(
    cloud_name=os.getenv("CLOUDINARY_CLOUD_NAME"),
    api_key=os.getenv("CLOUDINARY_API_KEY"),
    api_secret=os.getenv("CLOUDINARY_API_SECRET")
)

# =========================
# Load model (load 1 lần)
# =========================
MODEL_PATH = os.path.join("app", "models", "taco_yolov8n_seg_best_po.pt")
model = YOLO(MODEL_PATH)

# =========================
# API test
# =========================
@router.get("/home")
async def home():
    return {"message": "detect service ready 🚀"}

# =========================
# API detect
# =========================
@router.post("/detect-trash")
async def detect_image(file: UploadFile = File(...)):
    # 1. đọc ảnh
    contents = await file.read()
    np_arr = np.frombuffer(contents, np.uint8)
    img = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)

    # 2. detect
    results = model(img, conf=0.03)
    result = results[0]
    names = model.names

    # 3. xử lý output
    item_data = defaultdict(lambda: {"quantity": 0, "area": 0})

    if result.masks is not None:
        masks = result.masks.data.cpu().numpy()
        class_ids = result.boxes.cls.cpu().numpy().astype(int)

        for i, class_id in enumerate(class_ids):
            mask = masks[i]
            area = int(mask.sum())
            name = names[class_id]

            item_data[name]["quantity"] += 1
            item_data[name]["area"] += area

    items = [
        {"name": name, "quantity": val["quantity"], "area": val["area"]}
        for name, val in item_data.items()
    ]

    # 4. vẽ ảnh
    annotated_img = result.plot()

    # 5. encode
    _, buffer = cv2.imencode(".jpg", annotated_img)

    # 6. upload cloudinary
    upload_result = cloudinary.uploader.upload(
        buffer.tobytes(),
        public_id=f"detect/{uuid.uuid4().hex}",
        folder="yolo_detect",
        resource_type="image"
    )

    # 7. return
    return {
        "items": items,
        "total_objects": len(result.boxes) if result.boxes else 0,
        "image_url": upload_result["secure_url"]
    }