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
# Load model
# =========================
MODEL_PATH = os.path.join("app", "models", "taco_yolov8n_seg_best_po.pt")
model = YOLO(MODEL_PATH)

# =========================
# CATEGORY MAP
# =========================
CATEGORY_MAP = {
    "Aluminium foil": "recyclable",
    "Aluminium blister pack": "recyclable",
    "Carded blister pack": "recyclable",
    "Other plastic bottle": "recyclable",
    "Clear plastic bottle": "recyclable",
    "Glass bottle": "recyclable",
    "Plastic bottle cap": "recyclable",
    "Metal bottle cap": "recyclable",
    "Broken glass": "recyclable",
    "Food Can": "recyclable",
    "Drink can": "recyclable",
    "Glass cup": "recyclable",
    "Glass jar": "recyclable",
    "Metal lid": "recyclable",
    "Scrap metal": "recyclable",
    "Pop tab": "recyclable",

    "Toilet tube": "recyclable",
    "Other carton": "recyclable",
    "Egg carton": "recyclable",
    "Drink carton": "recyclable",
    "Corrugated carton": "recyclable",
    "Meal carton": "recyclable",
    "Pizza box": "recyclable",
    "Paper cup": "recyclable",
    "Magazine paper": "recyclable",
    "Normal paper": "recyclable",
    "Paper bag": "recyclable",

    "Food waste": "organic",

    "Battery": "hazardous",
    "Cigarette": "hazardous",
    "Aerosol": "hazardous",

    "Disposable plastic cup": "residual",
    "Foam cup": "residual",
    "Other plastic cup": "residual",
    "Plastic lid": "residual",
    "Other plastic": "residual",
    "Plastic film": "residual",
    "Six pack rings": "residual",
    "Garbage bag": "residual",
    "Other plastic wrapper": "residual",
    "Single-use carrier bag": "residual",
    "Polypropylene bag": "residual",
    "Crisp packet": "residual",
    "Spread tub": "residual",
    "Tupperware": "residual",
    "Disposable food container": "residual",
    "Foam food container": "residual",
    "Other plastic container": "residual",
    "Plastic glooves": "residual",
    "Plastic utensils": "residual",
    "Plastic straw": "residual",
    "Paper straw": "residual",
    "Styrofoam piece": "residual",
    "Rope & strings": "residual",
    "Shoe": "residual",
    "Squeezable tube": "residual",
    "Unlabeled litter": "residual",

    "Tissues": "residual",
    "Wrapping paper": "residual",
    "Plastified paper bag": "residual"
}

# =========================
# CATEGORY COLOR
# =========================
CATEGORY_COLOR = {
"recyclable": (0, 255, 0),
    "organic": (0, 165, 255),
    "hazardous": (0, 0, 255),
    "residual": (128, 128, 128),
    "unknown": (255, 255, 255)
}

# =========================
# API
# =========================
@router.post("/detect-trash-ver2")
async def detect_image(file: UploadFile = File(...)):

    # 1. Read image
    contents = await file.read()
    np_arr = np.frombuffer(contents, np.uint8)
    img = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)

    # 2. YOLO detect
    results = model(img, conf=0.06)
    result = results[0]
    names = model.names

    item_data = defaultdict(lambda: {"quantity": 0, "area": 0})
    segments_output = []

    # 3. Process segmentation
    if result.masks is not None:
        masks = result.masks.data.cpu().numpy()
        class_ids = result.boxes.cls.cpu().numpy().astype(int)

        for i, class_id in enumerate(class_ids):
            mask = masks[i]
            name = names[class_id]

            # Area
            area = int(mask.sum())
            item_data[name]["quantity"] += 1
            item_data[name]["area"] += area

            # Mask
            binary_mask = (mask > 0.5).astype(np.uint8)
            binary_mask = cv2.resize(binary_mask, (img.shape[1], img.shape[0]))

            # Bounding box
            x, y, w, h = cv2.boundingRect(binary_mask)

            # Crop ảnh gốc
            cropped = img[y:y+h, x:x+w]
            cropped_mask = binary_mask[y:y+h, x:x+w]

            # Category + Color
            category = CATEGORY_MAP.get(name, "unknown")
            color = CATEGORY_COLOR.get(category, (255, 255, 255))

            # Overlay mask
            overlay = cropped.copy()
            overlay[cropped_mask == 1] = color

            alpha = 0.4
            cropped = cv2.addWeighted(overlay, alpha, cropped, 1 - alpha, 0)

            # Encode
            _, buffer = cv2.imencode(".png", cropped)

            # Upload
            upload_seg = cloudinary.uploader.upload(
                buffer.tobytes(),
                public_id=f"segments/{uuid.uuid4().hex}",
                folder="yolo_segments",
                resource_type="image"
            )

            # Append
            segments_output.append({
                "class_name": name,
                "category": category,
                "image_url": upload_seg["secure_url"]
            })

    # 4. Items
    items = [
        {
            "name": name,
            "quantity": val["quantity"],
            "area": val["area"]
        }
        for name, val in item_data.items()
    ]

    # 5. Annotated image
    annotated_img = result.plot()
    _, buffer = cv2.imencode(".jpg", annotated_img)

    upload_result = cloudinary.uploader.upload(
        buffer.tobytes(),
        public_id=f"detect/{uuid.uuid4().hex}",
        folder="yolo_detect",
        resource_type="image"
    )

    # 6. Group
    grouped = defaultdict(list)
    for seg in segments_output:
        # grouped[seg["category"]].append(seg)
        grouped[seg["category"]].append(seg["image_url"])


    # 7. Return
    return {
            "total_objects": len(result.boxes) if result.boxes else 0,
            "image_url": upload_result["secure_url"],
            # "items": items,
            # "segments": segments_output,
            "grouped": dict(grouped)
        }