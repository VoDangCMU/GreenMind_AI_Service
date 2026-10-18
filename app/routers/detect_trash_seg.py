from fastapi import APIRouter, UploadFile, File, HTTPException
from ultralytics import YOLO
import numpy as np
import cv2
import os
import uuid
import cloudinary
import cloudinary.uploader
from dotenv import load_dotenv
import torch

router = APIRouter()

load_dotenv()

# =========================
# Cloudinary config
# =========================
cloudinary.config(
    cloud_name=os.getenv("CLOUDINARY_CLOUD_NAME"),
    api_key=os.getenv("CLOUDINARY_API_KEY"),
    api_secret=os.getenv("CLOUDINARY_API_SECRET")
)

# =========================
# YOLO model
# =========================
MODEL_PATH = os.path.join("app", "models", "best_landfill_yolov8_seg.pt")
model = YOLO(MODEL_PATH)

TARGET_CLASS_ID = 0

# =========================
# MiDaS model
# =========================
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

midas = torch.hub.load("intel-isl/MiDaS", "MiDaS_small", trust_repo=True)
midas.to(device)
midas.eval()

midas_transforms = torch.hub.load("intel-isl/MiDaS", "transforms", trust_repo=True)
transform = midas_transforms.small_transform


@router.get("/home")
async def home():
    return {"message": "segment + depth service ready"}


@router.post("/predict_trash_seg")
async def segment_trash_image(file: UploadFile = File(...)):

    # =========================
    # Validate image
    # =========================
    if not file.content_type or not file.content_type.startswith("image/"):
        raise HTTPException(status_code=400, detail="File phải là ảnh")

    contents = await file.read()
    if not contents:
        raise HTTPException(status_code=400, detail="File rỗng")

    np_arr = np.frombuffer(contents, np.uint8)
    img = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)

    if img is None:
        raise HTTPException(status_code=400, detail="Không đọc được ảnh")

    h, w = img.shape[:2]
    image_area = h * w

    # =========================
    # YOLO SEGMENT
    # =========================
    try:
        results = model.predict(source=img, conf=0.25, imgsz=640, verbose=False)
        result = results[0]
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Lỗi YOLO: {str(e)}")

    union_mask = np.zeros((h, w), dtype=np.uint8)

    if result.masks is not None and result.boxes is not None:
        class_ids = result.boxes.cls.cpu().numpy().astype(int)

        for i, polygon in enumerate(result.masks.xy):
            if class_ids[i] == TARGET_CLASS_ID:
                pts = np.array(polygon, dtype=np.int32)
                if len(pts) >= 3:
                    cv2.fillPoly(union_mask, [pts], 1)

    landfill_pixel_area = int(union_mask.sum())
    segment_ratio = landfill_pixel_area / image_area

    annotated_img = result.plot()

    # =========================
    # DEPTH (MiDaS)
    # =========================
    try:
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        input_batch = transform(img_rgb).to(device)

        with torch.no_grad():
            prediction = midas(input_batch)
            prediction = torch.nn.functional.interpolate(
                prediction.unsqueeze(1),
                size=(h, w),
                mode="bicubic",
                align_corners=False,
            ).squeeze()

        depth_map = prediction.cpu().numpy()

        # normalize
        depth_min = depth_map.min()
        depth_max = depth_map.max()
        depth_norm = (depth_map - depth_min) / (depth_max - depth_min + 1e-8)

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Lỗi MiDaS: {str(e)}")

    # =========================
    # ESTIMATE HEIGHT
    # =========================
    try:
        non_trash = depth_norm[union_mask == 0]

        if len(non_trash) > 0:
            ground_depth = np.percentile(non_trash, 95)  # nâng lên 95%
        else:
            ground_depth = depth_norm.max()

        relative_height = np.clip(ground_depth - depth_norm, 0, None)
        trash_height = relative_height * union_mask

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Lỗi height: {str(e)}")

    # =========================
    # POLLUTION SCORING (NEW 🔥)
    # =========================
    try:
        if landfill_pixel_area > 0:

            # --- Global height score ---
            height_score = trash_height.sum() / (landfill_pixel_area + 1e-8)

            # --- Grid-based local score ---
            grid_size = 8
            cell_h = h // grid_size
            cell_w = w // grid_size

            local_scores = []

            for i in range(grid_size):
                for j in range(grid_size):
                    y1, y2 = i * cell_h, (i + 1) * cell_h
                    x1, x2 = j * cell_w, (j + 1) * cell_w

                    cell_mask = union_mask[y1:y2, x1:x2]
                    cell_height = trash_height[y1:y2, x1:x2]

                    area = cell_mask.sum()

                    if area > 0:
                        local_scores.append(cell_height.sum() / area)

            if local_scores:
                mean_local = np.mean(local_scores)
                max_local = np.max(local_scores)
                local_score = 0.6 * mean_local + 0.4 * max_local
            else:
                local_score = 0.0

            # --- Combine ---
            pollution_score = segment_ratio * (0.5 * height_score + 0.5 * local_score)

        else:
            pollution_score = 0.0

        # scale 0-1
        pollution_score = float(np.clip(pollution_score * 5, 0, 1))

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Lỗi pollution: {str(e)}")

    # =========================
    # HEATMAP (BONUS 🔥)
    # =========================
    heatmap = trash_height.copy()
    heatmap = (heatmap / (heatmap.max() + 1e-8) * 255).astype(np.uint8)
    heatmap = cv2.applyColorMap(heatmap, cv2.COLORMAP_JET)

    # =========================
    # DEPTH VISUALIZE
    # =========================
    depth_img = (depth_norm * 255).astype(np.uint8)
    depth_colored = cv2.applyColorMap(depth_img, cv2.COLORMAP_INFERNO)

    # =========================
    # Upload images
    # =========================
    def upload_img(image, folder):
        ok, buffer = cv2.imencode(".jpg", image)
        if not ok:
            raise HTTPException(status_code=500, detail=f"Encode {folder} fail")

        return cloudinary.uploader.upload(
            buffer.tobytes(),
            public_id=f"{folder}/{uuid.uuid4().hex}",
            folder=folder
        )["secure_url"]

    seg_url = upload_img(annotated_img, "segment_landfill")
    depth_url = upload_img(depth_colored, "depth_map")
    heatmap_url = upload_img(heatmap, "pollution_heatmap")

    # =========================
    # CLASSIFICATION (UI 🔥)
    # =========================
    if pollution_score < 0.2:
        level = "Clean"
    elif pollution_score < 0.5:
        level = "Moderate"
    elif pollution_score < 0.8:
        level = "Heavy"
    else:
        level = "Critical"

    return {
        "segment_ratio": round(segment_ratio, 4),
        "pollution_score": round(pollution_score, 4),
        "pollution_level": level,
        "segment_image_url": seg_url,
        "depth_image_url": depth_url,
        "heatmap_url": heatmap_url
    }