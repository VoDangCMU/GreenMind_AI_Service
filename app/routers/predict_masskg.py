from fastapi import APIRouter, UploadFile, File, HTTPException
import os
import uuid
import tempfile
import numpy as np
import cv2
import torch
from ultralytics import YOLO
import matplotlib.pyplot as plt
from collections import defaultdict
import cloudinary
import cloudinary.uploader
from dotenv import load_dotenv

router = APIRouter()


load_dotenv()
cloudinary.config(
    cloud_name=os.getenv("CLOUDINARY_CLOUD_NAME"),
    api_key=os.getenv("CLOUDINARY_API_KEY"),
    api_secret=os.getenv("CLOUDINARY_API_SECRET")
)


YOLO_MODEL_PATH = os.path.join("app", "models", "taco_yolov8n_seg_best_po.pt")
yolo_model = YOLO(YOLO_MODEL_PATH)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

midas = torch.hub.load("intel-isl/MiDaS", "MiDaS_small", trust_repo=True)
midas.to(device)
midas.eval()
midas_transforms = torch.hub.load("intel-isl/MiDaS", "transforms", trust_repo=True)
transform = midas_transforms.small_transform

# Density function
def get_density(name: str):
    name = name.lower()
    if "plastic" in name:
        return 950
    elif "metal" in name or "can" in name:
        return 2700
    elif "paper" in name:
        return 800
    elif "glass" in name:
        return 2500
    else:
        return 1000

# Compute volume
def compute_volume(mask, depth_map, PIXEL_AREA, ground_depth, DEPTH_SCALE):
    ys, xs = np.where(mask > 0)
    if len(xs) == 0:
        return 0.0
    height = depth_map[ys, xs] - ground_depth
    height = np.clip(height, 1e-3, None)
    return float(np.sum(height * PIXEL_AREA * DEPTH_SCALE))


@router.get("/home")
async def home():
    return {"message": "Trash mass detection service ready 🚀"}


@router.post("/total-mass")
async def total_mass_api(file: UploadFile = File(...)):
    # Lưu tạm file upload
    tmp_path = os.path.join(tempfile.gettempdir(), f"{uuid.uuid4().hex}_{file.filename}")
    contents = await file.read()
    with open(tmp_path, "wb") as f:
        f.write(contents)

    # Đọc ảnh giống code local
    img = cv2.imread(tmp_path)
    if img is None:
        raise HTTPException(status_code=400, detail="Cannot read image")
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    H, W = img_rgb.shape[:2]

    # Pixel area + DEPTH_SCALE giống local
    REAL_WIDTH_M = 1.5
    REAL_HEIGHT_M = 1.5 * (H / W)
    REAL_AREA = REAL_WIDTH_M * REAL_HEIGHT_M
    PIXEL_AREA = REAL_AREA / (H * W)
    DEPTH_SCALE = 0.25  # giống code local

    # Depth estimation
    input_batch = transform(img_rgb).to(device)
    with torch.no_grad():
        depth = midas(input_batch)
        depth = torch.nn.functional.interpolate(
            depth.unsqueeze(1),
            size=(H, W),
            mode="bicubic",
            align_corners=False
        ).squeeze()
    depth_map = depth.cpu().numpy()
    depth_map = (depth_map - depth_map.min()) / (depth_map.max() - depth_map.min() + 1e-8)

    # YOLO segmentation từ file path
    results = yolo_model(tmp_path, conf=0.06)
    r = results[0]

    if r.masks is None:
        return {"items": [], "total_mass_kg": 0.0}

    masks = r.masks.data.cpu().numpy()
    classes = r.boxes.cls.cpu().numpy().astype(int)
    names = r.names

    # Resize masks giống local
    masks_resized = [(cv2.resize(mask, (W, H)) > 0.5).astype(np.uint8) for mask in masks]
    masks = np.array(masks_resized)

    # Fix depth direction giống code local
    combined_mask = np.sum(masks, axis=0) > 0
    bg_mean = np.mean(depth_map[~combined_mask])
    obj_mean = np.mean(depth_map[combined_mask])
    if bg_mean > obj_mean:
        depth_map = 1 - depth_map

    # Ground estimation giống local
    background_depth = depth_map[~combined_mask]
    ground_depth = np.percentile(background_depth, 90)

    # Tính mass
    total_mass = 0.0
    item_data = defaultdict(float)

    for i, mask in enumerate(masks):
        if mask.sum() < 500:
            continue
        cls_id = int(classes[i])
        name = names[cls_id]
        volume = compute_volume(mask, depth_map, PIXEL_AREA, ground_depth, DEPTH_SCALE)
        mass = volume * get_density(name)
        total_mass += mass
        item_data[name] += mass

    # Annotated image upload
    annotated_img = r.plot(conf=False)
    _, buffer = cv2.imencode(".jpg", annotated_img)
    upload_result = cloudinary.uploader.upload(
        buffer.tobytes(),
        public_id=f"mass_detect/{uuid.uuid4().hex}",
        folder="yolo_mass_detect",
        resource_type="image"
    )

    # Depth map image upload
    plt.figure(figsize=(6,6))
    plt.axis('off')
    plt.imshow(depth_map, cmap='plasma')
    plt.tight_layout()
    depth_img_path = os.path.join(tempfile.gettempdir(), f"depth_{uuid.uuid4().hex}.png")
    plt.savefig(depth_img_path, bbox_inches='tight', pad_inches=0)
    plt.close()

    with open(depth_img_path, "rb") as f:
        depth_upload = cloudinary.uploader.upload(
            f,
            public_id=f"depth_map/{uuid.uuid4().hex}",
            folder="yolo_depth_maps",
            resource_type="image"
        )

    # =========================
    # Return JSON
    # =========================
    items = [{"name": name, "mass_kg": round(float(mass), 4)} for name, mass in item_data.items()]

    # Xóa tạm file
    os.remove(tmp_path)
    os.remove(depth_img_path)

    return {
        "items": items,
        "total_mass_kg": round(float(total_mass), 4),
        "annotated_image_url": upload_result["secure_url"],
        "depth_map_url": depth_upload["secure_url"]
    }