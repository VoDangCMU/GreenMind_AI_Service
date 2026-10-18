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
import math
from fastapi.encoders import jsonable_encoder
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
# Load YOLO model
# =========================
MODEL_PATH = os.path.join("app", "models", "taco_yolov8n_seg_best_po.pt")
model = YOLO(MODEL_PATH)

# =========================
# ALPHA & BETA
# =========================
ALPHA = {
    "0":1.3,"1":2.0,"2":1.2,"3":1.0,"4":1.0,"5":1.0,"6":1.0,"7":0.8,
    "8":1.2,"9":1.0,"10":1.3,"11":1.2,"12":1.3,"13":0.6,"14":0.7,
    "15":0.6,"16":0.8,"17":0.7,"18":0.7,"19":0.8,"20":0.7,"21":0.9,
    "22":0.8,"23":1.0,"24":0.9,"25":1.2,"26":1.0,"27":0.9,"28":1.2,
    "29":1.0,"30":0.7,"31":0.6,"32":0.7,"33":0.7,"34":0.7,"35":0.8,
    "36":1.0,"37":1.0,"38":1.1,"39":1.0,"40":1.0,"41":1.0,"42":1.0,
    "43":1.0,"44":1.0,"45":1.1,"46":0.8,"47":1.0,"48":1.0,"49":1.0,
    "50":1.2,"51":0.9,"52":1.4,"53":1.3,"54":1.0,"55":0.9,"56":0.6,
    "57":0.8,"58":1.0,"59":1.5
}

BETA = {
    # (cop  # 🔋 Battery
  "1": {
    "Pb":1,"Hg":1,"Cd":1,
    "toxic_chemicals":1,
    "chemical_residue":0.9,
    "dioxin":0.3,
    "non_biodegradable":1
  },

  # 🍌 Food waste
  "25": {
    "CH4":1,"CO2":0.7,
    "nitrate":0.9,
    "chemical_residue":0.5,
    "PM2.5":0.2
  },

  # 🚬 Cigarette
  "59": {
    "PM2.5":1,"CO2":0.6,
    "Pb":0.6,"Cd":0.6,
    "toxic_chemicals":1,
    "dioxin":0.5,
    "microplastic":0.5
  },

  # 🧊 Foam / Styrofoam
  "22": {"styrene":1,"microplastic":1,"CO2":0.8,"dioxin":0.6,"toxic_chemicals":0.7,"non_biodegradable":1},
  "46": {"styrene":1,"microplastic":1,"CO2":0.8,"dioxin":0.6,"toxic_chemicals":0.7,"non_biodegradable":1},
  "57": {"styrene":1,"microplastic":1,"CO2":0.8,"dioxin":0.6,"toxic_chemicals":0.7,"non_biodegradable":1},

  # 🍾 Glass
  "6": {"non_biodegradable":1,"PM2.5":0.2},
  "9": {"non_biodegradable":1,"PM2.5":0.3},
  "23": {"non_biodegradable":1},
  "26": {"non_biodegradable":1},

  # 🥫 Metal
  "0": {"CO2":0.4,"NOx":0.3,"SO2":0.3,"Pb":0.3,"PM2.5":0.2},
  "2": {"CO2":0.4,"NOx":0.3,"SO2":0.3,"Pb":0.3},
  "8": {"CO2":0.4,"NOx":0.3,"SO2":0.3},
  "10": {"CO2":0.4,"NOx":0.3,"SO2":0.3},
  "11": {"CO2":0.4,"NOx":0.3,"SO2":0.3,"chemical_residue":0.6},
  "12": {"CO2":0.4,"NOx":0.3,"SO2":0.3},
  "28": {"CO2":0.4,"NOx":0.3,"SO2":0.3},
  "50": {"CO2":0.4,"NOx":0.3,"SO2":0.3},
  "52": {"CO2":0.5,"NOx":0.4,"SO2":0.4,"Pb":0.5},

  # 📦 Paper
  "13": {"CO2":0.6,"CH4":0.6,"nitrate":0.5,"chemical_residue":0.4},
  "14": {"CO2":0.6,"CH4":0.6,"nitrate":0.5},
  "15": {"CO2":0.6,"CH4":0.6},
  "16": {"CO2":0.6,"CH4":0.6,"microplastic":0.3},
  "17": {"CO2":0.6,"CH4":0.6},
  "18": {"CO2":0.6,"CH4":0.6},
  "19": {"CO2":0.6,"CH4":0.6,"chemical_residue":0.5},
  "20": {"CO2":0.6,"CH4":0.5},
  "30": {"CO2":0.6,"CH4":0.5},
  "31": {"CO2":0.6,"CH4":0.5},
  "32": {"CO2":0.6,"CH4":0.5},
  "33": {"CO2":0.6,"CH4":0.5},
  "34": {"CO2":0.6,"CH4":0.5},
  "35": {"CO2":0.6,"microplastic":0.7,"chemical_residue":0.5},
  "56": {"CO2":0.6,"CH4":0.5},

  # 👟 Rubber
  "53": {
    "microplastic":0.8,
    "toxic_chemicals":0.8,
    "non_biodegradable":1,
    "CO2":0.5
  },

  # ======================
  # 🧴 PLASTIC (THÊM MỚI)
  # ======================
  "3": {"CO2":1,"dioxin":0.8,"microplastic":1,"toxic_chemicals":0.8,"non_biodegradable":1},
  "4": {"CO2":1,"dioxin":0.8,"microplastic":1,"toxic_chemicals":0.8,"non_biodegradable":1},
  "5": {"CO2":1,"dioxin":0.8,"microplastic":1,"toxic_chemicals":0.8,"non_biodegradable":1},
  "7": {"CO2":1,"microplastic":0.9,"non_biodegradable":1},
  "21": {"CO2":1,"microplastic":1,"dioxin":0.7,"non_biodegradable":1},
  "24": {"CO2":1,"microplastic":1,"non_biodegradable":1},
  "27": {"CO2":1,"microplastic":1,"non_biodegradable":1},
  "29": {"CO2":1,"microplastic":1,"non_biodegradable":1},
  "36": {"CO2":1,"microplastic":1,"dioxin":0.8,"non_biodegradable":1},
  "37": {"CO2":1,"microplastic":1,"non_biodegradable":1},
  "38": {"CO2":1,"microplastic":1,"dioxin":0.9,"non_biodegradable":1},
  "39": {"CO2":1,"microplastic":1,"non_biodegradable":1},
  "40": {"CO2":1,"microplastic":1,"non_biodegradable":1},
  "41": {"CO2":1,"microplastic":1,"non_biodegradable":1},
  "42": {"CO2":1,"microplastic":1,"dioxin":0.9,"non_biodegradable":1},
  "43": {"CO2":1,"microplastic":1,"non_biodegradable":1},
  "44": {"CO2":1,"microplastic":1,"non_biodegradable":1},
  "45": {"CO2":1,"microplastic":1,"dioxin":0.7,"non_biodegradable":1},
  "47": {"CO2":1,"microplastic":1,"non_biodegradable":1},
  "48": {"CO2":1,"microplastic":1,"toxic_chemicals":0.6,"non_biodegradable":1},
  "49": {"CO2":1,"microplastic":1,"non_biodegradable":1},
  "54": {"CO2":1,"microplastic":1,"toxic_chemicals":0.6,"non_biodegradable":1},
  "55": {"CO2":1,"microplastic":1,"non_biodegradable":1},

  # 🧵 Rope / Fiber
  "51": {"microplastic":0.8,"non_biodegradable":1},

  # ❓ Unknown
  "58": {"CO2":0.8,"microplastic":0.8,"non_biodegradable":1}
}

ALL_POLLUTANTS = [
    "CO2", "CH4", "PM2.5", "NOx", "SO2",
    "Pb", "Hg", "Cd",
    "nitrate", "chemical_residue",
    "microplastic", "dioxin", "toxic_chemicals",
    "non_biodegradable",
    "styrene"
]

# =========================
# Pollution pipeline
# =========================
def compute_pollution(detections):
    pollution = defaultdict(float)
    for obj in detections:
        cls = str(obj["class"])
        area = obj["area"]
        alpha = ALPHA.get(cls, 1.0)
        beta = BETA.get(cls, {})
        for pollutant, weight in beta.items():
            pollution[pollutant] += area * alpha * weight
    for p in ALL_POLLUTANTS:
        if p not in pollution:
            pollution[p] = 0.0
    return dict(pollution)

def normalize_pollution(pollution):
    max_val = max(pollution.values()) + 1e-6
    return {k: v / max_val for k, v in pollution.items()}

def log_scale(pollution):
    return {k: math.log(1 + v) for k, v in pollution.items()}

def classify_environment(pollution):
    air_pollutants = ["CO2", "CH4", "NOx", "SO2", "PM2.5", "dioxin"]
    water_pollutants = ["microplastic", "Pb", "Hg", "Cd", "nitrate", "chemical_residue"]
    soil_pollutants = ["microplastic", "Pb", "Cd", "toxic_chemicals", "styrene", "non_biodegradable"]

    def score(keys):
        vals = [pollution.get(p, 0) for p in keys if pollution.get(p, 0) > 0]
        if not vals:
            return 0.0

        avg = sum(vals) / len(vals)
        coverage = len(vals) / len(keys)   # 🔥 quan trọng

        return avg * coverage

    return {
        "air_pollution": score(air_pollutants),
        "water_pollution": score(water_pollutants),
        "soil_pollution": score(soil_pollutants)
    }

def sigmoid_scale(pollution, k=0.00001, midpoint=50000):
    scaled = {}
    for key, v in pollution.items():
        if v == 0:
            scaled[key] = 0.0   # 🔥 giữ nguyên
        else:
            x = k * (v - midpoint)
            scaled[key] = 1 / (1 + math.exp(-x))
    return scaled

def predict_pollution(detections):
    pollution = compute_pollution(detections)
    pollution = normalize_pollution(pollution)
    pollution = sigmoid_scale(pollution, k=5, midpoint=0.3)
    pollution = log_scale(pollution)
    impact = classify_environment(pollution)
    return {"pollution": pollution, "impact": impact}

# =========================
# API predict pollutant
# =========================
@router.post("/predict-pollutant-impact")
async def predict_pollutant_impact(file: UploadFile = File(...)):
    # 1. đọc ảnh
    contents = await file.read()
    np_arr = np.frombuffer(contents, np.uint8)
    img = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)

    # 2. detect
    results = model(img, conf=0.06)
    result = results[0]
    names = model.names  # dict {id: name}
    # tạo dict đảo ngược name -> id
    name_to_id = {v: k for k, v in names.items()}

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
        {
            "name": name,
            "quantity": int(val["quantity"]),
            "area": int(val["area"])
        }
        for name, val in item_data.items()
    ]

    # 4. compute pollution
    detections_list = [{"class": int(name_to_id[name]), "area": val["area"]} for name, val in item_data.items()]
    pollution_result = predict_pollution(detections_list)

    # 5. encode ảnh
    annotated_img = result.plot()
    _, buffer = cv2.imencode(".jpg", annotated_img)
    upload_result = cloudinary.uploader.upload(
        buffer.tobytes(),
        public_id=f"detect/{uuid.uuid4().hex}",
        folder="yolo_detect",
        resource_type="image"
    )

    # 6. ensure native types
    pollution = {k: float(v) for k, v in pollution_result["pollution"].items()}
    impact = {k: float(v) for k, v in pollution_result["impact"].items()}

    # 7. return
    return jsonable_encoder({
        "items": items,
        "total_objects": len(result.boxes) if result.boxes else 0,
        "image_url": upload_result["secure_url"],
        "pollution": pollution,
        "impact": impact
    })