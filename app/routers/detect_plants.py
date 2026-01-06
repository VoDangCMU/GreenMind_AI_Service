
from fastapi import APIRouter, UploadFile, File
from fastapi.responses import JSONResponse
from ultralytics import YOLO
import cv2
import numpy as np
import math
import os
import tempfile
import base64

router = APIRouter()

model_veg = YOLO("app/models/best.pt")  
model_dish = YOLO("yolov8n.pt")          

@router.post("/analyze-image-plant")
async def analyze_image(file: UploadFile = File(...)):
    """
    Nhận 1 ảnh, trả về:
      - diện tích vùng thực vật
      - diện tích vùng món ăn
      - tỷ lệ %
      - ảnh kết quả (base64 cho 2 ảnh)
    """
    with tempfile.NamedTemporaryFile(delete=False, suffix=".jpg") as tmp:
        tmp.write(await file.read())
        tmp_path = tmp.name

    img = cv2.imread(tmp_path)
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    h_img, w_img = img_rgb.shape[:2]

   
    results_veg = model_veg.predict(source=tmp_path, conf=0.25, verbose=False)
    boxes_veg = results_veg[0].boxes.xyxy.cpu().numpy() if len(results_veg) > 0 else []
    mask_veg = np.zeros(img_rgb.shape[:2], dtype=np.uint8)
    img_veg_vis = img_rgb.copy()

    for box in boxes_veg:
        x1, y1, x2, y2 = map(int, box[:4])
        cv2.rectangle(img_veg_vis, (x1, y1), (x2, y2), (0, 255, 0), 2)
        mask_veg[y1:y2, x1:x2] = 1

    veg_area = int(np.sum(mask_veg > 0))

  
    results_dish = model_dish.predict(source=tmp_path, conf=0.25, verbose=False)
    boxes_dish = results_dish[0].boxes.xyxy.cpu().numpy() if len(results_dish) > 0 else []
    classes_dish = results_dish[0].boxes.cls.cpu().numpy().astype(int) if len(results_dish) > 0 else []
    names_dish = model_dish.names

    allowed_classes = ["bowl", "plate", "cup"]
    dish_area = 0.0
    img_dish_vis = img_rgb.copy()

    for box, cls_id in zip(boxes_dish, classes_dish):
        cls_name = names_dish[cls_id]
        if cls_name in allowed_classes:
            x1, y1, x2, y2 = map(int, box[:4])
            w, h = x2 - x1, y2 - y1
            r = min(w, h) / 2
            dish_area += math.pi * (r ** 2)
            cx, cy = (x1 + x2) // 2, (y1 + y2) // 2
            cv2.rectangle(img_dish_vis, (x1, y1), (x2, y2), (255, 0, 0), 2)
            cv2.circle(img_dish_vis, (cx, cy), int(r), (0, 0, 255), 2)
            cv2.putText(img_dish_vis, cls_name, (x1, y1 - 5),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 0, 0), 2)

    if dish_area == 0:
        print("Không detect được dĩa/chén — dùng toàn bộ ảnh để ước lượng món ăn.")
        cx, cy = w_img // 2, h_img // 2
        r = min(w_img, h_img) / 2
        dish_area = math.pi * (r ** 2)
        cv2.circle(img_dish_vis, (cx, cy), int(r), (255, 165, 0), 3)
        cv2.putText(img_dish_vis, "Fallback: whole image", (30, 40),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 165, 0), 2)

   
    ratio = (veg_area / dish_area * 100) if dish_area > 0 else 0


    _, plant_buf = cv2.imencode(".jpg", cv2.cvtColor(img_veg_vis, cv2.COLOR_RGB2BGR))
    _, dish_buf = cv2.imencode(".jpg", cv2.cvtColor(img_dish_vis, cv2.COLOR_RGB2BGR))

    plant_base64 = base64.b64encode(plant_buf).decode("utf-8")
    dish_base64 = base64.b64encode(dish_buf).decode("utf-8")

    os.remove(tmp_path)

 
    return JSONResponse({
        "vegetable_area": veg_area,
        "dish_area": round(dish_area, 2),
        "vegetable_ratio_percent": round(ratio, 2),
        "plant_image_base64": f"data:image/jpeg;base64,{plant_base64}",
        "dish_image_base64": f"data:image/jpeg;base64,{dish_base64}"
    })





