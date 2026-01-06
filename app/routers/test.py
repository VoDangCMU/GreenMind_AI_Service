from ultralytics import YOLO
import cv2
import numpy as np
import math
import os

def test_ratio():
    model_veg_path = os.path.join("app", "models", "best.pt")  # model thực vật
    model_dish_path = "yolov8n.pt"                             # model COCO mặc định

    print("📂 Model veg path:", model_veg_path)
    print("✅ Model veg tồn tại:", os.path.exists(model_veg_path))

  
    try:
        model_veg = YOLO(model_veg_path)
        model_dish = YOLO(model_dish_path)
        print("Load cả 2 model thành công!")
    except Exception as e:
        print("❌ Lỗi khi load model:", e)
        return

    img_path = "uploads/test.jpg"
    if not os.path.exists(img_path):
        print("Không tìm thấy ảnh test:", img_path)
        return

    img = cv2.imread(img_path)
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)


    results_veg = model_veg.predict(source=img_path, conf=0.25)
    boxes_veg = results_veg[0].boxes.xyxy.cpu().numpy()

    mask_veg = np.zeros(img_rgb.shape[:2], dtype=np.uint8)
    for box in boxes_veg:
        x1, y1, x2, y2 = map(int, box[:4])
        mask_veg[y1:y2, x1:x2] = 1

    veg_area = np.sum(mask_veg > 0)
    print(f"🌿 Số vùng thực vật detect: {len(boxes_veg)}")
    print(f"🌿 Tổng diện tích thực vật (pixel): {veg_area}")

 
    results_dish = model_dish.predict(source=img_path, conf=0.25)
    boxes_dish = results_dish[0].boxes.xyxy.cpu().numpy()
    classes_dish = results_dish[0].boxes.cls.cpu().numpy().astype(int)
    names_dish = model_dish.names

    allowed_classes = ["bowl", "plate", "cup"]
    dish_area = 0

    for box, cls_id in zip(boxes_dish, classes_dish):
        cls_name = names_dish[cls_id]
        if cls_name in allowed_classes:
            x1, y1, x2, y2 = map(int, box[:4])
            w, h = x2 - x1, y2 - y1
            r = min(w, h) / 2
            dish_area += math.pi * (r ** 2)

    print(f"🍽️ Số món ăn detect: {sum(names_dish[c] in allowed_classes for c in classes_dish)}")
    print(f"🍽️ Tổng diện tích món ăn (ước lượng hình tròn): {dish_area:.2f}")

    # ===============================
    # 🔹 Tính tỷ lệ rau củ
    # ===============================
    ratio = veg_area / dish_area * 100 if dish_area > 0 else 0
    print(f"📊 Tỷ lệ rau củ / tổng món ăn: {ratio:.2f}%")

if __name__ == "__main__":
    test_ratio()
