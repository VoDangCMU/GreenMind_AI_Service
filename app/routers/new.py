from fastapi import APIRouter, UploadFile, File, Form, HTTPException
from typing import Optional
from collections import defaultdict
import math
import json
import re

# ===== OCR =====
from app.services.ocr_text import ocr_text_from_source, ocr_text_from_upload
from app.services.orc_parse_gemini import parse_invoice_text

# ===== LLM =====
from app.models.open_ai_llm import get_openai_llm
router = APIRouter()

# =========================
# CLASS NAMES (🔥 NEW)
# =========================
CLASS_NAMES = {
    0:"Aluminium foil",1:"Battery",2:"Aluminium blister pack",3:"Carded blister pack",
    4:"Other plastic bottle",5:"Clear plastic bottle",6:"Glass bottle",7:"Plastic bottle cap",
    8:"Metal bottle cap",9:"Broken glass",10:"Food Can",11:"Aerosol",12:"Drink can",
    13:"Toilet tube",14:"Other carton",15:"Egg carton",16:"Drink carton",17:"Corrugated carton",
    18:"Meal carton",19:"Pizza box",20:"Paper cup",21:"Disposable plastic cup",22:"Foam cup",
    23:"Glass cup",24:"Other plastic cup",25:"Food waste",26:"Glass jar",27:"Plastic lid",
    28:"Metal lid",29:"Other plastic",30:"Magazine paper",31:"Tissues",32:"Wrapping paper",
    33:"Normal paper",34:"Paper bag",35:"Plastified paper bag",36:"Plastic film",
    37:"Six pack rings",38:"Garbage bag",39:"Other plastic wrapper",40:"Single-use carrier bag",
    41:"Polypropylene bag",42:"Crisp packet",43:"Spread tub",44:"Tupperware",
    45:"Disposable food container",46:"Foam food container",47:"Other plastic container",
    48:"Plastic glooves",49:"Plastic utensils",50:"Pop tab",51:"Rope & strings",
    52:"Scrap metal",53:"Shoe",54:"Squeezable tube",55:"Plastic straw",
    56:"Paper straw",57:"Styrofoam piece",58:"Unlabeled litter",59:"Cigarette"
}

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
    "CO2","CH4","PM2.5","NOx","SO2",
    "Pb","Hg","Cd","nitrate","chemical_residue",
    "microplastic","dioxin","toxic_chemicals",
    "non_biodegradable","styrene"
]

# =========================
# PROMPTS
# =========================
PROMPT_PARSE = """
Bạn là hệ thống OCR parser.

Nhiệm vụ:
- Trích xuất danh sách món từ OCR
- Sửa lỗi chính tả (Spnte → Sprite, Touc → Tonic)

QUY TẮC BẮT BUỘC:
- Chỉ trả về JSON hợp lệ (valid JSON)
- Không markdown
- Không text ngoài JSON
- Không giải thích
- Không xuống dòng linh tinh
- Không được cắt JSON giữa chừng

Format bắt buộc:

{{
  "items": [
    {{"raw_name": "coca", "quantity": 2}},
    {{"raw_name": "sprite", "quantity": 1}}
  ]
}}


OCR:
{ocr}
"""

PROMPT_MAP = """
Bạn là AI phân loại rác theo taxonomy TACO dataset.

Nhiệm vụ:
- Map item có thể là vỏ chứa hay liên quan → đúng class id
- Nếu không chắc → trả 58 (Unlabeled litter)
- Chỉ trả về JSON

Classes:
0 - Aluminium foil
1 - Battery
2 - Aluminium blister pack
3 - Carded blister pack
4 - Other plastic bottle
5 - Clear plastic bottle
6 - Glass bottle
7 - Plastic bottle cap
8 - Metal bottle cap
9 - Broken glass
10 - Food Can
11 - Aerosol
12 - Drink can
13 - Toilet tube
14 - Other carton
15 - Egg carton
16 - Drink carton
17 - Corrugated carton
18 - Meal carton
19 - Pizza box
20 - Paper cup
21 - Disposable plastic cup
22 - Foam cup
23 - Glass cup
24 - Other plastic cup
25 - Food waste
26 - Glass jar
27 - Plastic lid
28 - Metal lid
29 - Other plastic
30 - Magazine paper
31 - Tissues
32 - Wrapping paper
33 - Normal paper
34 - Paper bag
35 - Plastified paper bag
36 - Plastic film
37 - Six pack rings
38 - Garbage bag
39 - Other plastic wrapper
40 - Single-use carrier bag
41 - Polypropylene bag
42 - Crisp packet
43 - Spread tub
44 - Tupperware
45 - Disposable food container
46 - Foam food container
47 - Other plastic container
48 - Plastic gloves
49 - Plastic utensils
50 - Pop tab
51 - Rope & strings
52 - Scrap metal
53 - Shoe
54 - Squeezable tube
55 - Plastic straw
56 - Paper straw
57 - Styrofoam piece
58 - Unlabeled litter
59 - Cigarette

Trả JSON:
{{
  "item": "...",
  "class_id": number,
  "class_name": "...",
  "reason": "..."
}}

Item: {name}
"""

# =========================
# LLM
# =========================
def call_llm(prompt: str) -> str:
    llm = get_openai_llm()
    return llm.chat_raw(
        system_prompt="Return valid JSON only",
        user_prompt=prompt,
        max_output_tokens=1024
    ).strip()

def extract_json(raw: str):
    m = re.search(r"(\{.*\})", raw, re.DOTALL)
    return m.group(1) if m else "{}"

def safe_json_load(raw: str):
    try:
        return json.loads(raw)
    except Exception as e:
        print("[JSON ERROR 1]", e)

    # 🔥 extract JSON block
    raw = extract_json(raw)

    # 🔥 fix lỗi phổ biến
    raw = raw.replace("\n", " ")
    raw = re.sub(r",\s*}", "}", raw)   # remove trailing comma
    raw = re.sub(r",\s*]", "]", raw)

    # 🔥 fix thiếu dấu "
    raw = re.sub(r'("raw_name":\s*"[^"]+)(?=\s*,|\s*})', r'\1"', raw)

    try:
        return json.loads(raw)
    except Exception as e:
        print("[JSON ERROR 2]", e)

    return {"items": []}
def parse_invoice(ocr_text: str):
    print("\n========== ocr text ==========")
    print(ocr_text)
    print("=============================\n")
    raw = call_llm(PROMPT_PARSE.format(ocr=ocr_text))
    print("\n========== RAW LLM ==========")
    print(raw)
    print("=============================\n")
    return safe_json_load(raw)

def llm_map_item(name: str) -> int:
    try:
        raw = call_llm(PROMPT_MAP.format(name=name))
        return int(re.findall(r"\d+", raw)[0])
    except:
        return 58

# =========================
# BUILD (🔥 UPDATED)
# =========================
def build_detections(items):
    detections = []
    enriched_items = []

    for item in items:
        name = item.get("raw_name", "")
        quantity = item.get("quantity", 1)

        class_id = llm_map_item(name)

        # dùng cho pollution
        if class_id != 58:
            detections.append({
                "class": class_id,
                "quantity": quantity
            })

        # dùng cho output (🔥 quan trọng)
        enriched_items.append({
            "raw_name": name,
            "quantity": quantity,
            "class_id": class_id,
            "class_name": CLASS_NAMES.get(class_id, "Unknown")
        })

    return detections, enriched_items

# =========================
# POLLUTION
# =========================
def compute_pollution(detections):
    pollution = defaultdict(float)

    for obj in detections:
        cls = str(obj["class"])
        quantity = obj["quantity"]

        alpha = ALPHA.get(cls, 1.0)
        beta = BETA.get(cls, {})

        for p, w in beta.items():
            pollution[p] += quantity * alpha * w

    for p in ALL_POLLUTANTS:
        pollution[p] = pollution.get(p, 0.0)

    return dict(pollution)

def normalize(pollution):
    total = sum(pollution.values()) + 1e-6
    return {k: v/total for k,v in pollution.items()}

def classify(pollution):
    air = ["CO2","CH4","PM2.5","NOx","SO2"]
    water = ["microplastic","Pb","Hg","Cd"]
    soil = ["toxic_chemicals","non_biodegradable"]

    def score(keys):
        vals = [pollution[k] for k in keys if pollution[k] > 0]
        if not vals: return 0
        return sum(vals)/len(vals)

    return {
        "air": score(air),
        "water": score(water),
        "soil": score(soil)
    }

# =========================
# API
# =========================
@router.post("/invoice-pollution")
async def invoice_pollution(
    file: Optional[UploadFile] = File(None),
    url: Optional[str] = Form(None)
):
    if not file and not url:
        raise HTTPException(400, "Cần file hoặc url")

    # OCR
    if file:
        content = await file.read()
        text = ocr_text_from_upload(content, file.filename)
    else:
        text = ocr_text_from_source(url)

    if not text.strip():
        raise HTTPException(500, "OCR fail")
    print('text ocr', text)
    # Parse
    parsed = parse_invoice(text)
    items = parsed.get("items", []) 
    print('parsed', parsed)

    # Mapping
    detections, enriched_items = build_detections(items)

    if not detections:
        return {
            "items": enriched_items,
            "pollution": {k:0 for k in ALL_POLLUTANTS},
            "impact": {"air":0,"water":0,"soil":0}
        }

    # Compute
    pollution = compute_pollution(detections)
    pollution = normalize(pollution)
    impact = classify(pollution)

    return {
        "items": enriched_items,   # 🔥 đã map
        "pollution": pollution,
        "impact": impact
    }