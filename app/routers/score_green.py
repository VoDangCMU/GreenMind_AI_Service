from fastapi import APIRouter
from pydantic import BaseModel
from typing import List

router = APIRouter()

# =========================
# CONFIG
# =========================
BASE_SCORE = 50

# =========================
# WEIGHTS
# =========================
CLASS_WEIGHTS = {
    "aluminium foil": 3, "battery": -8, "aluminium blister pack": 1,
    "carded blister pack": -2, "other plastic bottle": 1,
    "clear plastic bottle": 2, "glass bottle": 4,
    "plastic bottle cap": -1, "metal bottle cap": 2,
    "broken glass": -2, "food can": 4, "aerosol": -6, "drink can": 4,

    "toilet tube": 2, "other carton": 2, "egg carton": 3,
    "drink carton": 2, "corrugated carton": 3,
    "meal carton": 2, "pizza box": 1, "paper cup": 1,
    "disposable plastic cup": -3, "foam cup": -4,
    "glass cup": 3, "other plastic cup": -2,

    "food waste": -2, "glass jar": 4, "plastic lid": -1,
    "metal lid": 2, "other plastic": -2,

    "magazine paper": 3, "tissues": -2, "wrapping paper": 1,
    "normal paper": 3, "paper bag": 2,
    "plastified paper bag": -2,

    "plastic film": -3, "six pack rings": -4,
    "garbage bag": -3, "other plastic wrapper": -3,
    "single-use carrier bag": -4, "polypropylene bag": -3,
    "crisp packet": -4,

    "spread tub": 1, "tupperware": 2,
    "disposable food container": -3,
    "foam food container": -4,
    "other plastic container": -2,

    "plastic glooves": -3, "plastic utensils": -3,
    "pop tab": 2, "rope & strings": -1,
    "scrap metal": 3, "shoe": -2,
    "squeezable tube": -3, "plastic straw": -4,
    "paper straw": 1, "styrofoam piece": -5,

    "unlabeled litter": -3, "cigarette": -5
}

# =========================
# GROUPS
# =========================
RECYCLABLE = {
    "glass bottle", "glass jar", "drink can", "food can",
    "metal lid", "metal bottle cap", "scrap metal",
    "normal paper", "magazine paper", "corrugated carton"
}

PLASTIC = {
    "plastic film", "other plastic wrapper", "single-use carrier bag",
    "polypropylene bag", "crisp packet", "plastic straw",
    "styrofoam piece", "foam food container", "foam cup"
}

HAZARD = {"battery", "aerosol"}
ORGANIC = {"food waste"}

# =========================
# SCHEMA
# =========================
class Item(BaseModel):
    name: str
    quantity: int

class ScoreRequest(BaseModel):
    current_score: float = BASE_SCORE
    items: List[Item]

# =========================
# UTILS
# =========================
def normalize_name(name: str):
    return name.strip().lower()

# =========================
# CALCULATE
# =========================
def calculate_delta(items: List[Item]):
    total_score = 0
    total_count = 0

    plastic_count = 0
    has_food = False
    has_recycle = False
    has_hazard = False

    reasons = []

    # ===== LOOP =====
    for item in items:
        name = normalize_name(item.name)
        quantity = item.quantity
        weight = CLASS_WEIGHTS.get(name, 0)

        score_change = weight * quantity
        total_score += score_change
        total_count += quantity

        # reason từng item
        if weight > 0:
            reasons.append(f"{quantity} x {name} → +{score_change} điểm (tốt)")
        elif weight < 0:
            reasons.append(f"{quantity} x {name} → {score_change} điểm (gây hại)")

        # group tracking
        if name in PLASTIC:
            plastic_count += quantity
        if name in ORGANIC:
            has_food = True
        if name in RECYCLABLE:
            has_recycle = True
        if name in HAZARD:
            has_hazard = True

    if total_count == 0:
        return 0, []

    # =========================
    # RULES
    # =========================

    # Rule 0: có nhựa là bị trừ
    if plastic_count > 0:
        penalty = plastic_count * 1
        total_score -= penalty
        reasons.append(f"Có rác nhựa ({plastic_count} món) → -{penalty} điểm")

    # 1. nhiều nhựa (>50%)
    if plastic_count / total_count > 0.5:
        total_score -= 5
        reasons.append("Nhiều nhựa dùng một lần → -5 điểm")

    # 2. food + recyclable
    if has_food and has_recycle:
        total_score -= 4
        reasons.append("Rác hữu cơ làm bẩn rác tái chế → -4 điểm")

    # 3. hazardous
    if has_hazard:
        total_score -= 6
        reasons.append("Có rác nguy hại (pin/aerosol) → -6 điểm")

    # 4. phân loại tốt
    if has_recycle and plastic_count == 0 and not has_food:
        total_score += 5
        reasons.append("Phân loại tốt, nhiều rác tái chế → +5 điểm")

    return total_score, reasons


@router.post("/score")
async def get_score(data: ScoreRequest):
    delta, reasons = calculate_delta(data.items)

    final_score = data.current_score + delta
    final_score = max(0, min(100, final_score))

    return {
        "previous_score": data.current_score,
        "delta": delta,
        "final_score": final_score,
        "reasons": reasons
    }


@router.get("/home")
async def home():
    return {"message": "ok"}