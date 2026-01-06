# monitor_ocean.py
from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, Field
from typing import List, Dict, Optional
import math
import statistics

router = APIRouter()

EPS = 1e-8

def sigmoid(x: float) -> float:
    return 1.0 / (1.0 + math.exp(-x))

def logit(x: float, e: float = EPS) -> float:
    x = min(max(x, 0.0), 1.0)
    return math.log((x + e) / (1 - x + e))

def clip01(x: float) -> float:
    return min(1.0, max(0.0, x))

def likert_to_ratio(l: int) -> float:
    l = int(l)
    l = min(max(l, 1), 5)
    return (l - 1) / 4.0 

def likert_to_cv(l: int) -> float:
    l = int(min(max(l, 1), 5))
    return 0.15 + 0.10 * (l - 1)

def compute_r_from_positive(vt: float, bt: float) -> float:
    return vt / max(bt, EPS) - 1.0

def compute_r_from_ratio(vt: float, bt: float) -> float:
    return logit(vt) - logit(bt)

def compute_r_from_variability(vt: float, bt: float) -> float:
    return vt / max(bt, EPS) - 1.0

def normalize_r_to_n(r: float, sigma_r: float, direction: str) -> float:
    if sigma_r <= 0:
        sigma_r = 1.0
    if direction == "up":
        return sigmoid(r / sigma_r)
    elif direction == "down":
        return sigmoid(-r / sigma_r)
    else:
        raise ValueError("direction must be 'up' or 'down'")

def compute_contrib(n: float, weight: float) -> float:
    return weight * (2.0 * n - 1.0)

def update_trait_score(ocean: Dict[str, float], trait: str, contrib: float, alpha: float) -> Dict[str, float]:
    new = ocean.copy()
    if trait in new:
        new[trait] = clip01(new[trait] + alpha * contrib)
    return new

def _significant_change_r(r: float, threshold: float = 0.15) -> bool:
    return abs(r) >= threshold

def generate_mechanism_feedback(
    ocean: Dict[str, float],
    target_trait: str,
    vt: float,
    bt: float,
    r: float,
    n: float,
    contrib: float,
    metric_name: str
) -> (Dict[str, str], str):
    """
    Returns mechanismFeedback dict {awareness, motivation, capability, opportunity}
    and a short reason string for this metric change.
    Feedback now includes Extraversion (E) explicitly.
    """
    feedback = {"awareness": "", "motivation": "", "capability": "", "opportunity": ""}
    trait_val = ocean.get(target_trait, 0.5)

    #awareness 
    if target_trait == "A":
        if trait_val >= 0.65:
            feedback["awareness"] = "Nhận thức cảm xúc / xã hội cao — người dùng dễ thấy tác động hành vi."
        elif trait_val <= 0.35:
            feedback["awareness"] = "Nhận thức cảm xúc thấp — có thể bỏ sót ảnh hưởng hành vi."
        else:
            feedback["awareness"] = "Nhận thức trung bình."
    elif target_trait == "N":
        if trait_val >= 0.65:
            feedback["awareness"] = "Người dùng nhạy cảm với thay đổi cảm xúc — dễ nhận biết dao động."
        else:
            feedback["awareness"] = "Nhận thức cảm xúc trung bình/ thấp."
    elif target_trait == "E":
        if trait_val >= 0.65:
            feedback["awareness"] = "Người dùng hướng ngoại — nhận thức tốt về cơ hội giao tiếp/hoạt động."
        elif trait_val <= 0.35:
            feedback["awareness"] = "Người dùng ít hướng ngoại — nhận thức hạn chế về cơ hội xã hội."
        else:
            feedback["awareness"] = "Nhận thức về hoạt động xã hội trung bình."
    elif target_trait == "O":
        if trait_val >= 0.65:
            feedback["awareness"] = "Người dùng cởi mở, dễ thử nghiệm/khám phá hành vi mới."
        else:
            feedback["awareness"] = "Nhận thức trung bình."
    elif target_trait == "C":
        if trait_val >= 0.65:
            feedback["awareness"] = "Người dùng cẩn thận, quan sát hành vi tốt."
        else:
            feedback["awareness"] = "Nhận thức trung bình."

    #motivation 
    if n >= 0.6 and contrib > 0:
        feedback["motivation"] = "Có động lực hiện tại — hành vi tăng/ổn định so với baseline."
    elif n <= 0.4 and contrib < 0:
        feedback["motivation"] = "Động lực yếu — hành vi giảm/không ổn định so với baseline."
    else:
        feedback["motivation"] = "Động lực trung bình — cần quan sát thêm nếu thay đổi nhỏ."

    #capability
    capability_score = 0.5
    try:
        if bt and bt > 0:
            if vt >= bt:
                capability_score = min(0.9, 0.5 + 0.4 * (vt / (bt + EPS)))
            else:
                capability_score = max(0.2, 0.5 * (vt / (bt + EPS)))
    except:
        capability_score = 0.5

    if capability_score >= 0.7:
        feedback["capability"] = "Người dùng có khả năng thực hiện hành vi; cần hỗ trợ duy trì."
    elif capability_score <= 0.35:
        feedback["capability"] = "Khả năng hiện thấp — cần can thiệp/kỹ năng hỗ trợ."
    else:
        feedback["capability"] = "Khả năng trung bình."

    #opportunity 
    if isinstance(vt, (int, float)):
        if vt <= 0 or (0 < vt < 0.1):
            opp = "Cơ hội thực hiện hành vi hạn chế."
        elif vt >= 0.6:
            opp = "Cơ hội thực hiện hành vi tốt."
        else:
            opp = "Cơ hội trung bình."
    else:
        opp = "Không đủ dữ liệu để đánh giá cơ hội."
    feedback["opportunity"] = opp

    direction_word = "tăng" if contrib > 0 else "giảm" if contrib < 0 else "không đổi"
    intensity = "mạnh" if abs(contrib) >= 0.2 else "vừa" if abs(contrib) >= 0.05 else "nhẹ"
    reason = (
        f"Metric `{metric_name}` ảnh hưởng tới trait `{target_trait}`: "
        f"n={n:.3f}, contrib={contrib:.3f} ({intensity} {direction_word}). "
        f"Baseline bt={bt:.3f}, vt={vt:.3f}."
    )

    return feedback, reason

class OceanScore(BaseModel):
    O: float = Field(..., ge=0.0, le=1.0)
    C: float = Field(..., ge=0.0, le=1.0)
    E: float = Field(..., ge=0.0, le=1.0)
    A: float = Field(..., ge=0.0, le=1.0)
    N: float = Field(..., ge=0.0, le=1.0)

class BaseResponse(BaseModel):
    metric: str
    vt: float
    bt: float
    r: float
    n: float
    contrib: float
    new_ocean_score: OceanScore
    mechanismFeedback: Dict[str, str]
    reason: Optional[str] = None

class AvgDailySpendIn(BaseModel):
    daily_total: float
    base_avg: float
    weight: float = 0.2
    direction: str = "down"
    sigma_r: float = 1.0
    alpha: float = 0.5
    ocean_score: OceanScore

class SpendVariabilityIn(BaseModel):
    daily_spend: List[float]  # 7 days
    base_likert: int  # 1..5
    weight: float = 0.2
    direction: str = "up"
    sigma_r: float = 1.0
    alpha: float = 0.5
    ocean_score: OceanScore

class BrandNoveltyIn(BaseModel):
    brands_prev: List[str]
    brands_now: List[str]
    base_likert: int
    weight: float = 0.2
    direction: str = "up"
    sigma_r: float = 1.0
    alpha: float = 0.5
    ocean_score: OceanScore

class TodoItem(BaseModel):
    task: str
    done: bool

class ListAdherenceInArray(BaseModel):
    todos: List[TodoItem]  # danh sách task + trạng thái hoàn thành
    base_likert: int        
    weight: float = 0.3
    direction: str = "up"
    sigma_r: float = 1.0
    alpha: float = 0.5
    ocean_score: OceanScore

class DailyDistanceIn(BaseModel):
    distance_today: float
    base_avg_distance: float
    weight: float = 0.2
    direction: str = "up"
    sigma_r: float = 1.0
    alpha: float = 0.5
    ocean_score: OceanScore

class NovelLocationIn(BaseModel):
    locations_prev: List[str]
    locations_now: List[str]
    base_likert: int
    weight: float = 0.2
    direction: str = "up"
    sigma_r: float = 1.0
    alpha: float = 0.5
    ocean_score: OceanScore

class PublicTransitIn(BaseModel):
    public_transit_trips: int
    total_trips: int
    base_likert: int
    weight: float = 0.2
    direction: str = "up"
    sigma_r: float = 1.0
    alpha: float = 0.5
    ocean_score: OceanScore

class NightOutIn(BaseModel):
    night_out_count: int
    base_night_out: int
    weight: float = 0.2
    direction: str = "up"
    sigma_r: float = 1.0
    alpha: float = 0.5
    ocean_score: OceanScore

class HealthyFoodIn(BaseModel):
    plant_meals: int
    total_meals: int
    base_likert: int
    weight: float = 0.25
    direction: str = "up"
    sigma_r: float = 1.0
    alpha: float = 0.5
    ocean_score: OceanScore


@router.post("/avg_daily_spend", response_model=BaseResponse)
async def avg_daily_spend(payload: AvgDailySpendIn):
    vt = float(payload.daily_total)
    bt = float(payload.base_avg)
    r = compute_r_from_positive(vt, bt)
    n = normalize_r_to_n(r, payload.sigma_r, payload.direction)
    contrib = compute_contrib(n, payload.weight)
    new = update_trait_score(payload.ocean_score.dict(), "C", contrib, payload.alpha)

    mechanismFeedback, reason = generate_mechanism_feedback(
        payload.ocean_score.dict(), "C", vt, bt, r, n, contrib, "avg_daily_spend"
    )

    return {
        "metric": "avg_daily_spend",
        "vt": vt,
        "bt": bt,
        "r": r,
        "n": n,
        "contrib": contrib,
        "new_ocean_score": new,
        "mechanismFeedback": mechanismFeedback,
        "reason": reason
    }

@router.post("/spend_variability", response_model=BaseResponse)
async def spend_variability(payload: SpendVariabilityIn):
    if len(payload.daily_spend) < 3:
        raise HTTPException(status_code=400, detail="need at least 3 days to compute variability")
    mean = statistics.mean(payload.daily_spend)
    std = statistics.pstdev(payload.daily_spend)
    cvt = std / max(mean, EPS)
    vt = cvt
    bt = likert_to_cv(payload.base_likert)
    r = compute_r_from_variability(vt, bt)
    n = normalize_r_to_n(r, payload.sigma_r, payload.direction)
    contrib = compute_contrib(n, payload.weight)
    new = update_trait_score(payload.ocean_score.dict(), "N", contrib, payload.alpha)

    mechanismFeedback, reason = generate_mechanism_feedback(
        payload.ocean_score.dict(), "N", vt, bt, r, n, contrib, "spend_variability"
    )

    return {
        "metric": "spend_variability",
        "vt": vt,
        "bt": bt,
        "r": r,
        "n": n,
        "contrib": contrib,
        "new_ocean_score": new,
        "mechanismFeedback": mechanismFeedback,
        "reason": reason
    }

@router.post("/brand_novelty", response_model=BaseResponse)
async def brand_novelty(payload: BrandNoveltyIn):
    prev = set(payload.brands_prev or [])
    now = payload.brands_now or []
    if not now:
        raise HTTPException(status_code=400, detail="brands_now required")
    novel = len([b for b in now if b not in prev])
    total = len(now)
    vt = novel / max(total, 1)
    bt = likert_to_ratio(payload.base_likert)
    r = compute_r_from_ratio(vt, bt)
    n = normalize_r_to_n(r, payload.sigma_r, payload.direction)
    contrib = compute_contrib(n, payload.weight)
    new = update_trait_score(payload.ocean_score.dict(), "O", contrib, payload.alpha)

    mechanismFeedback, reason = generate_mechanism_feedback(
        payload.ocean_score.dict(), "O", vt, bt, r, n, contrib, "brand_novelty"
    )

    return {
        "metric": "brand_novelty",
        "vt": vt,
        "bt": bt,
        "r": r,
        "n": n,
        "contrib": contrib,
        "new_ocean_score": new,
        "mechanismFeedback": mechanismFeedback,
        "reason": reason
    }

@router.post("/list_adherence", response_model=BaseResponse)
async def list_adherence_array(payload: ListAdherenceInArray):
    if not payload.todos:
        raise HTTPException(status_code=400, detail="todos cannot be empty")
    total = len(payload.todos)
    done_count = sum([1 for t in payload.todos if t.done])
    vt = done_count / total
    bt = likert_to_ratio(payload.base_likert)
    r = compute_r_from_ratio(vt, bt)
    n = normalize_r_to_n(r, payload.sigma_r, payload.direction)
    contrib = compute_contrib(n, payload.weight)
    new = update_trait_score(payload.ocean_score.dict(), "C", contrib, payload.alpha)

    mechanismFeedback, reason = generate_mechanism_feedback(
        payload.ocean_score.dict(), "C", vt, bt, r, n, contrib, "list_adherence"
    )

    return {
        "metric": "list_adherence",
        "vt": vt,
        "bt": bt,
        "r": r,
        "n": n,
        "contrib": contrib,
        "new_ocean_score": new,
        "mechanismFeedback": mechanismFeedback,
        "reason": reason
    }

@router.post("/daily_distance_km", response_model=BaseResponse)
async def daily_distance_km(payload: DailyDistanceIn):
    vt = float(payload.distance_today)
    bt = float(payload.base_avg_distance)
    r = compute_r_from_positive(vt, bt)
    n = normalize_r_to_n(r, payload.sigma_r, payload.direction)
    contrib = compute_contrib(n, payload.weight)
    new = update_trait_score(payload.ocean_score.dict(), "E", contrib, payload.alpha)

    mechanismFeedback, reason = generate_mechanism_feedback(
        payload.ocean_score.dict(), "E", vt, bt, r, n, contrib, "daily_distance_km"
    )

    return {
        "metric": "daily_distance_km",
        "vt": vt,
        "bt": bt,
        "r": r,
        "n": n,
        "contrib": contrib,
        "new_ocean_score": new,
        "mechanismFeedback": mechanismFeedback,
        "reason": reason
    }

@router.post("/novel_location_ratio", response_model=BaseResponse)
async def novel_location_ratio(payload: NovelLocationIn):
    prev = set(payload.locations_prev or [])
    now = payload.locations_now or []
    if not now:
        raise HTTPException(status_code=400, detail="locations_now required")
    novel = len([p for p in now if p not in prev])
    total = len(now)
    vt = novel / max(total, 1)
    bt = likert_to_ratio(payload.base_likert)
    r = compute_r_from_ratio(vt, bt)
    n = normalize_r_to_n(r, payload.sigma_r, payload.direction)
    contrib = compute_contrib(n, payload.weight)
    new = update_trait_score(payload.ocean_score.dict(), "O", contrib, payload.alpha)

    mechanismFeedback, reason = generate_mechanism_feedback(
        payload.ocean_score.dict(), "O", vt, bt, r, n, contrib, "novel_location_ratio"
    )

    return {
        "metric": "novel_location_ratio",
        "vt": vt,
        "bt": bt,
        "r": r,
        "n": n,
        "contrib": contrib,
        "new_ocean_score": new,
        "mechanismFeedback": mechanismFeedback,
        "reason": reason
    }

@router.post("/public_transit_ratio", response_model=BaseResponse)
async def public_transit_ratio(payload: PublicTransitIn):
    if payload.total_trips <= 0:
        raise HTTPException(status_code=400, detail="total_trips must be > 0")
    vt = payload.public_transit_trips / payload.total_trips
    bt = likert_to_ratio(payload.base_likert)
    r = compute_r_from_ratio(vt, bt)
    n = normalize_r_to_n(r, payload.sigma_r, payload.direction)
    contrib = compute_contrib(n, payload.weight)
    new = update_trait_score(payload.ocean_score.dict(), "A", contrib, payload.alpha)

    mechanismFeedback, reason = generate_mechanism_feedback(
        payload.ocean_score.dict(), "A", vt, bt, r, n, contrib, "public_transit_ratio"
    )

    return {
        "metric": "public_transit_ratio",
        "vt": vt,
        "bt": bt,
        "r": r,
        "n": n,
        "contrib": contrib,
        "new_ocean_score": new,
        "mechanismFeedback": mechanismFeedback,
        "reason": reason
    }

@router.post("/night_out_freq", response_model=BaseResponse)
async def night_out_freq(payload: NightOutIn):
    vt = float(payload.night_out_count)
    bt = float(payload.base_night_out)
    r = compute_r_from_positive(vt, bt)
    n = normalize_r_to_n(r, payload.sigma_r, payload.direction)
    contrib = compute_contrib(n, payload.weight)
    new = update_trait_score(payload.ocean_score.dict(), "E", contrib, payload.alpha)

    mechanismFeedback, reason = generate_mechanism_feedback(
        payload.ocean_score.dict(), "E", vt, bt, r, n, contrib, "night_out_freq"
    )

    return {
        "metric": "night_out_freq",
        "vt": vt,
        "bt": bt,
        "r": r,
        "n": n,
        "contrib": contrib,
        "new_ocean_score": new,
        "mechanismFeedback": mechanismFeedback,
        "reason": reason
    }

@router.post("/healthy_food_ratio", response_model=BaseResponse)
async def healthy_food_ratio(payload: HealthyFoodIn):
    if payload.total_meals <= 0:
        raise HTTPException(status_code=400, detail="total_meals must be > 0")
    vt = payload.plant_meals / payload.total_meals
    bt = likert_to_ratio(payload.base_likert)
    r = compute_r_from_ratio(vt, bt)
    n = normalize_r_to_n(r, payload.sigma_r, payload.direction)
    contrib = compute_contrib(n, payload.weight)
    # O strong, C light (30%)
    new_ocean = update_trait_score(payload.ocean_score.dict(), "O", contrib, payload.alpha)
    new_ocean = update_trait_score(new_ocean, "C", 0.3 * contrib, payload.alpha)

    mechanismFeedback, reason = generate_mechanism_feedback(
        payload.ocean_score.dict(), "O", vt, bt, r, n, contrib, "healthy_food_ratio"
    )

    return {
        "metric": "healthy_food_ratio",
        "vt": vt,
        "bt": bt,
        "r": r,
        "n": n,
        "contrib": contrib,
        "new_ocean_score": new_ocean,
        "mechanismFeedback": mechanismFeedback,
        "reason": reason
    }
