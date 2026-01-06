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
    base_likert: int        # 1..5
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
    # compute
    r = compute_r_from_positive(vt, bt)
    n = normalize_r_to_n(r, payload.sigma_r, payload.direction)
    contrib = compute_contrib(n, payload.weight)
    new = update_trait_score(payload.ocean_score.dict(), "C", contrib, payload.alpha)
    return {
        "metric": "avg_daily_spend",
        "vt": vt,
        "bt": bt,
        "r": r,
        "n": n,
        "contrib": contrib,
        "new_ocean_score": new
    }

@router.post("/spend_variability", response_model=BaseResponse)
async def spend_variability(payload: SpendVariabilityIn):
    if len(payload.daily_spend) < 3:
        raise HTTPException(status_code=400, detail="need at least 3 days to compute variability")
    # compute std and mean -> CV
    mean = statistics.mean(payload.daily_spend)
    std = statistics.pstdev(payload.daily_spend)  # population std
    cvt = std / max(mean, EPS)
    vt = cvt
    bt = likert_to_cv(payload.base_likert)
    r = compute_r_from_variability(vt, bt)
    n = normalize_r_to_n(r, payload.sigma_r, payload.direction)
    contrib = compute_contrib(n, payload.weight)
    new = update_trait_score(payload.ocean_score.dict(), "N", contrib, payload.alpha)
    return {
        "metric": "spend_variability",
        "vt": vt,
        "bt": bt,
        "r": r,
        "n": n,
        "contrib": contrib,
        "new_ocean_score": new
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
    return {
        "metric": "brand_novelty",
        "vt": vt,
        "bt": bt,
        "r": r,
        "n": n,
        "contrib": contrib,
        "new_ocean_score": new
    }

@router.post("/list_adherence", response_model=BaseResponse)
async def list_adherence_array(payload: ListAdherenceInArray):
    if not payload.todos:
        raise HTTPException(status_code=400, detail="todos cannot be empty")
    
    total = len(payload.todos)
    done_count = sum([1 for t in payload.todos if t.done])
    
    vt = done_count / total                 # tỷ lệ hoàn thành
    bt = likert_to_ratio(payload.base_likert)
    
    r = compute_r_from_ratio(vt, bt)
    n = normalize_r_to_n(r, payload.sigma_r, payload.direction)
    contrib = compute_contrib(n, payload.weight)
    
    new = update_trait_score(payload.ocean_score.dict(), "C", contrib, payload.alpha)
    
    return {
        "metric": "list_adherence",
        "vt": vt,
        "bt": bt,
        "r": r,
        "n": n,
        "contrib": contrib,
        "new_ocean_score": new
    }


@router.post("/daily_distance_km", response_model=BaseResponse)
async def daily_distance_km(payload: DailyDistanceIn):
    vt = float(payload.distance_today)
    bt = float(payload.base_avg_distance)
    r = compute_r_from_positive(vt, bt)
    n = normalize_r_to_n(r, payload.sigma_r, payload.direction)
    contrib = compute_contrib(n, payload.weight)
    new = update_trait_score(payload.ocean_score.dict(), "E", contrib, payload.alpha)
    return {
        "metric": "daily_distance_km",
        "vt": vt,
        "bt": bt,
        "r": r,
        "n": n,
        "contrib": contrib,
        "new_ocean_score": new
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
    return {
        "metric": "novel_location_ratio",
        "vt": vt,
        "bt": bt,
        "r": r,
        "n": n,
        "contrib": contrib,
        "new_ocean_score": new
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
    return {
        "metric": "public_transit_ratio",
        "vt": vt,
        "bt": bt,
        "r": r,
        "n": n,
        "contrib": contrib,
        "new_ocean_score": new
    }

@router.post("/night_out_freq", response_model=BaseResponse)
async def night_out_freq(payload: NightOutIn):
    vt = float(payload.night_out_count)
    bt = float(payload.base_night_out)
    r = compute_r_from_positive(vt, bt)
    n = normalize_r_to_n(r, payload.sigma_r, payload.direction)
    contrib = compute_contrib(n, payload.weight)
    new = update_trait_score(payload.ocean_score.dict(), "E", contrib, payload.alpha)
    return {
        "metric": "night_out_freq",
        "vt": vt,
        "bt": bt,
        "r": r,
        "n": n,
        "contrib": contrib,
        "new_ocean_score": new
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
    return {
        "metric": "healthy_food_ratio",
        "vt": vt,
        "bt": bt,
        "r": r,
        "n": n,
        "contrib": contrib,
        "new_ocean_score": new_ocean
    }
