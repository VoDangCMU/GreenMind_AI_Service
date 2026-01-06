from typing import List, Dict, Optional, Literal
from fastapi import APIRouter
from pydantic import BaseModel, Field

router = APIRouter()

# --- Enum types ---
Trait = Literal["O", "C", "E", "A", "N"]
KeyDir = Literal["pos", "neg"]
KindType = Literal["frequency", "yesno", "likert5", "rating"]

# --- Models ---
class AnswerItem(BaseModel):
    trait: Trait
    text: Optional[str] = None
    ans: str
    key: Optional[KeyDir] = "pos"
    kind: Optional[KindType] = "frequency"
    weight: Optional[float] = 1.0

class UserAnswers(BaseModel):
    user_id: str
    answers: List[AnswerItem] = Field(default_factory=list)

class OceanResult(BaseModel):
    user_id: str
    scores: Dict[Trait, float]


ANSWER_MAP: Dict[str, int] = {
    # Frequency (VN)
    "rất ít khi": 1, "hiếm khi": 1, "0 lần": 1, "0": 1, "0lần": 1, "0 lần/tháng": 1,
    "thỉnh thoảng": 2, "đôi khi": 2, "1–2 lần": 2, "1-2 lần": 2, "1-2": 2, "1–2": 2,
    "thường xuyên": 3, "3–5 lần": 3, "3-5 lần": 3, "3-5": 3, "3–5": 3,
    "rất thường xuyên": 4, "hơn 5 lần": 4, ">5": 4, ">=6": 4,

    # Yes/No (VN)
    "có": 4, "không": 1,

    # Likert / Rating (1–5)
    "1": 1, "2": 2, "3": 3, "4": 4, "5": 5,

    # English
    "never": 1, "rarely": 1, "sometimes": 2, "occasionally": 2,
    "often": 3, "frequently": 3, "almost always": 4, "always": 4,
    "yes": 4, "no": 1,
}


WEIGHTS = {
    "frequency": 1.0,
    "yesno": 0.75,
    "likert5": 1.0,
    "rating": 1.0
}


def map_answer(ans: str) -> Optional[int]:
    if not ans:
        return None

    s = str(ans).strip().lower()
    digits = "".join(ch for ch in s if ch.isdigit())
    if digits:
        try:
            num = int(digits)
            if 1 <= num <= 5:
                return num
            if num == 0:
                return 1
            if num >= 6:
                return 4
        except ValueError:
            pass

    return ANSWER_MAP.get(s)


def reverse_if_needed(x: int, key: KeyDir) -> int:
    """Đảo ngược nếu câu hỏi là loại nghịch (neg)"""
    return (5 - x) if key == "neg" else x


def to_0_100(mean_score: float, max_scale: int = 5) -> float:
    """Chuyển từ thang 1–5 hoặc 1–4 thành 0–100"""
    return round(max(0.0, min(100.0, ((mean_score - 1.0) / (max_scale - 1)) * 100.0)), 2)


def score_trait(items: List[AnswerItem]) -> float:
    if not items:
        return 0.0

    weighted_sum = 0.0
    total_weight = 0.0

    for it in items:
        x = map_answer(it.ans)
        if x is None:
            continue

        x = reverse_if_needed(x, it.key or "pos")

        w = it.weight or WEIGHTS.get(it.kind or "frequency", 1.0)

        weighted_sum += x * w
        total_weight += w

    if total_weight == 0:
        return 0.0

    mean_val = weighted_sum / total_weight
    max_scale = 5 if any(it.kind in ["likert5", "rating"] for it in items) else 4
    return to_0_100(mean_val, max_scale=max_scale)


@router.post("/calculate_ocean", response_model=OceanResult)
async def calculate_ocean(data: UserAnswers):
    # Gom nhóm câu trả lời theo trait
    buckets: Dict[Trait, List[AnswerItem]] = {t: [] for t in ["O", "C", "E", "A", "N"]}
    for ans in data.answers:
        if ans.trait in buckets:
            buckets[ans.trait].append(ans)

    # Tính điểm cho từng trait
    results: Dict[Trait, float] = {
        trait: score_trait(items) for trait, items in buckets.items()
    }

    return OceanResult(user_id=data.user_id, scores=results)
