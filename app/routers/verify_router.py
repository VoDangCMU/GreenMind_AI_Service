from fastapi import FastAPI, APIRouter
from pydantic import BaseModel
from typing import Dict, Optional

app = FastAPI()
router = APIRouter()

EXPECTED_TRAIT_SCORE = {
    "O": 0.80,
    "C": 0.75,
    "E": 0.85,
    "A": 0.65,
    "N": 0.50
}

THRESHOLDS = {
    "medium": 0.15,  
    "critical": 0.30  
}


class ModelInfo(BaseModel):
    id: str
    ocean: str
    behavior: str
    age: str
    location: str
    gender: str
    keywords: Optional[str] = None

class VerifySurveyRequest(BaseModel):
    model: ModelInfo
    user_id: str
    survey_result: Dict[str, float]


@router.post("/verify-survey")
async def verify_survey(payload: VerifySurveyRequest):

    model_trait = payload.model.ocean.upper()
    actual_score = payload.survey_result.get(model_trait, 0)
    expected_score = EXPECTED_TRAIT_SCORE.get(model_trait, 0)

    deviation = round(abs(expected_score - actual_score), 3)

    if deviation <= THRESHOLDS["medium"]:
        level = "match"
        match = True
        feedback = ["Model phù hợp với nhóm user dựa trên khảo sát."]
    elif deviation <= THRESHOLDS["critical"]:
        level = "medium_mismatch"
        match = False
        feedback = [
            f"Trait {model_trait} hơi lệch so với kỳ vọng.",
            "Cần tinh chỉnh wording câu hỏi hoặc keywords."
        ]
    else:
        level = "critical_mismatch"
        match = False
        feedback = [
            f"Trait {model_trait} lệch quá nhiều so với kỳ vọng.",
            "Nên xem lại behavior mapping và demographic target."
        ]

    return {
        "model_id": payload.model.id,
        "user_id": payload.user_id,
        "trait_checked": model_trait,
        "expected": expected_score,
        "actual": actual_score,
        "deviation": deviation,
        "match": match,
        "level": level,
        "feedback": feedback
    }

app.include_router(router)
