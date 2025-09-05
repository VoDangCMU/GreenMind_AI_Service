from fastapi import APIRouter
from pydantic import BaseModel
from typing import List, Dict

router = APIRouter()

class AnswerItem(BaseModel):
    trait: str             
    text: str              
    answer: int            
    reverse: bool = False
    answer_type: str = "likert_5"   

class UserAnswers(BaseModel):
    user_id: str
    answers: List[AnswerItem]

class OceanScore(BaseModel):
    user_id: str
    scores: Dict[str, float]   

def normalize(answer: int, reverse: bool, answer_type: str) -> float:
    if answer_type == "likert_5":
        val = 6 - answer if reverse else answer
        return (val - 1) / 4   
    elif answer_type == "yes_no":
        if reverse:
            val = 0 if answer == 1 else 1
        else:
            val = answer
        return float(val)      # 0 hoặc 1
    else:
        return 0.0

@router.post("/calculate_ocean")
async def calculate_ocean(data: UserAnswers):
    scores = {"O": [], "C": [], "E": [], "A": [], "N": []}

    for ans in data.answers:
        score = ans.answer
        if ans.answer_type == "likert_5":
            if ans.reverse:
                score = 6 - ans.answer
        elif ans.answer_type == "yes_no":
            score = 1 if ans.answer else 0
            if ans.reverse:
                score = 1 - score

        scores[ans.trait].append(score)

    avg_scores = {}
    for trait in scores:
        if scores[trait]:
            avg_scores[trait] = sum(scores[trait]) / len(scores[trait])
        else:
            avg_scores[trait] = 0.0  

    return {
        "user_id": data.user_id,
        "scores": avg_scores
    }
