from typing import List, Dict, Literal
from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, Field

router = APIRouter()

Trait = Literal["O", "C", "E", "A", "N"]

class TargetSegment(BaseModel):
    location_detail: str
    age_detail: int
    gender_detail: str


class UserOceanScore(BaseModel):
    user_id: str
    scores: Dict[Trait, float]


class GroupOceanRequest(BaseModel):
    target_segment: TargetSegment
    users: List[UserOceanScore] = Field(..., min_items=1)


class GroupOceanResponse(BaseModel):
    segment_profile: Dict
    group_ocean_score: Dict[Trait, float]



def calculate_group_ocean(users: List[UserOceanScore]) -> Dict[Trait, float]:
    trait_sum = {t: 0.0 for t in ["O", "C", "E", "A", "N"]}
    trait_count = {t: 0 for t in ["O", "C", "E", "A", "N"]}

    for user in users:
        for trait, value in user.scores.items():
            trait_sum[trait] += value
            trait_count[trait] += 1

    return {
        trait: round(trait_sum[trait] / trait_count[trait], 2)
        if trait_count[trait] > 0 else 0.0
        for trait in trait_sum
    }



@router.post("/calculate_group_ocean", response_model=GroupOceanResponse)
async def calculate_group_ocean_api(req: GroupOceanRequest):
    if not req.users:
        raise HTTPException(status_code=400, detail="Users list cannot be empty")

    group_score = calculate_group_ocean(req.users)

    return {
        "segment_profile": {
            "location_detail": req.target_segment.location_detail,
            "age_detail": req.target_segment.age_detail,
            "gender_detail": req.target_segment.gender_detail,
            "population_size": len(req.users)
        },
        "group_ocean_score": group_score
    }
