from typing import Literal, List
from fastapi import APIRouter, HTTPException
from pydantic import BaseModel
from app.models.gemini import get_gemini
import json
import re

router = APIRouter()

OCEAN = Literal["O", "C", "E", "A", "N"]

class Population(BaseModel):
    age_range: str                    # "18-25"
    gender: List[str]                 # ["male", "female"]
    locations: List[str]              # ["Da Nang", "Quang Nam"]
    urban: bool                       # True / False


class Context(BaseModel):
    population: Population
    setting: str                      # "gamified environmental campaign"
    event: str                        # "beach cleanup challenge"


class KeywordRequest(BaseModel):
    ocean: OCEAN
    behavior: str
    context: Context



def extract_keywords(response: str):
    """
    Parse Gemini output → always return clean array[str]
    """
    cleaned = response.replace("“", '"').replace("”", '"')
    cleaned = re.sub(r"```[\s\S]*?```", "", cleaned).strip()

    try:
        data = json.loads(cleaned)
        if isinstance(data, list):
            return [k.strip() for k in data if isinstance(k, str)]
        if isinstance(data, dict) and "keywords" in data:
            return [k.strip() for k in data["keywords"] if isinstance(k, str)]
    except json.JSONDecodeError:
        pass

    cleaned = re.sub(r'^\[+', '', cleaned)
    cleaned = re.sub(r'\]+$', '', cleaned)
    return [k.strip().strip('"') for k in re.split(r',|\n', cleaned) if k.strip()]


@router.post("/gen_keyword_ver2")
async def gen_keyword(req: KeywordRequest):
    try:
        llm = get_gemini()

        pop = req.context.population

        prompt = f"""
Bạn là chuyên gia nghiên cứu hành vi, xã hội học và thiết kế chiến lược
tác động môi trường cho dự án GreenMind.

Nhiệm vụ:
Phân tích mức độ ENGAGEMENT tiềm năng của một nhóm dân cư
trong một CONTEXT gồm Population + Setting + Event,
dựa trên đặc điểm tính cách OCEAN và hành vi mục tiêu.

Personality (OCEAN):
- Trait: {req.ocean}

Target behavior:
- {req.behavior}

Population:
- Age range: {pop.age_range}
- Gender: {pop.gender}
- Locations: {pop.locations}
- Urban area: {pop.urban}

Setting (chiến lược tác động):
- {req.context.setting}

Event (sự kiện cụ thể):
- {req.context.event}

QUY TẮC OUTPUT (BẮT BUỘC TUÂN THỦ theo nguyên tắc):
- Chỉ trả về 1 JSON array gồm 3 phần tử
- KHÔNG markdown
- KHÔNG ```json
- KHÔNG giải thích
- KHÔNG xuống dòng trong keyword
- Mỗi keyword là 1 hành động thật hay phản ánh đầy đủ cụ thể Setting và Event ở trên

FORMAT DUY NHẤT ĐƯỢC PHÉP:
["keyword1", "keyword2", "keyword3"]
"""

        response = llm.chat(prompt, max_output_tokens=1599)

        print("\n========== RAW RESPONSE ==========")
        print(response)
        print("=================================\n")

        keywords = extract_keywords(response)[:3]

        return {
            "context": req.context.dict(),
            "output": {
                "keywords": keywords
            }
        }

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
