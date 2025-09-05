from fastapi import APIRouter
from pydantic import BaseModel
from typing import List, Optional
import json, re

from app.models.qwen import get_qwen   

router = APIRouter()

class QuestionRequest(BaseModel):
    trait: str                
    behavior: str             
    num_questions: int     

class QuestionItem(BaseModel):
    text: str
    trait: str
    behavior: str
    reverse: Optional[bool] = False

class QuestionResponse(BaseModel):
    questions: List[QuestionItem]


@router.get("/home")
async def home():
    return {"message": "ok"}

def extract_json_from_output(raw_output: str) -> dict:
    """
    Tìm và parse JSON trong output LLM, kể cả khi bị bọc trong ```json ... ``` 
    hoặc có text thừa trước/sau.
    """
    cleaned = raw_output.strip()

    if cleaned.startswith("```"):
        cleaned = re.sub(r"^```[a-zA-Z]*\n", "", cleaned)
        cleaned = re.sub(r"\n```$", "", cleaned)

    try:
        return json.loads(cleaned)
    except Exception:
        match = re.search(r"\{.*\}", cleaned, re.DOTALL)
        if match:
            return json.loads(match.group())
    return {"questions": []}

@router.post("/generate_questions", response_model=QuestionResponse)
async def generate_questions(req: QuestionRequest):
    qwen = get_qwen()

    system_prompt = "Bạn là chuyên gia tâm lý học, nhiệm vụ là sinh câu hỏi hành vi dựa trên OCEAN + behavior."
    user_prompt = f"""
    Trait: {req.trait}
    Behavior domain (hành vi tổng quan): {req.behavior}

    Hãy sinh {req.num_questions} câu hỏi ngắn gọn (≤ 22 từ) bằng tiếng Việt.
    - Câu hỏi phải đo lường được hành vi tổng quan này trong nhiều tình huống (đa nguồn).
    - Không nhắc trực tiếp tên trait
    - Câu hỏi phải quan sát được hành vi, có thể trả lời Yes/No hoặc Likert
    - Đầu ra JSON với cấu trúc:
    {{
        "questions": [
        {{"text": "...", "trait": "{req.trait}", "behavior": "{req.behavior}", "reverse": false}}
        ]
    }}
    """


    raw_output = qwen.chat_raw(system_prompt, user_prompt, max_new_tokens=512)
    print("=== RAW OUTPUT ===")
    print(repr(raw_output))

    data = extract_json_from_output(raw_output)

    return QuestionResponse(**data)