from typing import List, Dict, Any
from fastapi import APIRouter, HTTPException
from pydantic import BaseModel
import json
import re

router = APIRouter()

class InputData(BaseModel):
    ocean: str
    behavior: str
    age: str
    location: str
    gender: str
    keywords: str

class Template(BaseModel):
    id: str
    name: str
    description: str
    intent: str
    placeholders: Dict[str, Any]
    prompt: str
    question_type: str
    answer: Dict[str, Any]

class CombineQuestionRequest(BaseModel):
    input: InputData
    templates: List[Template]


def fill_prompt(template_prompt: str, input_data: dict):
    """Điền dữ liệu từ input vào template"""
    def replace_placeholder(match):
        key = match.group(1)
        return input_data.get(key, f"{{{key}}}")  

    return re.sub(r"\{(.*?)\}", replace_placeholder, template_prompt)


@router.post("/combine_question")
async def gen_question(req: CombineQuestionRequest):
    try:
        input_dict = req.input.dict()
        results = []

        for tpl in req.templates:
            filled_prompt = fill_prompt(tpl.prompt, input_dict)
            results.append({
                "id": tpl.id,
                "name": tpl.name,
                "intent": tpl.intent,
                "question_type": tpl.question_type,
                "filled_prompt": filled_prompt,
                "answer": tpl.answer
            })

        return {"status": "success", "questions": results}

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
