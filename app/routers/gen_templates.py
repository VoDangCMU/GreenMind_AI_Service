from fastapi import APIRouter, HTTPException
from pydantic import BaseModel
from typing import Literal
import json
import importlib.resources as resources

router = APIRouter()

OCEAN = Literal["O", "C", "E", "A", "N"]

class TemplateRequest(BaseModel):
    ocean: OCEAN = None
    behavior: str = None
    age: str = None
    location: str = None
    gender: str = None
    keywords: str = None

# Import templates.json
with resources.files("app.ocean").joinpath("templates.json").open("r", encoding="utf-8") as f:
    TEMPLATES = json.load(f)

@router.post("/gen_template")
async def gen_template(req: TemplateRequest):
    try:
        user_params = {k: v for k, v in req.dict().items() if v is not None}
        user_keys = set(user_params.keys())

        matched_templates = []

        for t in TEMPLATES:
            used_placeholders = set(t.get("placeholders", {}).get("used_placeholders", []))

            # ✅ Template hợp lệ nếu số param ≤ số user gửi và tất cả used_placeholders có trong user_keys
            if used_placeholders.issubset(user_keys) and len(used_placeholders) <= len(user_keys):
                filled_prompt = t["prompt"].format(**{ph: user_params[ph] for ph in used_placeholders})
                temp = t.copy()
                temp["filled_prompt"] = filled_prompt
                matched_templates.append(temp)

        return {"input": user_params, "templates": matched_templates}

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

