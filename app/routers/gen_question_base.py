# app/routers/gen_question.py
from typing import List, Literal
from fastapi import APIRouter, HTTPException
from pydantic import BaseModel
from app.ocean.load_template import load_templates
from app.models.qwen import get_qwen
import json

router = APIRouter()

OCEAN = Literal["O", "C", "E", "A", "N"]

class GenQuestionRequest(BaseModel):
    ocean: OCEAN
    behavior_input: str


class Option(BaseModel):
    text: str
    value: str


class QuestionItem(BaseModel):
    ocean: OCEAN
    template_id: str
    question: str
    options: List[Option]
    behavior_input: str
    behavior_normalized: str
    normalize_score: float


class GenQuestionResponse(BaseModel):
    total_templates: int
    items: List[QuestionItem]

def _all_templates_for_ocean(ocean: str) -> List[dict]:
    """Load all templates for a given OCEAN factor."""
    tpl = load_templates()
    block = tpl.get(ocean)
    if not block:
        raise HTTPException(status_code=400, detail=f"OCEAN không hợp lệ: {ocean}")

    return (block.get("frequency") or []) + (block.get("yesno") or [])


def _render_question(sentence: str, behavior: str) -> str:
    """Replace {behavior} placeholder in template with normalized behavior text."""
    return sentence.replace("{behavior}", behavior)


async def _llm_normalize_behavior(
    behavior_input: str,
    template_sentences: List[str]
) -> tuple[str, float]:
    """
    Use LLM to normalize user-provided behavior input into
    a natural short phrase suitable for inserting into templates.
    """
    qwen = get_qwen()
    joined = "\n".join([f"- {s}" for s in template_sentences])
    prompt = f"""
    Bạn là chuyên gia biên tập câu hỏi khảo sát hành vi.

    Input: "{behavior_input}"
    Templates sẽ dùng:
    {joined}

    Yêu cầu: chọn một "behavior" ngắn gọn, tự nhiên, phù hợp để thay vào {{behavior}} trong các template trên.
    Chỉ trả về JSON thuần, không giải thích, không thêm ```json.
    Trả về JSON duy nhất:
    {{"normalized":"<string>", "confidence":<0..100>}}
    """.strip()

    resp = qwen.chat(prompt)
    text = str(resp).strip()
    print(text)

    try:
        data = json.loads(text)
        normalized = str(data.get("normalized", "")).strip()
        confidence = float(data.get("confidence", 0))
    except Exception:
        raise HTTPException(status_code=500, detail="LLM trả về sai định dạng JSON")

    return normalized or behavior_input, confidence


@router.post("/gen-question", response_model=GenQuestionResponse)
async def gen_question(req: GenQuestionRequest) -> GenQuestionResponse:
    """Generate survey questions for a given OCEAN factor and behavior input."""
    templates = _all_templates_for_ocean(req.ocean)
    if not templates:
        raise HTTPException(status_code=404, detail=f"Không có template cho OCEAN={req.ocean}")

    sentences = [t["sentence"] for t in templates if "{behavior}" in t.get("sentence", "")]
    behavior_norm, conf = await _llm_normalize_behavior(req.behavior_input, sentences)

    items: List[QuestionItem] = []
    for t in templates:
        q_text = _render_question(t["sentence"], behavior_norm)
        options = [Option(text=v, value=v) for v in t.get("value_slot", [])]

        items.append(
            QuestionItem(
                ocean=req.ocean,
                template_id=t["template_id"],
                question=q_text,
                options=options,
                behavior_input=req.behavior_input,
                behavior_normalized=behavior_norm,
                normalize_score=conf,
            )
        )

    return GenQuestionResponse(total_templates=len(templates), items=items)
