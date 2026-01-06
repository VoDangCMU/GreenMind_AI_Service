import json
import re
from fastapi import HTTPException
from app.models.gemini import get_gemini   # chỉnh path nếu cần


# ================== PROMPT (RÚT GỌN – CHỐNG CẮT JSON) ==================

PROMPT_TEMPLATE = """
Bạn là AI chuyên trích xuất dữ liệu hoá đơn.
Chỉ trả về JSON hợp lệ.
Không markdown. Không giải thích. Không cắt cụt.

Schema:
{{
  "doc": {{"source_id": "", "currency": "VND"}},
  "vendor": {{"name": ""}},
  "datetime": {{"date": "", "time": ""}},
  "items": [
    {{"raw_name": "", "quantity": 1, "unit_price": 0, "line_total": 0}}
  ],
  "totals": {{"grand_total": 0}}
}}

OCR:
{ocr_text}
"""


# ================== GEMINI CALL ==================

def _call_gemini(prompt: str, max_output_tokens: int = 4096) -> str:
    try:
        gemini = get_gemini()
        return gemini.chat_raw(
            system_prompt="Trả JSON hoá đơn, không text khác.",
            user_prompt=prompt,
            max_output_tokens=max_output_tokens
        ).strip()
    except Exception as e:
        raise HTTPException(status_code=502, detail=str(e))


# ================== JSON UTILS ==================

def _extract_json(raw: str) -> str:
    m = re.search(r"(\{.*\})", raw, flags=re.DOTALL)
    return m.group(1) if m else ""


# ================== MAIN PARSER ==================

def parse_invoice_text(ocr_text: str, retry: int = 2) -> dict:
    print("OCR TEXT:\n", ocr_text)

    prompt = PROMPT_TEMPLATE.format(ocr_text=ocr_text)
    raw = _call_gemini(prompt)

    print("RAW GEMINI:\n", raw)

    cleaned = raw.strip()
    json_text = cleaned if cleaned.startswith("{") else _extract_json(cleaned)

    try:
        return json.loads(json_text)

    except Exception as e:
        if retry > 0:
            print("[WARN] JSON lỗi / bị cắt → retry Gemini")
            return parse_invoice_text(ocr_text, retry - 1)

        raise HTTPException(
            status_code=500,
            detail=f"JSON lỗi sau retry: {e}"
        )
