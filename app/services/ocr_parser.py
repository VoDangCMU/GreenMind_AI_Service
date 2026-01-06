import json
from fastapi import HTTPException
from app.models.qwen import get_qwen

_SCHEMA_DESC = """Bạn là AI chuyên trích xuất dữ liệu hoá đơn.
Trả về JSON đúng theo schema sau, không thêm chữ thừa, không thêm giải thích:

{
  "doc": {
    "source_id": "<string - nếu không có thì sinh uuid>",
    "currency": "VND",
    "payment_method": "cash",
    "notes": ""
  },
  "vendor": {
    "name": "<string>",
    "address": "",
    "geo_hint": ""
  },
  "datetime": {
    "date": "dd/MM/yyyy",
    "time": "HH:mm"
  },
  "items": [
    {
      "raw_name": "<tên dòng hàng từ OCR>",
      "brand": "",
      "category": "",
      "plant_based": false,
      "quantity": 1,
      "unit_price": 0,
      "line_total": 0,
      "matched_shopping_list": false
    }
  ],
  "totals": {
    "subtotal": 0,
    "discount": 0,
    "tax": 0,
    "grand_total": 0
  }
}
"""

def parse_invoice_text(ocr_text: str) -> dict:
    print(ocr_text)
    llm = get_qwen()
    prompt = f"""
OCR text:
---
{ocr_text}
---

Nhiệm vụ: Parse text trên thành JSON theo schema đã cho. 
Chỉ trả về JSON hợp lệ, không thêm chữ khác.
    """.strip()

    resp = llm.chat(prompt, max_new_tokens=512).strip()
    try:
        return json.loads(resp)
    except Exception:
        raise HTTPException(status_code=500, detail=f"LLM trả về không phải JSON: {resp}")
