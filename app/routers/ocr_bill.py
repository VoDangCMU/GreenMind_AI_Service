from fastapi import APIRouter, UploadFile, File, Form, HTTPException
from typing import Optional
from app.services.ocr_text import ocr_text_from_source, ocr_text_from_upload
# from app.services.ocr_parser import parse_invoice_text
from app.services.orc_parse_gemini import parse_invoice_text

router = APIRouter()

@router.post("/ocr_text")
async def ocr_text(file: Optional[UploadFile] = File(default=None),
                   url: Optional[str] = Form(default=None)):
    if not file and not url:
        raise HTTPException(400, "Cần upload file hoặc truyền url")
    if file:
        content = await file.read()
        text_md = ocr_text_from_upload(content, file.filename)
    else:
        text_md = ocr_text_from_source(url)

    if not text_md.strip():
        raise HTTPException(500, "OCR không ra được text")

    schema_json = parse_invoice_text(text_md)
    return schema_json