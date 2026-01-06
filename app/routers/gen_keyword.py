from typing import Literal
from fastapi import APIRouter, HTTPException
from pydantic import BaseModel
from app.models.gemini import get_gemini
import re

router = APIRouter()

OCEAN = Literal["O", "C", "E", "A", "N"]

class KeywordRequest(BaseModel):
    ocean: OCEAN
    behavior: str
    age: str
    location: str
    gender: str


import json
import re

def extract_keywords(response: str):
    """
    Lấy danh sách keywords từ response của Gemini.
    Luôn trả về array string sạch, loại bỏ nested array.
    """
    # Xử lý smart quotes, block ```
    cleaned = response.replace("“", '"').replace("”", '"')
    cleaned = re.sub(r"```[\s\S]*?```", "", cleaned).strip()

    # Nếu Gemini trả dạng JSON array trong string → parse JSON
    try:
        data = json.loads(cleaned)
        if isinstance(data, list):
            return [k.strip() for k in data if isinstance(k, str)]
        elif isinstance(data, dict) and "keywords" in data:
            return [k.strip() for k in data["keywords"] if isinstance(k, str)]
    except json.JSONDecodeError:
        pass

    # Fallback: loại bỏ [ ] và tách chuỗi
    cleaned = re.sub(r'^\[+', '', cleaned)
    cleaned = re.sub(r'\]+$', '', cleaned)
    keywords = [k.strip().strip('"') for k in re.split(r',|\n', cleaned) if k.strip()]
    return keywords



@router.post("/gen_keyword")
async def gen_keyword(req: KeywordRequest):
    try:
        llm = get_gemini()

        prompt = f"""
                Bạn là chuyên gia tâm lý và marketing.  
                Hãy phân tích hồ sơ người dùng dưới đây để đề xuất 3 từ khóa (keyword), với mỗi từ khóa phản ánh đúng tính cách (OCEAN) và hành vi của họ và độ tuổi khu vực sống và giới tính liên quan rõ ràng

                Giải thích nhanh về mô hình OCEAN (Big Five Personality Traits):
                - O: Openness: Mức độ sáng tạo, tò mò, yêu thích trải nghiệm mới.
                - C: Conscientiousness: Mức độ kỷ luật, có tổ chức, đáng tin cậy.
                - E: Extraversion: Mức độ hướng ngoại, năng động, thích giao tiếp.
                - A: Agreeableness: Mức độ thân thiện, hợp tác, dễ đồng cảm.
                - N: Neuroticism: Mức độ nhạy cảm, dễ lo âu hoặc ổn định cảm xúc.

                Thông tin đầu vào:  
                - OCEAN trait: {req.ocean}  
                - Hành vi (Behavior): {req.behavior}  
                - Độ tuổi: {req.age}  
                - Khu vực sinh sống: {req.location}  
                - Giới tính: {req.gender}

                Yêu cầu:
                - Trả về duy nhất một đối tượng JSON hợp lệ.  
                - Không giải thích, không thêm ký hiệu ```json.  
                - Mỗi từ khóa là 1 hành động cụ thể phải có đầy đủ, phản ánh hết các đặc điểm của tính cách (OCEAN) nêu trên và hành vi của họ và độ tuổi khu vực sống và giới tính
                - - Trả về duy nhất **3 từ khóa**, chỉ trả array string JSON:
                ["keyword1", "keyword2", "keyword3"]
                - Không giải thích, không thêm ký tự khác, không trả object
                Ví dụ format JSON: {{ "keywords": ["streetwear", "clean fashion", "outdoor style"] }}

                """


        response = llm.chat(prompt, max_output_tokens=712)

        print("\n========== RAW RESPONSE FROM GEMINI ==========")
        print(response)
        print("==============================================\n")

        keywords = extract_keywords(response)

        keywords = keywords[:3]

        return {"input": req.dict(), "output": {"keywords": keywords}}

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))