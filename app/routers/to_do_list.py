# router/to_do_list.py
from fastapi import APIRouter, HTTPException
from pydantic import BaseModel
from app.models.gemini import get_gemini

router = APIRouter()

class TaskRequest(BaseModel):
    task: str

@router.get("/home")
async def home():
    return {"message": "ok"}

@router.post("/generate_subtasks")
async def generate_subtasks(request: TaskRequest):
    """
    Khi người dùng gửi 1 hành vi (task),
    hệ thống sẽ gọi Gemini để sinh ra các hành vi con.
    """
    gemini = get_gemini()

    try:
        # Prompt gửi đến Gemini
        prompt = f"""
        Hãy liệt kê các bước chính (keypoint) cần làm để hoàn thành công việc: "{request.task}".
        Trả về kết quả chỉ gồm các mục chính, dạng liệt kê ngắn gọn, mỗi dòng một bước 
        Không cần ghi chú hoặc giải thích chi tiết gì thêm
        """

        response_text = gemini.chat(prompt, max_output_tokens=1900)

        # Xử lý output thành list (cắt theo dòng hoặc dấu "-")
        subtasks = [
            line.strip("-• \n") 
            for line in response_text.split("\n") 
            if line.strip()
        ]

        return {"task": request.task, "subtasks": subtasks}

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
