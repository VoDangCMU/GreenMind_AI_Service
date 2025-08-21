from fastapi import APIRouter
import httpx 
router = APIRouter()

@router.get("/home")
async def home():
    return {"message": "ok"}
