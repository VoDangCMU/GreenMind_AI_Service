import sys
sys.path.append('E:/PizzaOder_FastAPI/yolov9')
from fastapi import FastAPI
from starlette.middleware.cors import CORSMiddleware
from app.routers.home import router as home_router
from app.routers.gen_question import router as gen_question
from app.routers.calculate_ocean import router as calculate_ocean
app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"]
)

app.include_router(home_router)
app.include_router(gen_question)
app.include_router(calculate_ocean)

