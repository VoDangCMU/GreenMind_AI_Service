import sys
import os
# os.environ["HF_HUB_DISABLE_SYMLINKS"] = "1"
# os.environ["HF_HOME"] = os.path.join(os.getcwd(), "hf_cache")
from fastapi import FastAPI
from starlette.middleware.cors import CORSMiddleware
from app.routers.home import router as home_router
from app.routers.gen_question_base import router as gen_question
from app.routers.calculate_ocean import router as calculate_ocean
from app.routers.calculate_group_ocean import router as calculate_group_ocean
from app.routers.gen_question_base import router as gen_question_base
from app.routers.ocr_bill import router as ocr_bill
from app.routers.gen_keyword import router as gen_keyword
from app.routers.gen_keyword_ver2 import router as gen_keyword_ver2

from app.routers.gen_templates import router as gen_templates
from app.routers.combine_question import router as combine_question
from app.routers.monitor_ocean import router as monitor_ocean
from app.routers.detect_plants import router as detect_plants
from app.routers.to_do_list import router as to_do_list
from app.routers.verify_router import router as verify_router
from app.routers.verify_behavior import router as verify_behavior
from app.routers.detect_trash import router as detect_trash
from app.routers.predict_pollutant import router as predict_pollutant
from app.routers.score_green import router as score_green
from app.routers.detect_trash_ver2 import router as detect_trash_ver2
from app.routers.predict_masskg import router as predict_masskg
from app.routers.detect_trash_seg import router as detect_trash_seg
from app.routers.new import router as new
from app.models.embedding import get_model
from app.models.qwen import get_qwen
from app.models.gemini import get_gemini


app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"]
)

# @app.on_event("startup")
# async def warmup():
    # emb = get_model()
    # print("Embedding model loaded:", emb.get_sentence_embedding_dimension(), "dims")

    # Load LLM Qwen
    # llm = get_qwen()
    # out = llm.chat("Xin chào!", max_new_tokens=10)
    # print("LLM Qwen loaded, test response:", out)

    # gemini = get_gemini()
    # out_gem = gemini.chat("Xin chào!", max_output_tokens=10)
    # print("LLM Gemini loaded, test response:", out_gem)


app.include_router(home_router)
app.include_router(new)
app.include_router(gen_question)
app.include_router(predict_pollutant)
app.include_router(calculate_ocean)
app.include_router(calculate_group_ocean)
app.include_router(detect_trash)
app.include_router(gen_question_base)
app.include_router(ocr_bill)
app.include_router(gen_keyword)
app.include_router(predict_masskg)
app.include_router(gen_keyword_ver2)
app.include_router(detect_trash_seg)
app.include_router(gen_templates)
app.include_router(combine_question)
app.include_router(detect_trash_ver2)
# app.include_router(monitor_ocean)
app.include_router(detect_plants)
app.include_router(to_do_list)
app.include_router(verify_router)
app.include_router(verify_behavior)
app.include_router(score_green)