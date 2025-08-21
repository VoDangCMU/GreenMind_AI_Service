from functools import lru_cache
from sentence_transformers import SentenceTransformer

MODEL_ID = "hothanhtienqb/mind_map_blog_model"

@lru_cache(maxsize=1)
def get_model() -> SentenceTransformer:
    return SentenceTransformer(MODEL_ID)

def embed(texts):
    if isinstance(texts, str):
        texts = [texts]
    model = get_model()
    return model.encode(texts, convert_to_numpy=True, normalize_embeddings=True).tolist()
