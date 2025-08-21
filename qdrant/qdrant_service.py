from qdrant_client import QdrantClient
from qdrant_client.http import models as qmodels
import numpy as np
import uuid

qdrant_client = QdrantClient(host="localhost", port=6333)

def create_collection(collection_name, vector_size):
    qdrant_client.recreate_collection(
        collection_name=collection_name,
        vectors_config=qmodels.VectorParams(size=vector_size, distance="Cosine"),
    )

def insert_embeddings(collection_name, embeddings, payloads):
    points = [
        qmodels.PointStruct(
            id=str(uuid.uuid4()),
            vector=embedding.tolist(),
            payload=payload
        )
        for idx, (embedding, payload) in enumerate(zip(embeddings, payloads))
    ]
    qdrant_client.upsert(
        collection_name=collection_name,
        points=points
    )

def search_embeddings(collection_name, query_embedding, limit):
    search_result = qdrant_client.search(
        collection_name=collection_name,
        query_vector=query_embedding.tolist(),
        limit=limit
    )
    return search_result