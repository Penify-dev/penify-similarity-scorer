from fastapi import FastAPI, Request
from pydantic import BaseModel
from sentence_transformers import SentenceTransformer, util
import torch
import os

# Get model name from environment variable or use default
model_name = os.environ.get("MODEL_NAME", "all-MiniLM-L6-v2")
print(f"Loading model: {model_name}")

# Load model (GPU will be used automatically if available)
model = SentenceTransformer(model_name)

# FastAPI app
app = FastAPI()

# Request body structure
class CompareRequest(BaseModel):
    sentence1: str
    sentence2: str

# Route to compare sentences
@app.post("/compare")
async def compare_sentences(data: CompareRequest):
    s1 = data.sentence1
    s2 = data.sentence2

    emb1 = model.encode(s1, convert_to_tensor=True)
    emb2 = model.encode(s2, convert_to_tensor=True)

    similarity = util.pytorch_cos_sim(emb1, emb2).item()

    return {
        "sentence1": s1,
        "sentence2": s2,
        "semantic_similarity": round(similarity, 4)
    }
