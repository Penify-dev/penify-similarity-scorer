from fastapi import FastAPI, Request
from pydantic import BaseModel
from sentence_transformers import SentenceTransformer, util
import torch
import os
import pathlib

# Set up model cache directory
models_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "models")
os.makedirs(models_dir, exist_ok=True)

# Get model name from environment variable or use default
model_name = os.environ.get("MODEL_NAME", "all-mpnet-base-v2")
print(f"Loading model: {model_name} from cache directory: {models_dir}")

# Load model with caching in the models directory
# GPU will be used automatically if available
model = SentenceTransformer(model_name, cache_folder=models_dir)

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
