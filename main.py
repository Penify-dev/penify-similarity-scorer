from fastapi import FastAPI, Request
from pydantic import BaseModel
from sentence_transformers import SentenceTransformer, util
import torch
import os
import pathlib
import platform

# Set up model cache directory
models_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "models")
os.makedirs(models_dir, exist_ok=True)

# Configure device for PyTorch
device = None
is_mac = platform.system() == "Darwin"
mac_gpu_available = is_mac and (torch.backends.mps.is_available() if hasattr(torch.backends, "mps") else False)
cuda_available = torch.cuda.is_available()

if mac_gpu_available:
    device = torch.device("mps")
    print("Using Apple MPS (Metal Performance Shaders) device")
elif cuda_available:
    device = torch.device("cuda")
    print("Using CUDA device")
else:
    device = torch.device("cpu")
    print("Using CPU device")

# Get model name from environment variable or use default
model_name = os.environ.get("MODEL_NAME", "all-mpnet-base-v2")
print(f"Loading model: {model_name} from cache directory: {models_dir}")

# Load model with caching in the models directory
# GPU will be used automatically if available
model = SentenceTransformer(model_name, cache_folder=models_dir)

# For macOS GPU (MPS), we need to explicitly move the model
if mac_gpu_available:
    try:
        # Try to move model to MPS device 
        model.to(device)
        print("Successfully moved model to MPS device")
    except Exception as e:
        print(f"Warning: Could not move model to MPS device: {e}")
        print("Falling back to CPU")

# FastAPI app
app = FastAPI(
    title="Semantic Similarity API",
    description="API for comparing semantic similarity between text strings",
    version="1.0.0"
)

# Route for system info
@app.get("/system-info")
async def system_info():
    """Get information about the system and available devices."""
    info = {
        "platform": platform.platform(),
        "python_version": platform.python_version(),
        "pytorch_version": torch.__version__,
        "device_info": {
            "cpu_available": True,
            "cpu_count": os.cpu_count(),
            "cuda_available": torch.cuda.is_available(),
            "mps_available": torch.backends.mps.is_available() if hasattr(torch.backends, "mps") else False,
            "current_device": str(device) if device else "cpu"
        },
        "model": {
            "name": model_name,
            "cache_dir": models_dir
        }
    }
    
    # Add GPU info if CUDA is available
    if torch.cuda.is_available():
        info["device_info"]["cuda_device_count"] = torch.cuda.device_count()
        info["device_info"]["cuda_device_name"] = torch.cuda.get_device_name(0)
    
    return info

# Request body structure
class CompareRequest(BaseModel):
    sentence1: str
    sentence2: str

# Route to compare sentences
@app.post("/compare")
async def compare_sentences(data: CompareRequest):
    s1 = data.sentence1
    s2 = data.sentence2

    # Explicitly use the configured device if available
    emb1 = model.encode(s1, convert_to_tensor=True)
    emb2 = model.encode(s2, convert_to_tensor=True)
    
    # If using macOS GPU, move tensors to the device
    if mac_gpu_available:
        try:
            emb1 = emb1.to(device)
            emb2 = emb2.to(device)
        except Exception as e:
            print(f"Warning: Failed to move tensors to MPS device: {e}")

    similarity = util.pytorch_cos_sim(emb1, emb2).item()

    return {
        "sentence1": s1,
        "sentence2": s2,
        "semantic_similarity": round(similarity, 4)
    }
