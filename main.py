from fastapi import FastAPI, Request
from pydantic import BaseModel
from contextlib import asynccontextmanager
import torch
import os
import platform
import logging
import time
import json
import sys
from datetime import datetime
import atexit

# Import simplified model service
from model_service import ModelService

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler(os.path.join(os.path.dirname(os.path.abspath(__file__)), 'app.log'))
    ]
)
logger = logging.getLogger("similarity-scorer")
logger.info("Starting Semantic Similarity Scorer application")
logger.info(f"Python version: {platform.python_version()}")
logger.info(f"PyTorch version: {torch.__version__}")
logger.info(f"Running on platform: {platform.platform()}")

# Set up model cache directory
models_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "models")
os.makedirs(models_dir, exist_ok=True)
logger.info(f"Model cache directory: {models_dir}")

# Configure device for PyTorch
device = None
is_mac = platform.system() == "Darwin"
mac_gpu_available = is_mac and (torch.backends.mps.is_available() if hasattr(torch.backends, "mps") else False)
cuda_available = torch.cuda.is_available()

logger.info(f"Device detection - Is Mac: {is_mac}")
logger.info(f"Device detection - MPS available: {mac_gpu_available}")
logger.info(f"Device detection - CUDA available: {cuda_available}")

if mac_gpu_available:
    device = torch.device("mps")
    logger.info("Using Apple MPS (Metal Performance Shaders) device")
elif cuda_available:
    device = torch.device("cuda")
    logger.info("Using CUDA device")
    logger.info(f"CUDA device count: {torch.cuda.device_count()}")
    logger.info(f"CUDA device name: {torch.cuda.get_device_name(0)}")
    logger.info(f"CUDA version: {torch.version.cuda}")
else:
    device = torch.device("cpu")
    logger.info("Using CPU device")
    logger.info(f"CPU count: {os.cpu_count()}")

# Get model name from environment variable or use default
model_name = os.environ.get("MODEL_NAME", "all-mpnet-base-v2")
logger.info(f"Loading model: {model_name} from cache directory: {models_dir}")

# Initialize the model service
start_time = time.time()
try:
    # Get the model service singleton - only creates one model instance
    model_service = ModelService.get_instance(model_name, models_dir, device)
    load_time = time.time() - start_time
    logger.info(f"Model service initialized in {load_time:.2f} seconds")
    
    # Register shutdown function to clean up the model service
    def shutdown_model_service():
        logger.info("Shutting down model service")
        try:
            model_service.shutdown()
            logger.info("Model service shutdown completed successfully")
        except Exception as e:
            logger.error(f"Error during model service shutdown: {e}")
    
    atexit.register(shutdown_model_service)
    
except Exception as e:
    logger.error(f"Failed to initialize model service: {e}")
    raise

# FastAPI lifespan context manager
@asynccontextmanager
async def lifespan(app: FastAPI):
    # Startup: verify model is properly loaded
    logger.info("Application startup - Verifying model service is properly loaded...")
    try:
        # Check model service health
        health_check_start = time.time()
        is_healthy = model_service.health_check()
        health_check_time = time.time() - health_check_start
        
        if not is_healthy:
            logger.error(f"Model service health check failed in {health_check_time:.4f} seconds")
            # Don't crash the app, but log that we're continuing with degraded functionality
            logger.warning("Application will continue but model functionality may be limited")
        else:    
            logger.info(f"Model service health check passed in {health_check_time:.4f} seconds")
            
            # Test similarity calculation
            s1 = "Hello world"
            s2 = "Hi there"
            sim_start = time.time()
            try:
                similarity = model_service.calculate_similarity(s1, s2)
                sim_time = time.time() - sim_start
                
                logger.info(f"Model verification successful. Similarity: {similarity:.4f}")
                logger.info(f"Similarity calculation time: {sim_time:.4f} seconds")
            except Exception as sim_error:
                logger.error(f"Similarity calculation test failed: {sim_error}")
                logger.warning("Application will continue but model functionality may be limited")
            
            # Log memory usage if possible
            try:
                import psutil
                process = psutil.Process(os.getpid())
                memory_info = process.memory_info()
                logger.info(f"Worker memory usage: {memory_info.rss / 1024 / 1024:.2f} MB")
            except ImportError:
                logger.info("psutil not available, memory usage stats skipped")
                
    except Exception as e:
        logger.error(f"Model verification failed: {e}")
        # We don't want to crash the app, but log the error
        
    logger.info("Application startup complete")
    
    # Yield control back to FastAPI
    yield
    
    # Shutdown logic
    logger.info("Application shutdown initiated")
    logger.info("Application shutdown complete")

# FastAPI app
app = FastAPI(
    title="Semantic Similarity API",
    description="API for comparing semantic similarity between text strings",
    version="1.0.0",
    docs_url="/docs",
    redoc_url="/redoc",
    openapi_url="/openapi.json",
    lifespan=lifespan
)
logger.info("FastAPI application initialized")

# Add CORS middleware
from fastapi.middleware.cors import CORSMiddleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Allows all origins
    allow_credentials=True,
    allow_methods=["*"],  # Allows all methods
    allow_headers=["*"],  # Allows all headers
)

# Add Gzip compression for responses
from fastapi.middleware.gzip import GZipMiddleware
app.add_middleware(GZipMiddleware, minimum_size=1000)  # Only compress responses larger than 1KB

# Request logging middleware
@app.middleware("http")
async def log_requests(request: Request, call_next):
    request_id = f"req-{int(time.time() * 1000)}"
    logger.info(f"Request started [ID: {request_id}] - {request.method} {request.url.path}")
    
    start_time = time.time()
    try:
        response = await call_next(request)
        process_time = time.time() - start_time
        logger.info(f"Request completed [ID: {request_id}] - Status: {response.status_code} - Time: {process_time:.4f}s")
        response.headers["X-Process-Time"] = str(process_time)
        return response
    except Exception as e:
        process_time = time.time() - start_time
        logger.error(f"Request failed [ID: {request_id}] - Error: {str(e)} - Time: {process_time:.4f}s")
        raise

# Add a health check endpoint
@app.get("/health")
async def health_check():
    """Health check endpoint with detailed metrics."""
    logger.info("Health check endpoint called")
    
    health_status = {
        "status": "healthy",
        "timestamp": datetime.now().isoformat(),
        "version": "1.0.0"
    }
    
    # Check model service status
    try:
        logger.info("Health check - Testing model service")
        health_check_start = time.time()
        model_service_healthy = model_service.health_check()
        health_check_time = time.time() - health_check_start
        
        logger.info(f"Health check - Model service: {model_service_healthy}, time: {health_check_time:.4f}s")
        
        if model_service_healthy:
            health_status["model_service"] = {
                "status": "healthy",
                "name": model_name,
                "response_time": health_check_time
            }
        else:
            health_status["status"] = "degraded"
            health_status["model_service"] = {
                "status": "error",
                "error": "Model service health check failed",
                "response_time": health_check_time
            }
            logger.error("Health check - Model service health check failed")
    except Exception as e:
        health_status["status"] = "degraded"
        health_status["model_service"] = {
            "status": "error",
            "error": str(e)
        }
        logger.error(f"Health check - Model service test failed: {e}")
    
    # Memory metrics
    try:
        import psutil
        process = psutil.Process(os.getpid())
        mem = process.memory_info()
        health_status["memory"] = {
            "rss_mb": mem.rss / 1024 / 1024,
            "vms_mb": mem.vms / 1024 / 1024,
            "percent": process.memory_percent()
        }
        
        # Check if memory usage is high
        if process.memory_percent() > 90:
            health_status["status"] = "warning"
            health_status["warnings"] = ["High memory usage"]
            logger.warning("Health check - High memory usage detected")
    except:
        health_status["memory"] = {"status": "unknown"}
    
    # CPU metrics
    try:
        health_status["cpu"] = {
            "percent": process.cpu_percent(interval=0.1),
            "num_threads": process.num_threads()
        }
    except:
        health_status["cpu"] = {"status": "unknown"}
    
    # Device info
    health_status["device"] = {"type": str(device)}
    
    logger.info(f"Health check result: {health_status['status']}")
    return health_status

# Request body structure
class CompareRequest(BaseModel):
    sentence1: str
    sentence2: str

# Route to compare sentences
@app.post("/compare")
async def compare_sentences(data: CompareRequest):
    request_id = f"req-{int(time.time() * 1000)}"
    logger.info(f"Compare endpoint called [ID: {request_id}]")
    logger.info(f"Sentence 1 ({len(data.sentence1)} chars): {data.sentence1[:50]}...")
    logger.info(f"Sentence 2 ({len(data.sentence2)} chars): {data.sentence2[:50]}...")
    
    # Calculate similarity
    sim_start = time.time()
    try:
        similarity = model_service.calculate_similarity(data.sentence1, data.sentence2)
        sim_time = time.time() - sim_start
        logger.info(f"[ID: {request_id}] Calculated similarity ({similarity:.4f}) in {sim_time:.4f}s")
    except Exception as e:
        logger.error(f"[ID: {request_id}] Similarity calculation failed: {str(e)}")
        raise
    
    result = {
        "sentence1": data.sentence1,
        "sentence2": data.sentence2,
        "semantic_similarity": round(similarity, 4),
        "processing_time_seconds": sim_time
    }
    
    logger.info(f"[ID: {request_id}] Request completed successfully")
    return result
