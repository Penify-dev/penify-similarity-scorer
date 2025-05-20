from fastapi import FastAPI, Request
from pydantic import BaseModel
from contextlib import asynccontextmanager
from sentence_transformers import SentenceTransformer, util
import torch
import os
import pathlib
import platform
import logging
import time
import json
import sys
from datetime import datetime
import atexit

# Import model service
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

# Initialize the model service instead of loading the model directly
# This creates a single shared model across all workers
start_time = time.time()
try:
    # Get the model service singleton
    model_service = ModelService.get_instance(model_name, models_dir, device)
    load_time = time.time() - start_time
    logger.info(f"Model service initialized in {load_time:.2f} seconds")
    
    # Register shutdown function to clean up the model service
    def shutdown_model_service():
        logger.info("Shutting down model service")
        model_service.shutdown()
    
    atexit.register(shutdown_model_service)
    
except Exception as e:
    logger.error(f"Failed to initialize model service: {e}")
    raise

# FastAPI lifespan context manager
@asynccontextmanager
async def lifespan(app: FastAPI):
    # Startup: verify model is properly loaded
    logger.info("Application startup - Verifying model service is properly loaded...")
    logger.info(f"Worker PID: {os.getpid()}")
    try:
        s1 = "Hello world"
        s2 = "Hi there"
        
        # Check model service health
        health_check_start = time.time()
        is_healthy = model_service.health_check()
        health_check_time = time.time() - health_check_start
        
        if not is_healthy:
            logger.error("Model service health check failed")
        else:    
            logger.info(f"Model service health check passed in {health_check_time:.4f} seconds")
            
            # Test similarity calculation
            sim_start = time.time()
            similarity = model_service.calculate_similarity(s1, s2)
            sim_time = time.time() - sim_start
            
            logger.info(f"Model verification successful. Similarity: {similarity:.4f}")
            logger.info(f"Similarity calculation time: {sim_time:.4f} seconds")
            
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
    
    # Log memory usage before shutdown
    try:
        import psutil
        process = psutil.Process(os.getpid())
        memory_info = process.memory_info()
        logger.info(f"Final memory usage: {memory_info.rss / 1024 / 1024:.2f} MB")
        
        # Log runtime statistics
        uptime = time.time() - process.create_time()
        logger.info(f"Application uptime: {uptime:.2f} seconds ({uptime/60:.2f} minutes)")
    except:
        logger.info("Could not log final memory usage")
        
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
logger.info("CORS middleware configured")

# Add Gzip compression for responses
from fastapi.middleware.gzip import GZipMiddleware
app.add_middleware(GZipMiddleware, minimum_size=1000)  # Only compress responses larger than 1KB
logger.info("GZip middleware configured")

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
        model_service_healthy = model_service.health_check()
        
        if model_service_healthy:
            health_status["model_service"] = {
                "status": "healthy",
                "name": model_name
            }
        else:
            health_status["status"] = "degraded"
            health_status["model_service"] = {
                "status": "error",
                "error": "Model service health check failed"
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

# Route for system info
@app.get("/system-info")
async def system_info():
    """Get information about the system and available devices."""
    logger.info("System info endpoint called")
    
    # Try to get additional system info
    memory_info = {}
    try:
        import psutil
        process = psutil.Process(os.getpid())
        mem = process.memory_info()
        memory_info = {
            "rss_mb": mem.rss / 1024 / 1024,
            "vms_mb": mem.vms / 1024 / 1024,
            "percent": process.memory_percent(),
            "system_total_mb": psutil.virtual_memory().total / 1024 / 1024,
            "system_available_mb": psutil.virtual_memory().available / 1024 / 1024,
            "system_percent": psutil.virtual_memory().percent
        }
    except ImportError:
        logger.info("psutil not available, memory info limited")
        memory_info = {"error": "psutil not installed"}
    
    # Get uptime
    try:
        import time
        uptime = time.time() - process.create_time()
        uptime_info = {
            "seconds": uptime,
            "minutes": uptime / 60,
            "hours": uptime / 3600,
        }
    except:
        uptime_info = {"error": "Could not determine uptime"}
    
    info = {
        "timestamp": datetime.now().isoformat(),
        "platform": platform.platform(),
        "processor": platform.processor(),
        "python_version": platform.python_version(),
        "python_implementation": platform.python_implementation(),
        "pytorch_version": torch.__version__,
        "device_info": {
            "cpu_available": True,
            "cpu_count": os.cpu_count(),
            "cuda_available": torch.cuda.is_available(),
            "mps_available": torch.backends.mps.is_available() if hasattr(torch.backends, "mps") else False,
            "current_device": str(device) if device else "cpu"
        },
        "model_service": {
            "name": model_name,
            "cache_dir": models_dir,
            "is_healthy": model_service.health_check()
        },
        "memory": memory_info,
        "uptime": uptime_info
    }
    
    # Add GPU info if CUDA is available
    if torch.cuda.is_available():
        info["device_info"]["cuda_device_count"] = torch.cuda.device_count()
        info["device_info"]["cuda_device_name"] = torch.cuda.get_device_name(0)
        
        # Try to get GPU memory info
        try:
            info["device_info"]["cuda_memory_allocated_mb"] = torch.cuda.memory_allocated() / 1024 / 1024
            info["device_info"]["cuda_memory_reserved_mb"] = torch.cuda.memory_reserved() / 1024 / 1024
        except:
            logger.info("Failed to get CUDA memory stats")
    
    logger.info(f"Returning system info with {len(info)} top-level keys")
    return info

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
    
    s1 = data.sentence1
    s2 = data.sentence2
    
    # Performance metrics
    processing_metrics = {}

    # Calculate similarity using the model service
    sim_start = time.time()
    try:
        similarity = model_service.calculate_similarity(s1, s2)
        sim_time = time.time() - sim_start
        processing_metrics["similarity_calculation_time"] = sim_time
        logger.info(f"[ID: {request_id}] Calculated similarity ({similarity:.4f}) in {sim_time:.4f}s")
    except Exception as e:
        logger.error(f"[ID: {request_id}] Similarity calculation failed: {str(e)}")
        raise
    
    # Log detailed performance metrics
    logger.info(f"[ID: {request_id}] Performance metrics: {json.dumps(processing_metrics)}")
    
    result = {
        "sentence1": s1,
        "sentence2": s2,
        "semantic_similarity": round(similarity, 4),
        "processing_time_seconds": processing_metrics["similarity_calculation_time"]
    }
    
    logger.info(f"[ID: {request_id}] Request completed successfully")
    return result
