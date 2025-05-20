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
# Import the new Equivalence model service
from equivalence_model_service import EquivalenceModelService

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
# Get equivalence model name from environment variable or use the specified one
equivalence_model_name = os.environ.get("EQUIVALENCE_MODEL_NAME", "ynie/roberta-large-snli_mnli_fever_anli_R1_R2_R3-nli")
logger.info(f"Loading models:\n - Similarity model: {model_name}\n - Equivalence model: {equivalence_model_name}")
logger.info(f"Using cache directory: {models_dir}")

# Initialize the model service
start_time = time.time()
try:
    # Get the model service singleton - only creates one model instance
    model_service = ModelService.get_instance(model_name, models_dir, device)
    load_time = time.time() - start_time
    logger.info(f"Model service initialized in {load_time:.2f} seconds")
    
    # Initialize the equivalence model service
    start_time = time.time()
    equivalence_model_service = EquivalenceModelService.get_instance(equivalence_model_name, models_dir, device)
    load_time = time.time() - start_time
    logger.info(f"Equivalence model service initialized in {load_time:.2f} seconds")
    
    # Register shutdown function to clean up the model service
    def shutdown_model_service():
        """Shuts down the model and equivalence model services."""
        logger.info("Shutting down model services")
        try:
            model_service.shutdown()
            equivalence_model_service.shutdown()
            logger.info("Model services shutdown completed successfully")
        except Exception as e:
            logger.error(f"Error during model services shutdown: {e}")
    
    atexit.register(shutdown_model_service)
    
except Exception as e:
    logger.error(f"Failed to initialize model services: {e}")
    raise

# FastAPI lifespan context manager
@asynccontextmanager
async def lifespan(app: FastAPI):
    # Startup: verify model is properly loaded
    """Manages application startup and shutdown, including model health checks and
    verification.
    
    This context manager performs several tasks during the application's lifecycle:
    1. Verifies the health of the similarity and equivalence model services. 2.
    Tests the functionality of these models by performing similarity calculations
    and text classification. 3. Logs any errors or warnings encountered during
    these checks. 4. Optionally logs memory usage if the `psutil` library is
    available. 5. Yields control back to FastAPI for normal operation. 6. Handles
    application shutdown logging.
    
    Args:
        app (FastAPI): The FastAPI application instance.
    """
    logger.info("Application startup - Verifying model services are properly loaded...")
    try:
        # Check similarity model service health
        health_check_start = time.time()
        is_healthy = model_service.health_check()
        health_check_time = time.time() - health_check_start
        
        if not is_healthy:
            logger.error(f"Similarity model service health check failed in {health_check_time:.4f} seconds")
            # Don't crash the app, but log that we're continuing with degraded functionality
            logger.warning("Application will continue but similarity model functionality may be limited")
        else:    
            logger.info(f"Similarity model service health check passed in {health_check_time:.4f} seconds")
            
            # Test similarity calculation
            s1 = "Hello world"
            s2 = "Hi there"
            sim_start = time.time()
            try:
                similarity = model_service.calculate_similarity(s1, s2)
                sim_time = time.time() - sim_start
                
                logger.info(f"Similarity model verification successful. Similarity: {similarity:.4f}")
                logger.info(f"Similarity calculation time: {sim_time:.4f} seconds")
            except Exception as sim_error:
                logger.error(f"Similarity calculation test failed: {sim_error}")
                logger.warning("Application will continue but similarity model functionality may be limited")
        
        # Check equivalence model service health
        health_check_start = time.time()
        is_healthy_eq = equivalence_model_service.health_check()
        health_check_time = time.time() - health_check_start
        
        if not is_healthy_eq:
            logger.error(f"Equivalence model service health check failed in {health_check_time:.4f} seconds")
            logger.warning("Application will continue but equivalence model functionality may be limited")
        else:    
            logger.info(f"Equivalence model service health check passed in {health_check_time:.4f} seconds")
            
            # Test equivalence classification
            p = "The cat is on the mat"
            h = "There is a cat sitting on a mat"
            eq_start = time.time()
            try:
                result = equivalence_model_service.classify_texts(p, h)
                eq_time = time.time() - eq_start
                
                logger.info(f"Equivalence model verification successful. Result: {result['predicted_label']}")
                logger.info(f"Equivalence calculation time: {eq_time:.4f} seconds")
            except Exception as eq_error:
                logger.error(f"Equivalence classification test failed: {eq_error}")
                logger.warning("Application will continue but equivalence model functionality may be limited")
            
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
    """Endpoint to perform a comprehensive health check of the application.
    
    This function checks the overall health status of the application by verifying
    the health of the similarity model service, equivalence model service, and
    monitoring memory and CPU usage. It logs detailed information about each check
    and updates the health status accordingly. If any service fails its health
    check, the overall status is marked as degraded or warning.
    """
    logger.info("Health check endpoint called")
    
    health_status = {
        "status": "healthy",
        "timestamp": datetime.now().isoformat(),
        "version": "1.0.0"
    }
    
    # Check similarity model service status
    try:
        logger.info("Health check - Testing similarity model service")
        health_check_start = time.time()
        similarity_service_healthy = model_service.health_check()
        health_check_time = time.time() - health_check_start
        
        logger.info(f"Health check - Similarity model service: {similarity_service_healthy}, time: {health_check_time:.4f}s")
        
        if similarity_service_healthy:
            health_status["similarity_model_service"] = {
                "status": "healthy",
                "name": model_name,
                "response_time": health_check_time
            }
        else:
            health_status["status"] = "degraded"
            health_status["similarity_model_service"] = {
                "status": "error",
                "error": "Similarity model service health check failed",
                "response_time": health_check_time
            }
            logger.error("Health check - Similarity model service health check failed")
    except Exception as e:
        health_status["status"] = "degraded"
        health_status["similarity_model_service"] = {
            "status": "error",
            "error": str(e)
        }
        logger.error(f"Health check - Similarity model service test failed: {e}")
    
    # Check equivalence model service status
    try:
        logger.info("Health check - Testing equivalence model service")
        health_check_start = time.time()
        equivalence_service_healthy = equivalence_model_service.health_check()
        health_check_time = time.time() - health_check_start
        
        logger.info(f"Health check - Equivalence model service: {equivalence_service_healthy}, time: {health_check_time:.4f}s")
        
        if equivalence_service_healthy:
            health_status["equivalence_model_service"] = {
                "status": "healthy",
                "name": equivalence_model_name,
                "response_time": health_check_time
            }
        else:
            health_status["status"] = "degraded"
            health_status["equivalence_model_service"] = {
                "status": "error",
                "error": "Equivalence model service health check failed",
                "response_time": health_check_time
            }
            logger.error("Health check - Equivalence model service health check failed")
    except Exception as e:
        health_status["status"] = "degraded"
        health_status["equivalence_model_service"] = {
            "status": "error",
            "error": str(e)
        }
        logger.error(f"Health check - Equivalence model service test failed: {e}")
    
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

# Request body structure for equivalence
class EquivalenceRequest(BaseModel):
    premise: str
    hypothesis: str

# Route to check semantic equivalence
@app.post("/classify-equivalence")
async def classify_equivalence(data: EquivalenceRequest):
    """Classify semantic equivalence between premise and hypothesis."""
    request_id = f"req-{int(time.time() * 1000)}"
    logger.info(f"Equivalence endpoint called [ID: {request_id}]")
    logger.info(f"Premise ({len(data.premise)} chars): {data.premise[:50]}...")
    logger.info(f"Hypothesis ({len(data.hypothesis)} chars): {data.hypothesis[:50]}...")
    
    # Calculate equivalence
    eq_start = time.time()
    try:
        result = equivalence_model_service.classify_texts(data.premise, data.hypothesis)
        eq_time = time.time() - eq_start
        logger.info(f"[ID: {request_id}] Classified equivalence ({result['predicted_label']}) in {eq_time:.4f}s")
    except Exception as e:
        logger.error(f"[ID: {request_id}] Equivalence classification failed: {str(e)}")
        raise
    
    # Add the original texts and processing time to the result
    result["premise"] = data.premise
    result["hypothesis"] = data.hypothesis
    result["processing_time_seconds"] = eq_time
    
    logger.info(f"[ID: {request_id}] Request completed successfully")
    return result
