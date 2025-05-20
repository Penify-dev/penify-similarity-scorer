"""
Model service for Sentence Transformers that uses multiprocessing to share a single model instance.
This service runs in a separate process and communicates with the web server workers via a queue.
"""
import multiprocessing
import threading
import queue
import time
import logging
import traceback
import os
import sys
import numpy as np
from sentence_transformers import SentenceTransformer, util
import torch

# Configure proper start method for multiprocessing
# This is crucial to prevent semaphore leaks
if sys.platform == 'darwin':  # macOS
    # Force 'spawn' on macOS to avoid semaphore leaks
    multiprocessing.set_start_method('spawn', force=True)
elif sys.platform.startswith('win'):  # Windows
    # Windows already uses 'spawn'
    pass
else:  # Linux and others
    # Use 'spawn' on Linux as well for consistency
    try:
        multiprocessing.set_start_method('spawn', force=True)
    except RuntimeError:
        # Already set, which is fine
        pass

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler(os.path.join(os.path.dirname(os.path.abspath(__file__)), 'model_service.log'))
    ]
)
logger = logging.getLogger("model-service")

class ModelRequest:
    """Represents a request to the model service."""
    def __init__(self, id, operation, data):
        self.id = id
        self.operation = operation  # encode, similarity, etc.
        self.data = data  # text for encode, (text1, text2) for similarity

class ModelResponse:
    """Represents a response from the model service."""
    def __init__(self, request_id, result=None, error=None, processing_time=None):
        self.request_id = request_id
        self.result = result
        self.error = error
        self.processing_time = processing_time

class ModelServiceProcess(multiprocessing.Process):
    """Process that loads and serves the model."""
    
    def __init__(self, request_queue, response_queue, model_name, cache_folder, device=None):
        super().__init__()
        self.request_queue = request_queue
        self.response_queue = response_queue
        self.model_name = model_name
        self.cache_folder = cache_folder
        self.device = device
        self.daemon = True  # Process will terminate when main process exits
        
    def run(self):
        """Load the model and start processing requests."""
        logger.info(f"Starting model service process (PID: {os.getpid()})")
        logger.info(f"Loading model: {self.model_name} from: {self.cache_folder}")
        
        # Log initial memory usage before model loading
        try:
            import psutil
            process = psutil.Process(os.getpid())
            memory_info = process.memory_info()
            logger.info(f"Memory before model loading - RSS: {memory_info.rss / 1024 / 1024:.2f} MB, VMS: {memory_info.vms / 1024 / 1024:.2f} MB")
        except:
            pass
        
        # Load the model
        try:
            start_time = time.time()
            self.model = SentenceTransformer(self.model_name, cache_folder=self.cache_folder)
            load_time = time.time() - start_time
            logger.info(f"Model loaded successfully in {load_time:.2f} seconds")
            
            # Log memory usage after model loading
            try:
                memory_info = process.memory_info()
                logger.info(f"Memory after model loading - RSS: {memory_info.rss / 1024 / 1024:.2f} MB, VMS: {memory_info.vms / 1024 / 1024:.2f} MB")
            except:
                pass
            
            # Move model to device if specified
            if self.device:
                try:
                    self.model.to(self.device)
                    logger.info(f"Model moved to device: {self.device}")
                except Exception as e:
                    logger.error(f"Failed to move model to device {self.device}: {e}")
                    logger.error(traceback.format_exc())
            
            # Process requests
            self._process_requests()
            
        except Exception as e:
            logger.error(f"Failed to initialize model service: {e}")
            logger.error(traceback.format_exc())
            
    def _process_requests(self):
        """Process requests from the queue."""
        logger.info("Model service ready to process requests")
        
        while True:
            try:
                # Get request from queue
                request = self.request_queue.get()
                start_time = time.time()
                
                # Log request
                logger.info(f"Processing request: {request.id} - Operation: {request.operation}")
                
                # Process request
                if request.operation == "encode":
                    # Encode text
                    text = request.data
                    embedding = self.model.encode(text, convert_to_tensor=True)
                    
                    # Convert embedding to list for serialization
                    embedding_list = embedding.cpu().numpy().tolist() if hasattr(embedding, 'cpu') else embedding.numpy().tolist()
                    
                    # Create response
                    response = ModelResponse(
                        request_id=request.id,
                        result=embedding_list,
                        processing_time=time.time() - start_time
                    )
                    
                elif request.operation == "similarity":
                    # Calculate similarity between two texts
                    text1, text2 = request.data
                    
                    # Encode texts
                    emb1 = self.model.encode(text1, convert_to_tensor=True)
                    emb2 = self.model.encode(text2, convert_to_tensor=True)
                    
                    # Move to device if available
                    if self.device:
                        try:
                            emb1 = emb1.to(self.device)
                            emb2 = emb2.to(self.device)
                        except Exception as e:
                            logger.warning(f"Failed to move tensors to device: {e}")
                    
                    # Calculate similarity
                    similarity = util.pytorch_cos_sim(emb1, emb2).item()
                    
                    # Create response
                    response = ModelResponse(
                        request_id=request.id,
                        result=similarity,
                        processing_time=time.time() - start_time
                    )
                    
                elif request.operation == "health_check":
                    # Health check - add more detailed logging
                    logger.info(f"Processing health check request {request.id}")
                    
                    # Quick test if model is accessible
                    try:
                        # Simple test to verify model is loaded
                        if hasattr(self, 'model') and self.model is not None:
                            logger.info(f"Health check {request.id}: model is loaded")
                            # Create a small test to ensure model is actually responding
                            if hasattr(self.model, 'encode'):
                                # Very simple test with a tiny input
                                test_result = self.model.encode("test", convert_to_tensor=False)
                                if test_result is not None:
                                    logger.info(f"Health check {request.id}: successful model test")
                                    response = ModelResponse(
                                        request_id=request.id,
                                        result="ok",
                                        processing_time=time.time() - start_time
                                    )
                                else:
                                    logger.error(f"Health check {request.id}: model test failed with null result")
                                    response = ModelResponse(
                                        request_id=request.id,
                                        error="Model test returned null result",
                                        processing_time=time.time() - start_time
                                    )
                            else:
                                # Model doesn't have encode method
                                logger.info(f"Health check {request.id}: model lacks encode method")
                                response = ModelResponse(
                                    request_id=request.id,
                                    result="ok",  # Still consider this ok, just note it
                                    processing_time=time.time() - start_time
                                )
                        else:
                            logger.error(f"Health check {request.id}: model is not loaded")
                            response = ModelResponse(
                                request_id=request.id,
                                error="Model is not loaded",
                                processing_time=time.time() - start_time
                            )
                    except Exception as e:
                        logger.error(f"Health check {request.id} failed: {e}")
                        logger.error(traceback.format_exc())
                        response = ModelResponse(
                            request_id=request.id,
                            error=f"Health check failed: {str(e)}",
                            processing_time=time.time() - start_time
                        )
                    
                    logger.info(f"Health check {request.id} result: {response.result if not response.error else f'error: {response.error}'}")
                    
                    # Prioritize sending health check responses by putting them at the front of the queue if possible
                    try:
                        # First try to send directly (non-blocking, might not work in all queue implementations)
                        self.response_queue.put(response, block=False)
                    except (queue.Full, AttributeError, TypeError):
                        # If that fails, just use the normal put
                        self.response_queue.put(response)
                    
                    # Continue processing other requests
                    continue
                    
                elif request.operation == "shutdown":
                    # Shutdown service
                    logger.info("Received shutdown request. Terminating model service.")
                    response = ModelResponse(
                        request_id=request.id,
                        result="shutdown",
                        processing_time=time.time() - start_time
                    )
                    self.response_queue.put(response)
                    break
                    
                else:
                    # Unknown operation
                    logger.error(f"Unknown operation: {request.operation}")
                    response = ModelResponse(
                        request_id=request.id,
                        error=f"Unknown operation: {request.operation}",
                        processing_time=time.time() - start_time
                    )
                
                # Send response
                self.response_queue.put(response)
                logger.info(f"Request {request.id} processed in {response.processing_time:.4f} seconds")
                
            except Exception as e:
                logger.error(f"Error processing request: {e}")
                logger.error(traceback.format_exc())
                
                # Try to send error response if we have a request
                try:
                    if 'request' in locals():
                        error_response = ModelResponse(
                            request_id=request.id,
                            error=str(e),
                            processing_time=time.time() - start_time if 'start_time' in locals() else None
                        )
                        self.response_queue.put(error_response)
                except:
                    pass

class ModelService:
    """Client interface to the model service."""
    
    _instance = None
    
    @classmethod
    def get_instance(cls, model_name=None, cache_folder=None, device=None):
        """Get the singleton instance of the model service."""
        if cls._instance is None:
            logger.info(f"Creating new ModelService instance with model: {model_name}")
            cls._instance = ModelService(model_name, cache_folder, device)
        else:
            logger.info("Reusing existing ModelService instance")
            # Log memory usage of the existing instance
            cls._instance._log_memory_usage("Existing model service memory usage")
        return cls._instance
    
    def __init__(self, model_name, cache_folder, device=None):
        """Initialize the model service."""
        if ModelService._instance is not None:
            raise RuntimeError("ModelService is a singleton class. Use get_instance() instead.")
            
        self.model_name = model_name
        self.cache_folder = cache_folder
        self.device = device
        
        # Log memory usage before loading model
        self._log_memory_usage("Before model service initialization")
        
        # Create queues for communication
        # Use a context variable to track created resources
        self._resources = []
        
        # Create request queue with proper context
        self.request_queue = multiprocessing.Queue()
        self._resources.append(("request_queue", self.request_queue))
        
        # Create response queue with proper context
        self.response_queue = multiprocessing.Queue()
        self._resources.append(("response_queue", self.response_queue))
        
        # Create and start the model service process
        self.process = ModelServiceProcess(
            self.request_queue, 
            self.response_queue, 
            self.model_name, 
            self.cache_folder, 
            self.device
        )
        self._resources.append(("process", self.process))
        self.process.start()
        
        # Storage for responses
        self.responses = {}
        
        # Create and start the response listener thread
        self.response_thread = threading.Thread(target=self._response_listener)
        self.response_thread.daemon = True
        self.response_thread.start()
        
        # Log memory usage after starting model service
        self._log_memory_usage("After model service initialization")
        
        logger.info(f"ModelService initialized with model {model_name} (Service PID: {self.process.pid}, Main PID: {os.getpid()})")
        
    def _response_listener(self):
        """Listen for responses from the model service."""
        logger.info("Response listener thread started")
        
        while True:
            try:
                # Get response from queue with a timeout
                try:
                    response = self.response_queue.get(timeout=0.5)  # Use a shorter timeout for more responsiveness
                    
                    # Safety checks on the response object
                    if not hasattr(response, 'request_id') or not response.request_id:
                        logger.warning(f"Received response with invalid or missing request_id: {response}")
                        continue
                        
                    logger.info(f"Received response for request: {response.request_id}")
                    
                    # Special handling for health checks to ensure they're prioritized
                    is_health_check = response.request_id and response.request_id.startswith('health_check-')
                    
                    # Extra logging for health check responses
                    if is_health_check:
                        logger.info(f"Processing health check response: {response.request_id}, " 
                                   f"Result: {response.result if hasattr(response, 'result') else 'None'}, "
                                   f"Error: {response.error if hasattr(response, 'error') else 'None'}")
                    
                    # Store response immediately
                    self.responses[response.request_id] = response
                    
                except queue.Empty:
                    # Just continue if no response within timeout
                    continue
                    
            except Exception as e:
                logger.error(f"Error in response listener: {e}")
                logger.error(traceback.format_exc())
                # Brief pause to avoid consuming 100% CPU in case of repeated errors
                time.sleep(0.1)
    
    def encode(self, text, timeout=30):
        """Encode text using the model."""
        # Generate request ID
        request_id = f"encode-{time.time()}-{id(text)}"
        
        # Create and send request
        request = ModelRequest(request_id, "encode", text)
        self.request_queue.put(request)
        
        # Wait for response
        start_time = time.time()
        while request_id not in self.responses:
            if time.time() - start_time > timeout:
                raise TimeoutError(f"Request timed out after {timeout} seconds")
            time.sleep(0.01)
        
        # Get and remove response
        response = self.responses.pop(request_id)
        
        # Check for error
        if response.error:
            raise RuntimeError(f"Error encoding text: {response.error}")
        
        # Return result as torch tensor
        import numpy as np
        embedding_np = np.array(response.result)
        return torch.tensor(embedding_np)
    
    def calculate_similarity(self, text1, text2, timeout=30):
        """Calculate similarity between two texts."""
        # Generate request ID
        request_id = f"similarity-{time.time()}-{id(text1)}-{id(text2)}"
        
        # Create and send request
        request = ModelRequest(request_id, "similarity", (text1, text2))
        self.request_queue.put(request)
        
        # Wait for response
        start_time = time.time()
        while request_id not in self.responses:
            if time.time() - start_time > timeout:
                raise TimeoutError(f"Request timed out after {timeout} seconds")
            time.sleep(0.01)
        
        # Get and remove response
        response = self.responses.pop(request_id)
        
        # Check for error
        if response.error:
            raise RuntimeError(f"Error calculating similarity: {response.error}")
        
        # Return result
        return response.result
    
    def health_check(self, timeout=5):
        """Check if the model service is healthy."""
        # Generate request ID with a unique identifier - use a simpler format to avoid any serialization issues
        request_id = f"health_check-{time.time():.6f}"
        
        # Create and send request
        request = ModelRequest(request_id, "health_check", None)
        
        try:
            # First check if process is running - using a safer method
            process_alive = False
            if hasattr(self, 'process') and self.process.pid:
                try:
                    # Check if process is alive in a way that works across processes
                    import psutil
                    process_alive = psutil.pid_exists(self.process.pid)
                    if not process_alive:
                        logger.error(f"Health check failed: Model service process (PID: {self.process.pid}) is not running")
                        return False
                except (ImportError, Exception) as e:
                    # If psutil not available, just assume process is running and continue
                    logger.warning(f"Could not verify process status: {e}. Continuing with health check.")
                    process_alive = True
            else:
                logger.error("Health check failed: No process reference available")
                return False
                
            # Log before sending the request
            logger.info(f"Sending health check request: {request_id}")
            
            # Clear any existing response with same prefix to avoid confusion (defensive)
            for key in list(self.responses.keys()):
                if key.startswith('health_check-'):
                    logger.info(f"Clearing old health check response: {key}")
                    self.responses.pop(key)
            
            # Send the request
            self.request_queue.put(request)
            
            # Wait for response with more frequent checks
            start_time = time.time()
            while request_id not in self.responses:
                if time.time() - start_time > timeout:
                    logger.error(f"Health check timed out after {timeout} seconds for request {request_id}")
                    return False
                # Use shorter sleep interval for more responsive checks
                time.sleep(0.005)
            
            # Get and remove response
            response = self.responses.pop(request_id)
            logger.info(f"Received health check response for {request_id}: {response.result if hasattr(response, 'result') else 'None'}")
            
            # Check for error
            if hasattr(response, 'error') and response.error:
                logger.error(f"Health check response contained error: {response.error}")
                return False
            
            # Return health status
            return hasattr(response, 'result') and response.result == "ok"
        except Exception as e:
            logger.error(f"Exception in health check: {e}")
            logger.error(traceback.format_exc())
            return False
    
    def shutdown(self):
        """Shutdown the model service."""
        logger.info("Starting model service shutdown...")
        
        try:
            # Generate request ID
            request_id = f"shutdown-{time.time()}"
            
            # Create and send request
            request = ModelRequest(request_id, "shutdown", None)
            self.request_queue.put(request)
            
            # Wait for response
            start_time = time.time()
            while request_id not in self.responses and time.time() - start_time < 5:
                time.sleep(0.01)
            
            # Wait for process to terminate
            if self.process.is_alive():
                logger.info(f"Joining process {self.process.pid}...")
                self.process.join(timeout=5)
                
                # If it's still alive, terminate it
                if self.process.is_alive():
                    logger.warning(f"Process {self.process.pid} didn't terminate, forcing termination...")
                    self.process.terminate()
                    self.process.join(timeout=2)
                    
                    # Last resort - kill
                    if self.process.is_alive():
                        logger.warning(f"Process {self.process.pid} still alive after terminate, killing...")
                        try:
                            import signal
                            os.kill(self.process.pid, signal.SIGKILL)
                        except Exception as e:
                            logger.error(f"Failed to kill process: {e}")
            
            # Clean up multiprocessing resources
            self._cleanup_resources()
            
            logger.info("Model service shutdown complete")
        except Exception as e:
            logger.error(f"Error during shutdown: {e}")
            logger.error(traceback.format_exc())
    
    def _cleanup_resources(self):
        """Clean up multiprocessing resources to prevent leaks."""
        try:
            # Close the queues to release resources
            if hasattr(self, '_resources') and self._resources:
                for name, resource in self._resources:
                    logger.info(f"Cleaning up resource: {name}")
                    if name == "request_queue" or name == "response_queue":
                        try:
                            # First cancel any pending operations
                            while not resource.empty():
                                try:
                                    resource.get_nowait()
                                except:
                                    break
                            # Now close and join the queue
                            resource.close()
                            resource.join_thread()
                            logger.info(f"Successfully closed and joined {name}")
                        except Exception as e:
                            logger.warning(f"Error cleaning up {name}: {e}")
                
                # Clear resources list
                self._resources.clear()
            else:
                # Fallback if resources list not available
                logger.info("Closing request queue...")
                if hasattr(self, 'request_queue'):
                    try:
                        self.request_queue.close()
                        self.request_queue.join_thread()  # Wait for background thread to exit
                    except Exception as e:
                        logger.warning(f"Error closing request queue: {e}")
                
                logger.info("Closing response queue...")
                if hasattr(self, 'response_queue'):
                    try:
                        self.response_queue.close()
                        self.response_queue.join_thread()  # Wait for background thread to exit
                    except Exception as e:
                        logger.warning(f"Error closing response queue: {e}")
            
            # Clear responses dictionary
            if hasattr(self, 'responses'):
                self.responses.clear()
            
            logger.info("Multiprocessing resources cleaned up")
            
            # Force garbage collection to help clean up any remaining resources
            try:
                import gc
                gc.collect()
                logger.info("Garbage collection completed")
            except:
                pass
                
        except Exception as e:
            logger.error(f"Error cleaning up resources: {e}")
            logger.error(traceback.format_exc())
        
    def _log_memory_usage(self, message="Current memory usage"):
        """Log the current memory usage."""
        try:
            import psutil
            process = psutil.Process(os.getpid())
            memory_info = process.memory_info()
            logger.info(f"{message} - RSS: {memory_info.rss / 1024 / 1024:.2f} MB, VMS: {memory_info.vms / 1024 / 1024:.2f} MB")
            
            # Try to get model service process memory if available
            try:
                if hasattr(self, 'process') and self.process.pid:
                    model_process = psutil.Process(self.process.pid)
                    model_memory = model_process.memory_info()
                    logger.info(f"Model service process memory - RSS: {model_memory.rss / 1024 / 1024:.2f} MB, VMS: {model_memory.vms / 1024 / 1024:.2f} MB")
            except:
                pass
                
        except ImportError:
            logger.info(f"{message} - psutil not available, memory usage stats skipped")
        except Exception as e:
            logger.warning(f"Failed to log memory usage: {e}")

if __name__ == "__main__":
    # Enable proper multiprocessing handling on macOS
    multiprocessing.set_start_method('spawn', force=True)
    
    # Test the model service
    import os
    import numpy as np
    
    # Get model directory
    models_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "models")
    
    # Create model service
    service = ModelService.get_instance("all-MiniLM-L6-v2", models_dir)
    
    # Test encoding
    result = service.calculate_similarity("Hello world", "Hi there")
    print(f"Similarity: {result}")
    
    # Test health check
    health = service.health_check()
    print(f"Health: {health}")
    
    # Shutdown
    service.shutdown()
