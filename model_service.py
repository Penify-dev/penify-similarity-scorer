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
from sentence_transformers import SentenceTransformer, util
import torch

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
        
        # Load the model
        try:
            start_time = time.time()
            self.model = SentenceTransformer(self.model_name, cache_folder=self.cache_folder)
            load_time = time.time() - start_time
            logger.info(f"Model loaded successfully in {load_time:.2f} seconds")
            
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
                    # Health check
                    response = ModelResponse(
                        request_id=request.id,
                        result="ok",
                        processing_time=time.time() - start_time
                    )
                    
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
            cls._instance = ModelService(model_name, cache_folder, device)
        return cls._instance
    
    def __init__(self, model_name, cache_folder, device=None):
        """Initialize the model service."""
        if ModelService._instance is not None:
            raise RuntimeError("ModelService is a singleton class. Use get_instance() instead.")
            
        self.model_name = model_name
        self.cache_folder = cache_folder
        self.device = device
        
        # Create queues for communication
        self.request_queue = multiprocessing.Queue()
        self.response_queue = multiprocessing.Queue()
        
        # Create and start the model service process
        self.process = ModelServiceProcess(
            self.request_queue, 
            self.response_queue, 
            self.model_name, 
            self.cache_folder, 
            self.device
        )
        self.process.start()
        
        # Storage for responses
        self.responses = {}
        
        # Create and start the response listener thread
        self.response_thread = threading.Thread(target=self._response_listener)
        self.response_thread.daemon = True
        self.response_thread.start()
        
        logger.info(f"ModelService initialized with model {model_name}")
        
    def _response_listener(self):
        """Listen for responses from the model service."""
        logger.info("Response listener thread started")
        
        while True:
            try:
                # Get response from queue
                response = self.response_queue.get()
                
                # Store response
                self.responses[response.request_id] = response
                
            except Exception as e:
                logger.error(f"Error in response listener: {e}")
                logger.error(traceback.format_exc())
    
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
        # Generate request ID
        request_id = f"health_check-{time.time()}"
        
        # Create and send request
        request = ModelRequest(request_id, "health_check", None)
        self.request_queue.put(request)
        
        # Wait for response
        start_time = time.time()
        while request_id not in self.responses:
            if time.time() - start_time > timeout:
                return False
            time.sleep(0.01)
        
        # Get and remove response
        response = self.responses.pop(request_id)
        
        # Check for error
        if response.error:
            return False
        
        # Return health status
        return response.result == "ok"
    
    def shutdown(self):
        """Shutdown the model service."""
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
        self.process.join(timeout=5)
        
        logger.info("Model service shutdown complete")

# Example usage
if __name__ == "__main__":
    # Test the model service
    import os
    
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
