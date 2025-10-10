"""
Simplified Model service for Sentence Transformers.
This service uses a singleton pattern to share a single model instance across all workers.
"""
import os
import sys
import time
import logging
import traceback
import torch
import numpy as np
from sentence_transformers import SentenceTransformer, util

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

class ModelService:
    """Model service using a singleton pattern to share one model instance."""
    
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
        """Initialize the model service with direct model loading."""
        if ModelService._instance is not None:
            raise RuntimeError("ModelService is a singleton class. Use get_instance() instead.")
        
        # Store parameters
        self.model_name = model_name
        self.cache_folder = cache_folder
        self.device = device
        
        # Log memory usage before loading model
        self._log_memory_usage("Before model loading")
        
        # Load the model directly
        logger.info(f"Loading model: {self.model_name} from: {self.cache_folder}")
        start_time = time.time()
        
        try:
            # Load the model
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
            
            # Log memory usage after loading model
            self._log_memory_usage("After model loading")
            
        except Exception as e:
            logger.error(f"Failed to initialize model: {e}")
            logger.error(traceback.format_exc())
            raise
    
    def encode(self, text):
        """Encode text using the model."""
        try:
            start_time = time.time()
            embedding = self.model.encode(text, convert_to_tensor=True)
            encoding_time = time.time() - start_time
            logger.info(f"Encoded text ({len(text)} chars) in {encoding_time:.4f} seconds")
            return embedding
        except Exception as e:
            logger.error(f"Error encoding text: {e}")
            raise RuntimeError(f"Error encoding text: {e}")
    
    def calculate_similarity(self, text1, text2):
        """Calculate similarity between two texts."""
        try:
            start_time = time.time()
            
            # Encode texts
            emb1 = self.model.encode(text1, convert_to_tensor=True)
            emb2 = self.model.encode(text2, convert_to_tensor=True)
            
            # Move to device if needed
            if self.device:
                try:
                    emb1 = emb1.to(self.device)
                    emb2 = emb2.to(self.device)
                except Exception as e:
                    logger.warning(f"Failed to move tensors to device: {e}")
            
            # Calculate similarity
            similarity = util.pytorch_cos_sim(emb1, emb2).item()
            
            processing_time = time.time() - start_time
            logger.info(f"Calculated similarity ({similarity:.4f}) in {processing_time:.4f} seconds")
            
            return similarity
        except Exception as e:
            logger.error(f"Error calculating similarity: {e}")
            raise RuntimeError(f"Error calculating similarity: {e}")
    
    def health_check(self):
        """Check if the model is healthy by running a simple test."""
        try:
            logger.info("Running health check")
            
            # Simple test to verify model is loaded
            if not hasattr(self, 'model') or self.model is None:
                logger.error("Health check failed: Model is not loaded")
                return False
                
            # Verify the model can encode text
            if not hasattr(self.model, 'encode'):
                logger.error("Health check failed: Model lacks encode method")
                return False
                
            # Try a simple encoding
            test_result = self.model.encode("test", convert_to_tensor=False)
            if test_result is None:
                logger.error("Health check failed: Model test returned null result")
                return False
                
            logger.info("Health check passed")
            return True
        except Exception as e:
            logger.error(f"Health check failed with exception: {e}")
            return False
    
    def shutdown(self):
        """Clean up resources."""
        logger.info("Shutting down model service")
        
        # Clear model reference to help with garbage collection
        if hasattr(self, 'model'):
            try:
                del self.model
                logger.info("Model reference removed")
            except Exception as e:
                logger.error(f"Error removing model reference: {e}")
        
        # Force garbage collection
        try:
            import gc
            gc.collect()
            logger.info("Garbage collection completed")
        except:
            pass
            
        logger.info("Model service shutdown completed")
    
    def _log_memory_usage(self, message="Current memory usage"):
        """Log the current memory usage."""
        try:
            import psutil
            process = psutil.Process(os.getpid())
            memory_info = process.memory_info()
            logger.info(f"{message} - RSS: {memory_info.rss / 1024 / 1024:.2f} MB, VMS: {memory_info.vms / 1024 / 1024:.2f} MB")
        except ImportError:
            logger.info(f"{message} - psutil not available, memory usage stats skipped")
        except Exception as e:
            logger.warning(f"Failed to log memory usage: {e}")

if __name__ == "__main__":
    # Test the model service
    import os
    
    # Get model directory
    models_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "models")
    
    # Create model service
    service = ModelService.get_instance("all-MiniLM-L6-v2", models_dir)
    
    # Test similarity calculation
    result = service.calculate_similarity("Hello world", "Hi there")
    print(f"Similarity: {result}")
    
    # Test health check
    health = service.health_check()
    print(f"Health: {health}")
    
    # Shutdown
    service.shutdown()
