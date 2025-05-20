import os
import time
import torch
import logging
from transformers import AutoModelForSequenceClassification, AutoTokenizer
from typing import Dict, Any, Union, List, Tuple

# Configure logger
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler(os.path.join(os.path.dirname(os.path.abspath(__file__)), 'equivalence_model_service.log'))
    ]
)

logger = logging.getLogger("equivalence-classifier")

class EquivalenceModelService:
    """Singleton service for semantic equivalence classification based on NLI models."""
    
    _instance = None
    
    @classmethod
    def get_instance(cls, model_name: str, cache_dir: str, device: torch.device) -> 'EquivalenceModelService':
        """Get or create the singleton instance of EquivalenceModelService."""
        if cls._instance is None:
            cls._instance = cls(model_name, cache_dir, device)
        return cls._instance
    
    def __init__(self, model_name: str, cache_dir: str, device: torch.device):
        """Initialize the EquivalenceModelService with the specified model.
        
        Args:
            model_name: Name of the Hugging Face model to use
            cache_dir: Directory to cache the model
            device: PyTorch device to run inference on (CPU, CUDA, MPS)
        """
        self.model_name = model_name
        self.cache_dir = cache_dir
        self.device = device
        self.model = None
        self.tokenizer = None
        self.id2label = None
        self.initialized = False
        
        self._initialize_model()
    
    def _initialize_model(self):
        """Load the model and tokenizer."""
        try:
            start_time = time.time()
            logger.info(f"Loading {self.model_name} tokenizer from {self.cache_dir}")
            self.tokenizer = AutoTokenizer.from_pretrained(self.model_name, cache_dir=self.cache_dir)
            
            logger.info(f"Loading {self.model_name} model from {self.cache_dir}")
            self.model = AutoModelForSequenceClassification.from_pretrained(
                self.model_name, 
                cache_dir=self.cache_dir
            )
            
            # Get the label mapping from the model config
            self.id2label = self.model.config.id2label
            
            logger.info(f"Label mapping: {self.id2label}")
            
            # Move model to the appropriate device
            self.model.to(self.device)
            self.model.eval()  # Set to evaluation mode
            
            load_time = time.time() - start_time
            logger.info(f"Model loaded successfully in {load_time:.2f} seconds")
            self.initialized = True
            
        except Exception as e:
            logger.error(f"Error initializing the model: {e}")
            self.initialized = False
            raise
    
    def health_check(self) -> bool:
        """Check if the model is properly loaded and can perform inference."""
        if not self.initialized:
            logger.error("Model not initialized")
            return False
        
        try:
            # Try a simple inference to verify the model works
            premise = "This is a test sentence."
            hypothesis = "This is a test."
            
            # Run inference
            result = self.classify_texts(premise, hypothesis)
            logger.info(f"Health check - Classification result: {result}")
            
            return True
        except Exception as e:
            logger.error(f"Health check failed: {e}")
            return False
    
    def classify_texts(self, premise: str, hypothesis: str) -> Dict[str, Any]:
        """Classify the semantic relationship between two texts.
        
        Args:
            premise: The first text
            hypothesis: The second text to compare with the premise
            
        Returns:
            A dictionary containing:
                - entailment_score: Probability that hypothesis follows from premise
                - contradiction_score: Probability that hypothesis contradicts premise
                - neutral_score: Probability that hypothesis is neutral to premise
                - predicted_label: The predicted relationship
                - is_equivalent: Boolean indicating if texts are semantically equivalent
        """
        if not self.initialized:
            raise RuntimeError("Model not initialized. Cannot perform classification.")
        
        start_time = time.time()
        
        try:
            # Prepare input
            encoded_input = self.tokenizer(
                premise, 
                hypothesis, 
                padding=True, 
                truncation=True, 
                return_tensors="pt"
            ).to(self.device)
            
            # Perform inference
            with torch.no_grad():
                outputs = self.model(**encoded_input)
                logits = outputs.logits
                probs = torch.nn.functional.softmax(logits, dim=1)
            
            # Convert to numpy for easier handling
            probs = probs.cpu().numpy()[0]
            predicted_class_id = logits.argmax().item()
            
            # Get the predicted label
            predicted_label = self.id2label[predicted_class_id]
            
            # Prepare the result dictionary
            result = {}
            for i, label in self.id2label.items():
                result[f"{label}_score"] = float(probs[i])
            
            result["predicted_label"] = predicted_label
            
            # Determine if the texts are semantically equivalent
            # For RoBERTa NLI models, "entailment" usually indicates equivalence or logical implication
            result["is_equivalent"] = predicted_label == "entailment"
            
            process_time = time.time() - start_time
            logger.info(f"Classification completed in {process_time:.4f} seconds")
            
            return result
            
        except Exception as e:
            process_time = time.time() - start_time
            logger.error(f"Classification failed after {process_time:.4f} seconds: {e}")
            raise
    
    def shutdown(self):
        """Clean up resources."""
        try:
            logger.info("Shutting down model service")
            if self.model is not None:
                # Force cleanup of CUDA memory if applicable
                if self.device.type == 'cuda' and hasattr(torch.cuda, 'empty_cache'):
                    torch.cuda.empty_cache()
            
            logger.info("Model service shutdown complete")
        except Exception as e:
            logger.error(f"Error during model shutdown: {e}")
