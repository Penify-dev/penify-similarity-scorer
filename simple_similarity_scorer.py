"""
Simple Similarity Scorer - A streamlined version for calculating semantic similarity.
This removes the overcomplicated singleton patterns, excessive logging, and memory tracking.
"""
import torch
from sentence_transformers import SentenceTransformer, util
import os
import time
from typing import Optional, Union, List

class SimpleSimilarityScorer:
    """A simple, straightforward similarity scorer."""
    
    def __init__(self, model_name: str = "all-MiniLM-L6-v2", cache_folder: Optional[str] = None):
        """
        Initialize the similarity scorer.
        
        Args:
            model_name: Name of the sentence transformer model
            cache_folder: Optional cache folder for model storage
        """
        self.model_name = model_name
        self.cache_folder = cache_folder or "./models"
        self.device = self._get_best_device()
        self.model = None
        self._load_model()
    
    def _get_best_device(self) -> torch.device:
        """Get the best available device."""
        if torch.cuda.is_available():
            return torch.device("cuda")
        elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            return torch.device("mps")
        else:
            return torch.device("cpu")
    
    def _load_model(self):
        """Load the sentence transformer model."""
        try:
            os.makedirs(self.cache_folder, exist_ok=True)
            self.model = SentenceTransformer(self.model_name, cache_folder=self.cache_folder)
            if self.device.type != "cpu":
                self.model.to(self.device)
        except Exception as e:
            raise RuntimeError(f"Failed to load model {self.model_name}: {e}")
    
    def calculate_similarity(self, text1: str, text2: str) -> float:
        """
        Calculate cosine similarity between two texts.
        
        Args:
            text1: First text
            text2: Second text
            
        Returns:
            Similarity score between 0 and 1
        """
        if not self.model:
            raise RuntimeError("Model not loaded")
        
        try:
            # Encode both texts
            embedding1 = self.model.encode(text1, convert_to_tensor=True)
            embedding2 = self.model.encode(text2, convert_to_tensor=True)
            
            # Calculate cosine similarity
            similarity = util.pytorch_cos_sim(embedding1, embedding2).item()
            
            return max(0.0, min(1.0, similarity))  # Ensure score is between 0 and 1
        
        except Exception as e:
            raise RuntimeError(f"Error calculating similarity: {e}")
    
    def calculate_batch_similarity(self, texts1: List[str], texts2: List[str]) -> List[float]:
        """
        Calculate similarities for multiple text pairs efficiently.
        
        Args:
            texts1: List of first texts
            texts2: List of second texts
            
        Returns:
            List of similarity scores
        """
        if len(texts1) != len(texts2):
            raise ValueError("texts1 and texts2 must have the same length")
        
        if not self.model:
            raise RuntimeError("Model not loaded")
        
        try:
            # Batch encode for efficiency
            embeddings1 = self.model.encode(texts1, convert_to_tensor=True)
            embeddings2 = self.model.encode(texts2, convert_to_tensor=True)
            
            # Calculate similarities
            similarities = []
            for i in range(len(texts1)):
                sim = util.pytorch_cos_sim(embeddings1[i], embeddings2[i]).item()
                similarities.append(max(0.0, min(1.0, sim)))
            
            return similarities
        
        except Exception as e:
            raise RuntimeError(f"Error calculating batch similarities: {e}")
    
    def encode(self, text: Union[str, List[str]]) -> torch.Tensor:
        """
        Encode text(s) into embeddings.
        
        Args:
            text: Single text or list of texts
            
        Returns:
            Tensor containing embeddings
        """
        if not self.model:
            raise RuntimeError("Model not loaded")
        
        try:
            return self.model.encode(text, convert_to_tensor=True)
        except Exception as e:
            raise RuntimeError(f"Error encoding text: {e}")
    
    def is_healthy(self) -> bool:
        """Simple health check."""
        try:
            if not self.model:
                return False
            # Quick test
            self.calculate_similarity("test", "test")
            return True
        except:
            return False
    
    @property
    def info(self) -> dict:
        """Get basic info about the scorer."""
        return {
            "model_name": self.model_name,
            "device": str(self.device),
            "cache_folder": self.cache_folder,
            "is_loaded": self.model is not None
        }

# Convenience function for quick similarity calculation
def calculate_similarity(text1: str, text2: str, model_name: str = "all-MiniLM-L6-v2") -> float:
    """
    Quick function to calculate similarity without creating a class instance.
    Note: This will reload the model each time, so use SimpleSimilarityScorer for multiple calls.
    """
    scorer = SimpleSimilarityScorer(model_name)
    return scorer.calculate_similarity(text1, text2)

if __name__ == "__main__":
    # Simple test
    scorer = SimpleSimilarityScorer()
    
    print(f"Scorer info: {scorer.info}")
    print(f"Health check: {scorer.is_healthy()}")
    
    # Test similarity
    text1 = "The cat sits on the mat"
    text2 = "A cat is sitting on a mat"
    
    start_time = time.time()
    similarity = scorer.calculate_similarity(text1, text2)
    elapsed = time.time() - start_time
    
    print(f"\nText 1: {text1}")
    print(f"Text 2: {text2}")
    print(f"Similarity: {similarity:.4f}")
    print(f"Time: {elapsed:.4f}s")
    
    # Test batch processing
    texts1 = ["Hello world", "Good morning", "Machine learning is great"]
    texts2 = ["Hi there", "Good evening", "AI is amazing"]
    
    start_time = time.time()
    similarities = scorer.calculate_batch_similarity(texts1, texts2)
    elapsed = time.time() - start_time
    
    print(f"\nBatch similarities:")
    for i, sim in enumerate(similarities):
        print(f"  '{texts1[i]}' vs '{texts2[i]}': {sim:.4f}")
    print(f"Batch time: {elapsed:.4f}s")
