#!/usr/bin/env python3
"""
Simple test to demonstrate the critical bug in ModelService.
This will show that the model is never actually loaded.
"""
import os
import sys
import traceback

# Add current directory to path so we can import the modules
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from model_service import ModelService

def test_model_loading_bug():
    """Test to demonstrate the model loading bug."""
    print("=" * 60)
    print("TESTING MODEL SERVICE BUG")
    print("=" * 60)
    
    # Get model directory
    models_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "models")
    print(f"Models directory: {models_dir}")
    
    try:
        print("\n1. Creating ModelService instance...")
        service = ModelService.get_instance("all-MiniLM-L6-v2", models_dir)
        print(f"   Service created: {service}")
        print(f"   Model attribute: {service.model}")
        print(f"   Model type: {type(service.model)}")
        
        print("\n2. Testing health check...")
        health = service.health_check()
        print(f"   Health check result: {health}")
        
        print("\n3. Attempting to calculate similarity (this should fail)...")
        try:
            result = service.calculate_similarity("Hello world", "Hi there")
            print(f"   Similarity result: {result}")
        except Exception as e:
            print(f"   ERROR: {type(e).__name__}: {e}")
            print(f"   This confirms the bug - model is None!")
            
        print("\n4. Testing encode method (this should also fail)...")
        try:
            embedding = service.encode("Test text")
            print(f"   Encoding result: {embedding}")
        except Exception as e:
            print(f"   ERROR: {type(e).__name__}: {e}")
            print(f"   This also confirms the bug!")
            
    except Exception as e:
        print(f"INITIALIZATION ERROR: {type(e).__name__}: {e}")
        print("Full traceback:")
        traceback.print_exc()
    
    print("\n" + "=" * 60)
    print("BUG DEMONSTRATION COMPLETE")
    print("The model is set to None instead of being loaded!")
    print("=" * 60)

if __name__ == "__main__":
    test_model_loading_bug()
