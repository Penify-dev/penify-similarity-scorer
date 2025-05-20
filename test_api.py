#!/usr/bin/env python3
"""
Test script for the Semantic Equivalence Classifier API.
This script sends sample requests to the API to demonstrate its usage.
"""
import requests
import json
import time
from typing import Dict, Any

# Change this to your API endpoint if it's different
API_URL = "http://localhost:16000"

def test_equivalence_api(premise: str, hypothesis: str) -> Dict[str, Any]:
    """Test the semantic equivalence classifier API endpoint."""
    endpoint = f"{API_URL}/classify-equivalence"
    
    data = {
        "premise": premise, 
        "hypothesis": hypothesis
    }
    
    start_time = time.time()
    response = requests.post(endpoint, json=data)
    elapsed = time.time() - start_time
    
    print(f"Request processed in {elapsed:.4f} seconds")
    
    if response.status_code == 200:
        result = response.json()
        return result
    else:
        print(f"Error: {response.status_code}")
        print(response.text)
        return {"error": response.text}

def test_similarity_api(sentence1: str, sentence2: str) -> Dict[str, Any]:
    """Test the semantic similarity API endpoint."""
    endpoint = f"{API_URL}/compare"
    
    data = {
        "sentence1": sentence1, 
        "sentence2": sentence2
    }
    
    start_time = time.time()
    response = requests.post(endpoint, json=data)
    elapsed = time.time() - start_time
    
    print(f"Request processed in {elapsed:.4f} seconds")
    
    if response.status_code == 200:
        result = response.json()
        return result
    else:
        print(f"Error: {response.status_code}")
        print(response.text)
        return {"error": response.text}

def check_api_health() -> Dict[str, Any]:
    """Check the health of the API."""
    endpoint = f"{API_URL}/health"
    
    response = requests.get(endpoint)
    
    if response.status_code == 200:
        result = response.json()
        return result
    else:
        print(f"Error: {response.status_code}")
        print(response.text)
        return {"error": response.text}

def print_result(result: Dict[str, Any], title: str) -> None:
    """Pretty print the result."""
    print("\n" + "=" * 50)
    print(title)
    print("=" * 50)
    print(json.dumps(result, indent=2))
    print("=" * 50 + "\n")

if __name__ == "__main__":
    # Check API health
    print("Checking API health...")
    health = check_api_health()
    print_result(health, "API Health Check")
    
    # Example 1: Entailment (Equivalence)
    premise = "The cat sat on the mat"
    hypothesis = "There was a cat on the mat"
    print(f"\nTesting equivalence API with:")
    print(f"Premise: '{premise}'")
    print(f"Hypothesis: '{hypothesis}'")
    result = test_equivalence_api(premise, hypothesis)
    print_result(result, "Equivalence Result (Expected: Entailment)")
    
    # Example 2: Contradiction
    premise = "The cat sat on the mat"
    hypothesis = "The cat was not on the mat"
    print(f"\nTesting equivalence API with:")
    print(f"Premise: '{premise}'")
    print(f"Hypothesis: '{hypothesis}'")
    result = test_equivalence_api(premise, hypothesis)
    print_result(result, "Equivalence Result (Expected: Contradiction)")
    
    # Example 3: Neutral
    premise = "The cat sat on the mat"
    hypothesis = "The mat was comfortable"
    print(f"\nTesting equivalence API with:")
    print(f"Premise: '{premise}'")
    print(f"Hypothesis: '{hypothesis}'")
    result = test_equivalence_api(premise, hypothesis)
    print_result(result, "Equivalence Result (Expected: Neutral)")
    
    # Example 4: Similarity comparison (from original API)
    sentence1 = "I love machine learning"
    sentence2 = "I enjoy artificial intelligence"
    print(f"\nTesting similarity API with:")
    print(f"Sentence 1: '{sentence1}'")
    print(f"Sentence 2: '{sentence2}'")
    result = test_similarity_api(sentence1, sentence2)
    print_result(result, "Similarity Result")
