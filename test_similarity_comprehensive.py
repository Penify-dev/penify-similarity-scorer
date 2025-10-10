"""
Comprehensive Test Suite for Simple Similarity Scorer
Contains 100 test cases covering various scenarios and edge cases.
"""
import time
import statistics
import json
from typing import List, Dict, Tuple
from simple_similarity_scorer import SimpleSimilarityScorer

class SimilarityTestSuite:
    """Comprehensive test suite for similarity scoring."""
    
    def __init__(self):
        self.scorer = SimpleSimilarityScorer()
        self.test_cases = self._generate_test_cases()
        self.results = []
    
    def _generate_test_cases(self) -> List[Dict]:
        """Generate 100 comprehensive test cases."""
        test_cases = []
        
        # Category 1: Identical and Near-Identical (20 cases)
        identical_tests = [
            ("Hello world", "Hello world", 1.0, "identical_exact"),
            ("The quick brown fox", "The quick brown fox", 1.0, "identical_sentence"),
            ("Machine learning", "machine learning", 0.95, "case_difference"),
            ("AI is amazing", "AI is amazing.", 0.95, "punctuation_difference"),
            ("Python programming", "Python programming!", 0.95, "exclamation_added"),
            ("Data science rocks", "Data science rocks?", 0.95, "question_mark_added"),
            ("I love coding", "I love coding...", 0.95, "ellipsis_added"),
            ("Natural language processing", "Natural Language Processing", 0.95, "title_case"),
            ("Deep learning models", "deep learning models", 0.95, "lower_case"),
            ("Neural networks", "Neural Networks", 0.95, "title_case_partial"),
            ("Algorithm optimization", "ALGORITHM OPTIMIZATION", 0.90, "all_caps"),
            ("Software development", "software  development", 0.95, "extra_space"),
            ("Computer vision", "Computer-vision", 0.90, "hyphen_difference"),
            ("Big data analytics", "Big data analytics ", 0.98, "trailing_space"),
            ("Cloud computing", " Cloud computing", 0.98, "leading_space"),
            ("Artificial intelligence", "Artificial intelligence", 1.0, "ai_full_term"),
            ("Database management", "Database management", 1.0, "db_term"),
            ("Web development", "Web development", 1.0, "web_dev"),
            ("Mobile applications", "Mobile applications", 1.0, "mobile_apps"),
            ("Cybersecurity measures", "Cybersecurity measures", 1.0, "security_term")
        ]
        
        # Category 2: High Similarity - Different Wording (20 cases)
        high_similarity_tests = [
            ("The cat sits on the mat", "A cat is sitting on a mat", 0.80, "cat_mat_paraphrase"),
            ("I enjoy programming", "I love coding", 0.75, "enjoy_vs_love"),
            ("The weather is nice today", "Today's weather is pleasant", 0.70, "weather_pleasant"),
            ("She runs every morning", "She jogs each morning", 0.80, "runs_vs_jogs"),
            ("The book was interesting", "The novel was fascinating", 0.70, "book_novel_interesting"),
            ("He drives to work", "He commutes by car", 0.75, "drives_commutes"),
            ("The movie was excellent", "The film was outstanding", 0.80, "movie_film_excellent"),
            ("I'm feeling happy", "I'm feeling joyful", 0.85, "happy_joyful"),
            ("The dog is barking loudly", "The puppy is making loud noises", 0.65, "dog_puppy_barking"),
            ("She bought a new car", "She purchased a new vehicle", 0.80, "bought_purchased_car"),
            ("The food tastes delicious", "The meal is very tasty", 0.75, "food_meal_delicious"),
            ("He's studying mathematics", "He's learning math", 0.80, "studying_learning_math"),
            ("The concert was amazing", "The performance was incredible", 0.70, "concert_performance"),
            ("I need to sleep", "I require rest", 0.70, "sleep_rest"),
            ("The house is beautiful", "The home is lovely", 0.75, "house_home_beautiful"),
            ("She's writing a letter", "She's composing a message", 0.70, "writing_composing"),
            ("The train is late", "The locomotive is delayed", 0.70, "train_locomotive_late"),
            ("He's cooking dinner", "He's preparing the evening meal", 0.75, "cooking_preparing_dinner"),
            ("The garden looks nice", "The yard appears pleasant", 0.70, "garden_yard_nice"),
            ("I'm reading a book", "I'm perusing a novel", 0.70, "reading_perusing_book")
        ]
        
        # Category 3: Medium Similarity - Related Topics (20 cases)
        medium_similarity_tests = [
            ("Machine learning algorithms", "Artificial intelligence systems", 0.60, "ml_ai_related"),
            ("Python programming", "Java development", 0.55, "programming_languages"),
            ("Database design", "Data modeling", 0.60, "data_related"),
            ("Web development", "Mobile app creation", 0.50, "development_types"),
            ("Cloud computing", "Distributed systems", 0.55, "computing_paradigms"),
            ("Cybersecurity threats", "Network vulnerabilities", 0.60, "security_related"),
            ("Data visualization", "Chart generation", 0.65, "visualization_related"),
            ("Natural language processing", "Text analysis", 0.65, "nlp_text"),
            ("Computer vision", "Image recognition", 0.70, "vision_recognition"),
            ("Software testing", "Quality assurance", 0.65, "testing_qa"),
            ("Project management", "Team coordination", 0.55, "management_related"),
            ("User interface design", "User experience optimization", 0.60, "ui_ux"),
            ("Version control", "Source code management", 0.65, "version_control"),
            ("API development", "Service integration", 0.55, "api_services"),
            ("Data mining", "Information extraction", 0.60, "data_extraction"),
            ("Algorithm complexity", "Performance optimization", 0.55, "performance_related"),
            ("Software architecture", "System design", 0.60, "architecture_design"),
            ("Code review", "Peer programming", 0.55, "code_collaboration"),
            ("Technical documentation", "Developer guides", 0.60, "documentation_related"),
            ("Agile methodology", "Scrum framework", 0.65, "agile_scrum")
        ]
        
        # Category 4: Low Similarity - Different Domains (20 cases)
        low_similarity_tests = [
            ("Machine learning", "Cooking recipes", 0.10, "tech_vs_cooking"),
            ("Database queries", "Musical instruments", 0.05, "tech_vs_music"),
            ("Python programming", "Gardening tips", 0.05, "programming_vs_gardening"),
            ("Cloud computing", "Weather forecast", 0.15, "cloud_computing_vs_weather"),
            ("Artificial intelligence", "Ancient history", 0.05, "ai_vs_history"),
            ("Software development", "Ocean exploration", 0.05, "software_vs_ocean"),
            ("Cybersecurity", "Fashion trends", 0.05, "security_vs_fashion"),
            ("Data analytics", "Space travel", 0.10, "data_vs_space"),
            ("Web design", "Mountain climbing", 0.05, "web_vs_mountain"),
            ("Mobile apps", "Marine biology", 0.05, "apps_vs_biology"),
            ("Algorithm optimization", "Fruit cultivation", 0.05, "algorithms_vs_fruit"),
            ("User interface", "Desert landscapes", 0.05, "ui_vs_desert"),
            ("Code debugging", "Poetry writing", 0.05, "debugging_vs_poetry"),
            ("Network security", "Dance performances", 0.05, "security_vs_dance"),
            ("API integration", "Astronomy research", 0.05, "api_vs_astronomy"),
            ("Version control", "Culinary arts", 0.05, "version_vs_culinary"),
            ("Software testing", "Wildlife photography", 0.05, "testing_vs_wildlife"),
            ("System architecture", "Medieval literature", 0.05, "architecture_vs_literature"),
            ("Performance tuning", "Ice skating techniques", 0.05, "performance_vs_skating"),
            ("Technical writing", "Butterfly migration", 0.05, "writing_vs_butterflies")
        ]
        
        # Category 5: Edge Cases and Special Scenarios (20 cases)
        edge_case_tests = [
            ("", "", 1.0, "empty_strings"),
            ("a", "a", 1.0, "single_character"),
            ("The", "the", 0.95, "single_word_case"),
            ("123", "123", 1.0, "numbers_identical"),
            ("123", "456", 0.20, "different_numbers"),
            ("Hello! @#$%", "Hello! @#$%", 1.0, "special_characters"),
            ("Very long sentence with many words that keeps going and going and going", 
             "Very long sentence with many words that continues and continues and continues", 0.70, "long_similar"),
            ("Short", "Very long sentence with many words", 0.10, "length_mismatch"),
            ("?!@#$%^&*()", "!@#$%^&*()?", 0.90, "symbols_only"),
            ("The quick brown fox jumps", "fox brown quick The jumps", 0.60, "word_order_scrambled"),
            ("COVID-19 pandemic response", "COVID-19 pandemic response", 1.0, "covid_term"),
            ("AI/ML engineering roles", "AI/ML engineering roles", 1.0, "slash_notation"),
            ("www.example.com/path", "www.example.com/path", 1.0, "url_identical"),
            ("user@domain.com", "user@domain.com", 1.0, "email_identical"),
            ("2023-10-10 15:30:00", "2023-10-10 15:30:00", 1.0, "timestamp_identical"),
            ("Price: $19.99", "Cost: $19.99", 0.70, "price_vs_cost"),
            ("React.js development", "Vue.js development", 0.70, "js_frameworks"),
            ("iOS app store", "Android play store", 0.60, "app_stores"),
            ("MacBook Pro 2023", "MacBook Air 2023", 0.80, "mac_models"),
            ("GitHub repository", "GitLab project", 0.65, "git_platforms")
        ]
        
        # Combine all test cases
        all_tests = (identical_tests + high_similarity_tests + 
                    medium_similarity_tests + low_similarity_tests + edge_case_tests)
        
        # Convert to standardized format
        for i, (text1, text2, expected_min, category) in enumerate(all_tests):
            test_cases.append({
                "id": i + 1,
                "text1": text1,
                "text2": text2,
                "expected_min_similarity": expected_min,
                "category": category,
                "description": f"Test case {i+1}: {category}"
            })
        
        return test_cases
    
    def run_single_test(self, test_case: Dict) -> Dict:
        """Run a single test case."""
        start_time = time.time()
        
        try:
            similarity = self.scorer.calculate_similarity(test_case["text1"], test_case["text2"])
            elapsed_time = time.time() - start_time
            
            # Determine if test passed based on expected minimum similarity
            expected_min = test_case["expected_min_similarity"]
            
            # For high expected similarity (>0.8), allow some tolerance
            if expected_min >= 0.8:
                tolerance = 0.15
                passed = similarity >= (expected_min - tolerance)
            # For medium similarity (0.4-0.8), use moderate tolerance
            elif expected_min >= 0.4:
                tolerance = 0.20
                passed = similarity >= (expected_min - tolerance)
            # For low similarity (<0.4), just check it's reasonably low
            else:
                passed = similarity <= 0.4
            
            result = {
                "id": test_case["id"],
                "category": test_case["category"],
                "text1": test_case["text1"],
                "text2": test_case["text2"],
                "expected_min_similarity": expected_min,
                "actual_similarity": round(similarity, 4),
                "elapsed_time_ms": round(elapsed_time * 1000, 2),
                "passed": passed,
                "description": test_case["description"]
            }
            
        except Exception as e:
            result = {
                "id": test_case["id"],
                "category": test_case["category"],
                "text1": test_case["text1"],
                "text2": test_case["text2"],
                "expected_min_similarity": test_case["expected_min_similarity"],
                "actual_similarity": None,
                "elapsed_time_ms": None,
                "passed": False,
                "error": str(e),
                "description": test_case["description"]
            }
        
        return result
    
    def run_all_tests(self) -> Dict:
        """Run all test cases and return comprehensive results."""
        print("Running comprehensive similarity test suite...")
        print(f"Total test cases: {len(self.test_cases)}")
        
        start_time = time.time()
        results = []
        
        for i, test_case in enumerate(self.test_cases):
            if (i + 1) % 10 == 0:
                print(f"Progress: {i + 1}/{len(self.test_cases)} tests completed")
            
            result = self.run_single_test(test_case)
            results.append(result)
        
        total_time = time.time() - start_time
        
        # Calculate statistics
        passed_tests = [r for r in results if r["passed"]]
        failed_tests = [r for r in results if not r["passed"]]
        
        valid_times = [r["elapsed_time_ms"] for r in results if r["elapsed_time_ms"] is not None]
        
        summary = {
            "total_tests": len(results),
            "passed": len(passed_tests),
            "failed": len(failed_tests),
            "success_rate": round((len(passed_tests) / len(results)) * 100, 2),
            "total_time_seconds": round(total_time, 2),
            "average_time_per_test_ms": round(statistics.mean(valid_times) if valid_times else 0, 2),
            "median_time_per_test_ms": round(statistics.median(valid_times) if valid_times else 0, 2),
            "min_time_ms": round(min(valid_times) if valid_times else 0, 2),
            "max_time_ms": round(max(valid_times) if valid_times else 0, 2)
        }
        
        # Category breakdown
        categories = {}
        for result in results:
            cat = result["category"].split("_")[0] if "_" in result["category"] else result["category"]
            if cat not in categories:
                categories[cat] = {"total": 0, "passed": 0}
            categories[cat]["total"] += 1
            if result["passed"]:
                categories[cat]["passed"] += 1
        
        for cat in categories:
            total = categories[cat]["total"]
            passed = categories[cat]["passed"]
            categories[cat]["success_rate"] = round((passed / total) * 100, 2) if total > 0 else 0
        
        return {
            "summary": summary,
            "categories": categories,
            "results": results,
            "failed_tests": failed_tests
        }
    
    def save_results(self, results: Dict, filename: str = "similarity_test_results.json"):
        """Save test results to a JSON file."""
        with open(filename, 'w', encoding='utf-8') as f:
            json.dump(results, f, indent=2, ensure_ascii=False)
        print(f"Results saved to {filename}")
    
    def print_summary(self, results: Dict):
        """Print a formatted summary of test results."""
        summary = results["summary"]
        categories = results["categories"]
        
        print("\n" + "="*80)
        print("SIMILARITY SCORER TEST RESULTS SUMMARY")
        print("="*80)
        
        print(f"Total Tests: {summary['total_tests']}")
        print(f"Passed: {summary['passed']}")
        print(f"Failed: {summary['failed']}")
        print(f"Success Rate: {summary['success_rate']}%")
        print(f"Total Time: {summary['total_time_seconds']}s")
        print(f"Average Time per Test: {summary['average_time_per_test_ms']}ms")
        print(f"Median Time per Test: {summary['median_time_per_test_ms']}ms")
        
        print("\nCategory Breakdown:")
        print("-" * 40)
        for cat, stats in categories.items():
            print(f"{cat}: {stats['passed']}/{stats['total']} ({stats['success_rate']}%)")
        
        # Show some failed tests if any
        failed_tests = results["failed_tests"]
        if failed_tests:
            print(f"\nFirst 5 Failed Tests:")
            print("-" * 40)
            for test in failed_tests[:5]:
                print(f"ID {test['id']}: {test['category']}")
                print(f"  Text1: {test['text1'][:50]}...")
                print(f"  Text2: {test['text2'][:50]}...")
                print(f"  Expected: ≥{test['expected_min_similarity']}, Got: {test['actual_similarity']}")
                if 'error' in test:
                    print(f"  Error: {test['error']}")
                print()
        
        print("="*80)

def main():
    """Run the comprehensive test suite."""
    print("Initializing Simple Similarity Scorer Test Suite...")
    
    # Initialize test suite
    test_suite = SimilarityTestSuite()
    
    # Check if scorer is healthy
    if not test_suite.scorer.is_healthy():
        print("ERROR: Similarity scorer failed health check!")
        return
    
    print(f"Scorer Info: {test_suite.scorer.info}")
    print()
    
    # Run all tests
    results = test_suite.run_all_tests()
    
    # Print summary
    test_suite.print_summary(results)
    
    # Save results
    test_suite.save_results(results)
    
    # Print some sample results
    print("\nSample Test Results:")
    print("-" * 40)
    for result in results[:5]:
        print(f"Test {result['id']}: {result['category']}")
        print(f"  Similarity: {result['actual_similarity']}")
        print(f"  Time: {result['elapsed_time_ms']}ms")
        print(f"  Status: {'PASS' if result['passed'] else 'FAIL'}")
        print()

if __name__ == "__main__":
    main()
