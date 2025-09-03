#!/usr/bin/env python3
"""
Verifier Development and Testing

This script tests and validates the different verifiers used in the RLVR system.
It tests code execution, mathematical reasoning, and logical verification.

Usage:
    python 02_verifier_development.py
"""

import sys
import json
import time
from typing import List, Dict, Any
import numpy as np
from pathlib import Path

# Add src to path
sys.path.append('src')

from config.training_config import get_fast_config
from verifiers.code_verifier import CodeVerifier
from verifiers.math_verifier import MathVerifier
from verifiers.logic_verifier import LogicVerifier
from utils.logging import setup_logging
from utils.metrics import VerificationMetrics


def test_code_verifier():
    """Test code verifier with various examples."""
    print("Testing Code Verifier:")
    
    code_verifier = CodeVerifier(config={
        "timeout": 30,
        "safe_mode": True,
        "allowed_modules": ["math", "random", "datetime", "collections"]
    })
    
    code_test_cases = [
        {
            "instruction": "Write a function to calculate factorial.",
            "model_output": "def factorial(n):\n    if n <= 1:\n        return 1\n    return n * factorial(n-1)",
            "expected_output": "def factorial(n):\n    if n <= 1:\n        return 1\n    return n * factorial(n-1)",
            "description": "Correct factorial function"
        },
        {
            "instruction": "Write a function to find maximum in a list.",
            "model_output": "def find_max(lst):\n    if not lst:\n        return None\n    return max(lst)",
            "expected_output": "def find_max(lst):\n    if not lst:\n        return None\n    return max(lst)",
            "description": "Correct max function"
        },
        {
            "instruction": "Write a function to check palindrome.",
            "model_output": "def is_palindrome(s):\n    return s == s[::-1]",
            "expected_output": "def is_palindrome(s):\n    return s == s[::-1]",
            "description": "Correct palindrome function"
        },
        {
            "instruction": "Write a function to calculate factorial.",
            "model_output": "def factorial(n):\n    return n * factorial(n-1)",
            "expected_output": "def factorial(n):\n    if n <= 1:\n        return 1\n    return n * factorial(n-1)",
            "description": "Incorrect factorial (missing base case)"
        }
    ]
    
    results = []
    for i, test_case in enumerate(code_test_cases):
        print(f"\nTest {i+1}: {test_case['description']}")
        
        try:
            result = code_verifier.verify(
                instruction=test_case["instruction"],
                model_output=test_case["model_output"],
                expected_output=test_case["expected_output"]
            )
            
            print(f"  Result: {result.result}")
            print(f"  Score: {result.score:.3f}")
            print(f"  Details: {result.details}")
            
            results.append({
                "test_case": test_case,
                "result": result,
                "success": result.result.value == "CORRECT"
            })
            
        except Exception as e:
            print(f"  Error: {e}")
            results.append({
                "test_case": test_case,
                "result": None,
                "success": False,
                "error": str(e)
            })
    
    # Summary
    successful_tests = sum(1 for r in results if r["success"])
    print(f"\nCode Verifier Summary: {successful_tests}/{len(results)} tests passed")
    
    return results


def test_math_verifier():
    """Test math verifier with various examples."""
    print("\nTesting Math Verifier:")
    
    math_verifier = MathVerifier(config={
        "tolerance": 1e-6,
        "max_steps": 10
    })
    
    math_test_cases = [
        {
            "instruction": "What is 15% of 200?",
            "model_output": "15% of 200 = 0.15 × 200 = 30",
            "expected_output": "15% of 200 = 0.15 × 200 = 30",
            "description": "Correct percentage calculation"
        },
        {
            "instruction": "Calculate the area of a circle with radius 5.",
            "model_output": "Area = πr² = π × 5² = 25π ≈ 78.54",
            "expected_output": "Area = πr² = π × 5² = 25π ≈ 78.54",
            "description": "Correct circle area calculation"
        },
        {
            "instruction": "Find the sum of first 10 natural numbers.",
            "model_output": "Sum = n(n+1)/2 = 10(11)/2 = 55",
            "expected_output": "Sum = n(n+1)/2 = 10(11)/2 = 55",
            "description": "Correct arithmetic sum"
        },
        {
            "instruction": "What is 15% of 200?",
            "model_output": "15% of 200 = 0.15 × 200 = 25",
            "expected_output": "15% of 200 = 0.15 × 200 = 30",
            "description": "Incorrect calculation"
        }
    ]
    
    results = []
    for i, test_case in enumerate(math_test_cases):
        print(f"\nTest {i+1}: {test_case['description']}")
        
        try:
            result = math_verifier.verify(
                instruction=test_case["instruction"],
                model_output=test_case["model_output"],
                expected_output=test_case["expected_output"]
            )
            
            print(f"  Result: {result.result}")
            print(f"  Score: {result.score:.3f}")
            print(f"  Details: {result.details}")
            
            results.append({
                "test_case": test_case,
                "result": result,
                "success": result.result.value == "CORRECT"
            })
            
        except Exception as e:
            print(f"  Error: {e}")
            results.append({
                "test_case": test_case,
                "result": None,
                "success": False,
                "error": str(e)
            })
    
    # Summary
    successful_tests = sum(1 for r in results if r["success"])
    print(f"\nMath Verifier Summary: {successful_tests}/{len(results)} tests passed")
    
    return results


def test_logic_verifier():
    """Test logic verifier with various examples."""
    print("\nTesting Logic Verifier:")
    
    logic_verifier = LogicVerifier(config={
        "reasoning_depth": 3,
        "consistency_check": True
    })
    
    logic_test_cases = [
        {
            "instruction": "If all roses are flowers and some flowers are red, can we conclude that some roses are red?",
            "model_output": "No, we cannot conclude that some roses are red. The premises don't establish a connection between roses and red flowers.",
            "expected_output": "No, we cannot conclude that some roses are red. The premises don't establish a connection between roses and red flowers.",
            "description": "Correct logical reasoning"
        },
        {
            "instruction": "A train leaves station A at 2 PM and arrives at station B at 4 PM. Another train leaves B at 1 PM and arrives at A at 3 PM. Do they meet?",
            "model_output": "Yes, they meet. The first train travels from 2-4 PM, the second from 1-3 PM, so they overlap from 2-3 PM.",
            "expected_output": "Yes, they meet. The first train travels from 2-4 PM, the second from 1-3 PM, so they overlap from 2-3 PM.",
            "description": "Correct temporal reasoning"
        },
        {
            "instruction": "If all A are B and all B are C, can we conclude that all A are C?",
            "model_output": "Yes, this is a valid syllogism. If all A are B and all B are C, then all A must be C.",
            "expected_output": "Yes, this is a valid syllogism. If all A are B and all B are C, then all A must be C.",
            "description": "Correct syllogistic reasoning"
        }
    ]
    
    results = []
    for i, test_case in enumerate(logic_test_cases):
        print(f"\nTest {i+1}: {test_case['description']}")
        
        try:
            result = logic_verifier.verify(
                instruction=test_case["instruction"],
                model_output=test_case["model_output"],
                expected_output=test_case["expected_output"]
            )
            
            print(f"  Result: {result.result}")
            print(f"  Score: {result.score:.3f}")
            print(f"  Details: {result.details}")
            
            results.append({
                "test_case": test_case,
                "result": result,
                "success": result.result.value == "CORRECT"
            })
            
        except Exception as e:
            print(f"  Error: {e}")
            results.append({
                "test_case": test_case,
                "result": None,
                "success": False,
                "error": str(e)
            })
    
    # Summary
    successful_tests = sum(1 for r in results if r["success"])
    print(f"\nLogic Verifier Summary: {successful_tests}/{len(results)} tests passed")
    
    return results


def batch_verify(verifier, data, task_type):
    """Run batch verification on training data."""
    results = []
    start_time = time.time()
    
    for i, item in enumerate(data):
        try:
            result = verifier.verify(
                instruction=item['instruction'],
                model_output=item['expected_output'],  # Using expected as model output for testing
                expected_output=item['expected_output']
            )
            results.append({
                'id': item.get('id', f'{task_type}_{i}'),
                'result': result,
                'success': result.result.value == "CORRECT"
            })
        except Exception as e:
            results.append({
                'id': item.get('id', f'{task_type}_{i}'),
                'result': None,
                'success': False,
                'error': str(e)
            })
    
    end_time = time.time()
    return results, end_time - start_time


def test_on_training_data():
    """Test verifiers on actual training data."""
    print("\nRunning batch verification tests on training data...")
    
    # Load training data
    try:
        with open('data/processed/train_data.json', 'r') as f:
            training_data = json.load(f)
        print(f"Loaded {len(training_data)} training examples")
    except FileNotFoundError:
        print("No processed data found. Please run 01_data_preparation.py first.")
        return {}
    
    # Filter data by task type
    code_data = [d for d in training_data if d['task_type'] == 'code_generation'][:50]
    math_data = [d for d in training_data if d['task_type'] == 'math_reasoning'][:50]
    logic_data = [d for d in training_data if d['task_type'] == 'logic_reasoning'][:50]
    
    print(f"Testing on {len(code_data)} code examples, {len(math_data)} math examples, {len(logic_data)} logic examples")
    
    # Initialize verifiers
    code_verifier = CodeVerifier(config={
        "timeout": 30,
        "safe_mode": True,
        "allowed_modules": ["math", "random", "datetime", "collections"]
    })
    
    math_verifier = MathVerifier(config={
        "tolerance": 1e-6,
        "max_steps": 10
    })
    
    logic_verifier = LogicVerifier(config={
        "reasoning_depth": 3,
        "consistency_check": True
    })
    
    # Run batch tests
    batch_results = {}
    
    if code_data:
        print("\nTesting code verifier on batch data...")
        code_batch_results, code_time = batch_verify(code_verifier, code_data, 'code')
        batch_results['code'] = {'results': code_batch_results, 'time': code_time}
        
        success_rate = sum(1 for r in code_batch_results if r['success']) / len(code_batch_results)
        print(f"Code verifier: {success_rate:.2%} success rate, {code_time:.2f}s")
    
    if math_data:
        print("\nTesting math verifier on batch data...")
        math_batch_results, math_time = batch_verify(math_verifier, math_data, 'math')
        batch_results['math'] = {'results': math_batch_results, 'time': math_time}
        
        success_rate = sum(1 for r in math_batch_results if r['success']) / len(math_batch_results)
        print(f"Math verifier: {success_rate:.2%} success rate, {math_time:.2f}s")
    
    if logic_data:
        print("\nTesting logic verifier on batch data...")
        logic_batch_results, logic_time = batch_verify(logic_verifier, logic_data, 'logic')
        batch_results['logic'] = {'results': logic_batch_results, 'time': logic_time}
        
        success_rate = sum(1 for r in logic_batch_results if r['success']) / len(logic_batch_results)
        print(f"Logic verifier: {success_rate:.2%} success rate, {logic_time:.2f}s")
    
    return batch_results


def analyze_performance(batch_results):
    """Analyze verification performance."""
    print("\n=== Verification Performance Analysis ===")
    
    for verifier_type, batch_data in batch_results.items():
        results = batch_data['results']
        total_time = batch_data['time']
        
        success_count = sum(1 for r in results if r['success'])
        error_count = sum(1 for r in results if 'error' in r)
        avg_time_per_test = total_time / len(results)
        
        print(f"\n{verifier_type.title()} Verifier:")
        print(f"  Total tests: {len(results)}")
        print(f"  Success rate: {success_count/len(results):.2%}")
        print(f"  Error rate: {error_count/len(results):.2%}")
        print(f"  Total time: {total_time:.2f}s")
        print(f"  Average time per test: {avg_time_per_test:.3f}s")
        
        # Score distribution
        scores = [r['result'].score for r in results if r['result'] is not None]
        if scores:
            print(f"  Average score: {np.mean(scores):.3f}")
            print(f"  Score std: {np.std(scores):.3f}")
            print(f"  Min score: {min(scores):.3f}")
            print(f"  Max score: {max(scores):.3f}")


def save_results(individual_results, batch_results):
    """Save verification results."""
    verification_results = {
        'individual_tests': {
            'code': individual_results['code'],
            'math': individual_results['math'],
            'logic': individual_results['logic']
        },
        'batch_tests': batch_results,
        'summary': {
            'total_tests': len(individual_results['code']) + len(individual_results['math']) + len(individual_results['logic']),
            'code_success_rate': sum(1 for r in individual_results['code'] if r['success']) / len(individual_results['code']) if individual_results['code'] else 0,
            'math_success_rate': sum(1 for r in individual_results['math'] if r['success']) / len(individual_results['math']) if individual_results['math'] else 0,
            'logic_success_rate': sum(1 for r in individual_results['logic'] if r['success']) / len(individual_results['logic']) if individual_results['logic'] else 0
        }
    }
    
    results_dir = Path("results")
    results_dir.mkdir(exist_ok=True)
    
    with open(results_dir / "verification_results.json", 'w') as f:
        json.dump(verification_results, f, indent=2, default=str)
    
    print("\nSaved verification results to results/verification_results.json")


def main():
    """Main verifier development function."""
    print("=== Verifier Development and Testing ===\n")
    
    # Setup logging
    logger = setup_logging(log_level="INFO")
    
    # Load configuration
    config = get_fast_config()
    
    # Step 1: Test individual verifiers
    print("1. Testing individual verifiers...")
    code_results = test_code_verifier()
    math_results = test_math_verifier()
    logic_results = test_logic_verifier()
    
    individual_results = {
        'code': code_results,
        'math': math_results,
        'logic': logic_results
    }
    
    # Step 2: Test on training data
    print("\n2. Testing verifiers on training data...")
    batch_results = test_on_training_data()
    
    # Step 3: Analyze performance
    if batch_results:
        analyze_performance(batch_results)
    
    # Step 4: Save results
    print("\n3. Saving verification results...")
    save_results(individual_results, batch_results)
    
    # Summary
    print("\n=== Verifier Development Summary ===")
    print(f"✓ Tested {len(code_results)} code verification examples")
    print(f"✓ Tested {len(math_results)} math verification examples")
    print(f"✓ Tested {len(logic_results)} logic verification examples")
    print(f"✓ Ran batch verification on training data")
    print(f"✓ Analyzed verification performance")
    print(f"✓ Saved verification results")
    
    print("\nVerifier Performance Summary:")
    for verifier_type, batch_data in batch_results.items():
        results = batch_data['results']
        success_rate = sum(1 for r in results if r['success']) / len(results)
        print(f"  {verifier_type.title()}: {success_rate:.2%} success rate")
    
    print("\nNext steps:")
    print("1. Run 03_rlvr_training.py to start training with verified rewards")
    print("2. Run 04_evaluation.py to evaluate training results")


if __name__ == "__main__":
    main() 