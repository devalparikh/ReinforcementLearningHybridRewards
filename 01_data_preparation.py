#!/usr/bin/env python3
"""
Data Preparation for RLVR Training

This script prepares training data for the Reinforcement Learning with Verifiable Rewards (RLVR) system.
It generates synthetic data, validates it, and prepares it for training.

Usage:
    python 01_data_preparation.py
"""

import sys
import json
import random
from typing import List, Dict, Any
from dataclasses import dataclass
import pandas as pd
import numpy as np
from pathlib import Path
from sklearn.model_selection import train_test_split

# Add src to path
sys.path.append('src')

from config.training_config import get_fast_config
from data.dataset import RLVRTrainingDataset
from data.preprocessing import DataPreprocessor
from utils.logging import setup_logging
from utils.metrics import DataQualityMetrics


@dataclass
class TrainingExample:
    """Represents a training example for RLVR."""
    instruction: str
    expected_output: str
    task_type: str
    difficulty: str
    context: Dict[str, Any] = None


def generate_synthetic_data(num_samples: int = 1000) -> List[TrainingExample]:
    """Generate synthetic training data for RLVR."""
    
    # Code generation tasks
    code_tasks = [
        {
            "instruction": "Write a function to calculate the factorial of a number.",
            "expected_output": "def factorial(n):\n    if n <= 1:\n        return 1\n    return n * factorial(n-1)",
            "task_type": "code_generation",
            "difficulty": "easy"
        },
        {
            "instruction": "Create a function to find the maximum element in a list.",
            "expected_output": "def find_max(lst):\n    if not lst:\n        return None\n    return max(lst)",
            "task_type": "code_generation",
            "difficulty": "easy"
        },
        {
            "instruction": "Write a function to check if a string is a palindrome.",
            "expected_output": "def is_palindrome(s):\n    return s == s[::-1]",
            "task_type": "code_generation",
            "difficulty": "medium"
        },
        {
            "instruction": "Implement a binary search algorithm.",
            "expected_output": "def binary_search(arr, target):\n    left, right = 0, len(arr) - 1\n    while left <= right:\n        mid = (left + right) // 2\n        if arr[mid] == target:\n            return mid\n        elif arr[mid] < target:\n            left = mid + 1\n        else:\n            right = mid - 1\n    return -1",
            "task_type": "code_generation",
            "difficulty": "hard"
        }
    ]
    
    # Math reasoning tasks
    math_tasks = [
        {
            "instruction": "Solve: What is 15% of 200?",
            "expected_output": "15% of 200 = 0.15 × 200 = 30",
            "task_type": "math_reasoning",
            "difficulty": "easy"
        },
        {
            "instruction": "Calculate the area of a circle with radius 5.",
            "expected_output": "Area = πr² = π × 5² = 25π ≈ 78.54",
            "task_type": "math_reasoning",
            "difficulty": "medium"
        },
        {
            "instruction": "Find the sum of the first 10 natural numbers.",
            "expected_output": "Sum = n(n+1)/2 = 10(11)/2 = 55",
            "task_type": "math_reasoning",
            "difficulty": "easy"
        }
    ]
    
    # Logic reasoning tasks
    logic_tasks = [
        {
            "instruction": "If all roses are flowers and some flowers are red, can we conclude that some roses are red?",
            "expected_output": "No, we cannot conclude that some roses are red. The premises don't establish a connection between roses and red flowers.",
            "task_type": "logic_reasoning",
            "difficulty": "medium"
        },
        {
            "instruction": "A train leaves station A at 2 PM and arrives at station B at 4 PM. Another train leaves B at 1 PM and arrives at A at 3 PM. Do they meet?",
            "expected_output": "Yes, they meet. The first train travels from 2-4 PM, the second from 1-3 PM, so they overlap from 2-3 PM.",
            "task_type": "logic_reasoning",
            "difficulty": "hard"
        }
    ]
    
    all_tasks = code_tasks + math_tasks + logic_tasks
    
    # Generate synthetic data
    synthetic_data = []
    for i in range(num_samples):
        task = random.choice(all_tasks)
        
        # Add some variation to instructions
        instruction_variations = [
            task["instruction"],
            f"Please {task['instruction'].lower()}",
            f"Can you {task['instruction'].lower()}",
            f"I need help with: {task['instruction']}"
        ]
        
        example = TrainingExample(
            instruction=random.choice(instruction_variations),
            expected_output=task["expected_output"],
            task_type=task["task_type"],
            difficulty=task["difficulty"],
            context={"source": "synthetic", "id": f"task_{i:04d}"}
        )
        
        synthetic_data.append(example)
    
    return synthetic_data


def validate_data_quality(data: List[TrainingExample]) -> Dict[str, Any]:
    """Validate and analyze data quality."""
    metrics = DataQualityMetrics()
    
    # Convert to dict format for metrics
    data_dicts = [
        {
            "instruction": ex.instruction,
            "expected_output": ex.expected_output,
            "task_type": ex.task_type,
            "difficulty": ex.difficulty,
            "context": ex.context
        }
        for ex in data
    ]
    
    quality_report = metrics.analyze_dataset(data_dicts)
    return quality_report


def split_data(data: List[TrainingExample]) -> tuple:
    """Split data into train/validation/test sets."""
    # Convert to list for splitting
    data_list = [
        {
            "instruction": ex.instruction,
            "expected_output": ex.expected_output,
            "task_type": ex.task_type,
            "difficulty": ex.difficulty,
            "context": ex.context
        }
        for ex in data
    ]
    
    # Split data
    train_data, temp_data = train_test_split(
        data_list, 
        test_size=0.3, 
        random_state=42,
        stratify=[d['task_type'] for d in data_list]
    )
    
    val_data, test_data = train_test_split(
        temp_data, 
        test_size=0.5, 
        random_state=42,
        stratify=[d['task_type'] for d in temp_data]
    )
    
    return train_data, val_data, test_data


def main():
    """Main data preparation function."""
    print("=== Data Preparation for RLVR Training ===\n")
    
    # Setup logging
    logger = setup_logging(log_level="INFO")
    
    # Load configuration
    config = get_fast_config()
    print(f"Using configuration: {config.model.model_name}")
    print(f"Max sequence length: {config.model.max_length}")
    print(f"Batch size: {config.batch_size}")
    
    # Step 1: Generate synthetic data
    print("\n1. Generating synthetic training data...")
    training_data = generate_synthetic_data(num_samples=1000)
    print(f"Generated {len(training_data)} training examples")
    
    # Display sample data
    print("\nSample training data:")
    for i, example in enumerate(training_data[:3]):
        print(f"\nExample {i+1}:")
        print(f"Instruction: {example.instruction}")
        print(f"Expected: {example.expected_output[:100]}...")
        print(f"Type: {example.task_type}, Difficulty: {example.difficulty}")
    
    # Step 2: Validate data quality
    print("\n2. Validating data quality...")
    quality_report = validate_data_quality(training_data)
    
    print("\nData Quality Report:")
    print(f"Total samples: {quality_report['total_samples']}")
    print(f"Task type distribution: {quality_report['task_type_distribution']}")
    print(f"Difficulty distribution: {quality_report['difficulty_distribution']}")
    print(f"Average instruction length: {quality_report['avg_instruction_length']:.1f} characters")
    print(f"Average output length: {quality_report['avg_output_length']:.1f} characters")
    print(f"Data quality score: {quality_report['quality_score']:.2f}/1.0")
    
    # Step 3: Preprocess data
    print("\n3. Preprocessing data...")
    preprocessor = DataPreprocessor(config)
    
    # Convert to dict format for preprocessing
    data_dicts = [
        {
            "instruction": ex.instruction,
            "expected_output": ex.expected_output,
            "task_type": ex.task_type,
            "difficulty": ex.difficulty,
            "context": ex.context
        }
        for ex in training_data
    ]
    
    processed_data = preprocessor.preprocess(data_dicts)
    print(f"Preprocessed {len(processed_data)} examples")
    print(f"Filtered out {len(training_data) - len(processed_data)} invalid examples")
    
    # Show preprocessing statistics
    preprocessing_stats = preprocessor.get_preprocessing_stats()
    print("\nPreprocessing Statistics:")
    for stat, value in preprocessing_stats.items():
        print(f"{stat}: {value}")
    
    # Step 4: Split data
    print("\n4. Splitting data into train/val/test sets...")
    train_data, val_data, test_data = split_data(training_data)
    
    print(f"Data split:")
    print(f"  Training: {len(train_data)} samples ({len(train_data)/len(processed_data)*100:.1f}%)")
    print(f"  Validation: {len(val_data)} samples ({len(val_data)/len(processed_data)*100:.1f}%)")
    print(f"  Test: {len(test_data)} samples ({len(test_data)/len(processed_data)*100:.1f}%)")
    
    # Verify stratification
    print("\nTask type distribution across splits:")
    for split_name, split_data in [("Train", train_data), ("Val", val_data), ("Test", test_data)]:
        task_counts = {}
        for item in split_data:
            task_counts[item['task_type']] = task_counts.get(item['task_type'], 0) + 1
        print(f"  {split_name}: {task_counts}")
    
    # Step 5: Create training datasets
    print("\n5. Creating training datasets...")
    train_dataset = RLVRTrainingDataset(train_data, config)
    val_dataset = RLVRTrainingDataset(val_data, config)
    test_dataset = RLVRTrainingDataset(test_data, config)
    
    print(f"Created datasets:")
    print(f"  Training dataset: {len(train_dataset)} samples")
    print(f"  Validation dataset: {len(val_dataset)} samples")
    print(f"  Test dataset: {len(test_dataset)} samples")
    
    # Test dataset iteration
    print("\nTesting dataset iteration...")
    sample_batch = next(iter(train_dataset))
    print(f"Sample batch keys: {list(sample_batch.keys())}")
    print(f"Sample instruction: {sample_batch['instruction'][:100]}...")
    print(f"Sample expected output: {sample_batch['expected_output'][:100]}...")
    
    # Step 6: Save processed data
    print("\n6. Saving processed data...")
    data_dir = Path("data/processed")
    data_dir.mkdir(parents=True, exist_ok=True)
    
    # Save splits
    with open(data_dir / "train_data.json", 'w') as f:
        json.dump(train_data, f, indent=2)
    
    with open(data_dir / "val_data.json", 'w') as f:
        json.dump(val_data, f, indent=2)
    
    with open(data_dir / "test_data.json", 'w') as f:
        json.dump(test_data, f, indent=2)
    
    print("Saved processed data to data/processed/")
    print("Files created:")
    print("  - train_data.json")
    print("  - val_data.json")
    print("  - test_data.json")
    
    # Summary
    print("\n=== Data Preparation Summary ===")
    print(f"✓ Generated {len(training_data)} synthetic training examples")
    print(f"✓ Validated data quality (score: {quality_report['quality_score']:.2f}/1.0)")
    print(f"✓ Preprocessed and filtered data")
    print(f"✓ Split into train/val/test sets")
    print(f"✓ Created training datasets")
    print(f"✓ Saved processed data to data/processed/")
    print("\nNext steps:")
    print("1. Run 02_verifier_development.py to test verifiers")
    print("2. Run 03_rlvr_training.py to start training")
    print("3. Run 04_evaluation.py to evaluate results")


if __name__ == "__main__":
    main() 