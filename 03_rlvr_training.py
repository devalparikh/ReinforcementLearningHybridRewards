#!/usr/bin/env python3
"""
RLVR Training Script

This script demonstrates the complete Reinforcement Learning with Verifiable Rewards (RLVR) training pipeline.
It initializes all components and runs the training process.

Usage:
    python 03_rlvr_training.py
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
from data.dataset import RLVRTrainingDataset
from models.language_model import LanguageModel
from verifiers.code_verifier import CodeVerifier
from verifiers.math_verifier import MathVerifier
from verifiers.logic_verifier import LogicVerifier
from rewards.hybrid_reward import HybridReward
from training.rlvr_trainer import RLVRTrainer
from utils.logging import setup_logging
from utils.metrics import TrainingMetrics


class MockLanguageModel:
    """Mock language model for demonstration purposes."""
    
    def __init__(self, config):
        self.config = config
        self.name = "mock_model"
        self.training_mode = False
    
    def generate(self, instruction: str, max_length: int = None) -> str:
        """Generate a mock response."""
        # Simple mock responses based on instruction type
        if "factorial" in instruction.lower():
            return "def factorial(n):\n    if n <= 1:\n        return 1\n    return n * factorial(n-1)"
        elif "percentage" in instruction.lower() or "15%" in instruction:
            return "15% of 200 = 0.15 × 200 = 30"
        elif "palindrome" in instruction.lower():
            return "def is_palindrome(s):\n    return s == s[::-1]"
        elif "maximum" in instruction.lower() or "max" in instruction.lower():
            return "def find_max(lst):\n    if not lst:\n        return None\n    return max(lst)"
        elif "binary search" in instruction.lower():
            return "def binary_search(arr, target):\n    left, right = 0, len(arr) - 1\n    while left <= right:\n        mid = (left + right) // 2\n        if arr[mid] == target:\n            return mid\n        elif arr[mid] < target:\n            left = mid + 1\n        else:\n            right = mid - 1\n    return -1"
        elif "area" in instruction.lower() and "circle" in instruction.lower():
            return "Area = πr² = π × 5² = 25π ≈ 78.54"
        elif "sum" in instruction.lower() and "natural" in instruction.lower():
            return "Sum = n(n+1)/2 = 10(11)/2 = 55"
        elif "roses" in instruction.lower() and "flowers" in instruction.lower():
            return "No, we cannot conclude that some roses are red. The premises don't establish a connection between roses and red flowers."
        elif "train" in instruction.lower() and "station" in instruction.lower():
            return "Yes, they meet. The first train travels from 2-4 PM, the second from 1-3 PM, so they overlap from 2-3 PM."
        else:
            return "This is a mock response for demonstration purposes."
    
    def get_model_info(self) -> Dict[str, Any]:
        return {
            "name": self.name,
            "type": "mock",
            "parameters": "0"
        }
    
    def train(self):
        """Enable training mode."""
        self.training_mode = True
    
    def eval(self):
        """Enable evaluation mode."""
        self.training_mode = False


def load_training_data():
    """Load training data from processed files."""
    try:
        with open('data/processed/train_data.json', 'r') as f:
            train_data = json.load(f)
        with open('data/processed/val_data.json', 'r') as f:
            val_data = json.load(f)
        print(f"Loaded {len(train_data)} training examples and {len(val_data)} validation examples")
        return train_data, val_data
    except FileNotFoundError:
        print("No processed data found. Please run 01_data_preparation.py first.")
        return None, None


def initialize_components(config):
    """Initialize all RLVR components."""
    print("Initializing RLVR components...")
    
    # Initialize language model
    try:
        language_model = LanguageModel(config.model)
        print(f"✓ Initialized language model: {config.model.model_name}")
    except Exception as e:
        print(f"⚠ Could not initialize language model: {e}")
        print("Using mock language model for demonstration")
        language_model = MockLanguageModel(config.model)
    
    # Initialize verifiers
    verifiers = [
        CodeVerifier(config={
            "timeout": 30,
            "safe_mode": True,
            "allowed_modules": ["math", "random", "datetime", "collections"]
        }),
        MathVerifier(config={
            "tolerance": 1e-6,
            "max_steps": 10
        }),
        LogicVerifier(config={
            "reasoning_depth": 3,
            "consistency_check": True
        })
    ]
    print(f"✓ Initialized {len(verifiers)} verifiers")
    
    # Initialize reward function
    reward_function = HybridReward(config={
        "verification_weight": 0.7,
        "quality_weight": 0.2,
        "diversity_weight": 0.05,
        "efficiency_weight": 0.05
    })
    print("✓ Initialized hybrid reward function")
    
    return language_model, verifiers, reward_function


def create_training_datasets(train_data, val_data, config):
    """Create training and validation datasets."""
    print("Creating training datasets...")
    
    train_dataset = RLVRTrainingDataset(train_data, config)
    val_dataset = RLVRTrainingDataset(val_data, config)
    
    print(f"✓ Created training dataset: {len(train_dataset)} samples")
    print(f"✓ Created validation dataset: {len(val_dataset)} samples")
    
    return train_dataset, val_dataset


def run_training(language_model, verifiers, reward_function, train_dataset, val_dataset, config):
    """Run the RLVR training process."""
    print("\nStarting RLVR training...")
    
    # Initialize trainer
    trainer = RLVRTrainer(
        config=config,
        language_model=language_model,
        verifiers=verifiers,
        reward_function=reward_function
    )
    
    # Start training
    start_time = time.time()
    training_results = trainer.train(train_dataset)
    end_time = time.time()
    
    training_time = end_time - start_time
    print(f"Training completed in {training_time:.2f} seconds")
    
    return training_results, trainer


def analyze_training_results(training_results):
    """Analyze and display training results."""
    print("\n=== Training Results Analysis ===")
    
    metrics = TrainingMetrics()
    analysis = metrics.analyze_training_results(training_results)
    
    print(f"Total episodes: {analysis.get('total_episodes', 0)}")
    print(f"Average reward: {analysis.get('avg_reward', 0):.3f}")
    print(f"Final reward: {analysis.get('final_reward', 0):.3f}")
    print(f"Reward improvement: {analysis.get('reward_improvement', 0):.3f}")
    
    if 'episode_rewards' in training_results:
        rewards = training_results['episode_rewards']
        print(f"Reward statistics:")
        print(f"  Min: {min(rewards):.3f}")
        print(f"  Max: {max(rewards):.3f}")
        print(f"  Std: {np.std(rewards):.3f}")
    
    if 'verification_accuracy' in training_results:
        accuracies = training_results['verification_accuracy']
        print(f"Verification accuracy: {np.mean(accuracies):.3f}")
    
    return analysis


def save_training_results(training_results, analysis):
    """Save training results to files."""
    print("\nSaving training results...")
    
    results_dir = Path("results")
    results_dir.mkdir(exist_ok=True)
    
    # Save training results
    with open(results_dir / "training_results.json", 'w') as f:
        json.dump(training_results, f, indent=2, default=str)
    
    # Save analysis
    with open(results_dir / "training_analysis.json", 'w') as f:
        json.dump(analysis, f, indent=2, default=str)
    
    print("✓ Saved training results to results/")
    print("  - training_results.json")
    print("  - training_analysis.json")


def demonstrate_verification(language_model, verifiers, reward_function, test_examples):
    """Demonstrate verification on test examples."""
    print("\n=== Verification Demonstration ===")
    
    for i, example in enumerate(test_examples[:3]):
        print(f"\nExample {i+1}:")
        print(f"Instruction: {example['instruction']}")
        print(f"Expected: {example['expected_output'][:100]}...")
        
        # Generate model output
        model_output = language_model.generate(example['instruction'])
        print(f"Generated: {model_output[:100]}...")
        
        # Run verification
        verification_outputs = []
        for verifier in verifiers:
            try:
                result = verifier.verify(
                    instruction=example['instruction'],
                    model_output=model_output,
                    expected_output=example['expected_output']
                )
                verification_outputs.append(result)
                print(f"  {type(verifier).__name__}: {result.result} (score: {result.score:.3f})")
            except Exception as e:
                print(f"  {type(verifier).__name__}: Error - {e}")
        
        # Compute reward
        if verification_outputs:
            try:
                reward_output = reward_function.compute_reward(
                    instruction=example['instruction'],
                    model_output=model_output,
                    verification_outputs=verification_outputs
                )
                print(f"  Reward: {reward_output.reward:.3f}")
            except Exception as e:
                print(f"  Reward computation error: {e}")


def main():
    """Main training function."""
    print("=== RLVR Training Pipeline ===\n")
    
    # Setup logging
    logger = setup_logging(log_level="INFO")
    
    # Load configuration
    config = get_fast_config()
    print(f"Using configuration: {config.model.model_name}")
    print(f"Learning rate: {config.learning_rate}")
    print(f"Batch size: {config.batch_size}")
    print(f"Number of episodes: {config.num_episodes}")
    
    # Step 1: Load training data
    print("\n1. Loading training data...")
    train_data, val_data = load_training_data()
    if train_data is None:
        print("❌ Failed to load training data. Exiting.")
        return
    
    # Step 2: Initialize components
    print("\n2. Initializing components...")
    language_model, verifiers, reward_function = initialize_components(config)
    
    # Step 3: Create datasets
    print("\n3. Creating training datasets...")
    train_dataset, val_dataset = create_training_datasets(train_data, val_data, config)
    
    # Step 4: Demonstrate verification
    print("\n4. Demonstrating verification...")
    test_examples = train_data[:5]  # Use first 5 examples for demonstration
    demonstrate_verification(language_model, verifiers, reward_function, test_examples)
    
    # Step 5: Run training
    print("\n5. Running training...")
    training_results, trainer = run_training(
        language_model, verifiers, reward_function, 
        train_dataset, val_dataset, config
    )
    
    # Step 6: Analyze results
    print("\n6. Analyzing training results...")
    analysis = analyze_training_results(training_results)
    
    # Step 7: Save results
    print("\n7. Saving results...")
    save_training_results(training_results, analysis)
    
    # Summary
    print("\n=== Training Summary ===")
    print(f"✓ Loaded {len(train_data)} training examples")
    print(f"✓ Initialized {len(verifiers)} verifiers")
    print(f"✓ Trained for {config.num_episodes} episodes")
    print(f"✓ Achieved average reward: {analysis.get('avg_reward', 0):.3f}")
    print(f"✓ Saved results to results/")
    
    print("\nNext steps:")
    print("1. Run 04_evaluation.py to evaluate training results")
    print("2. Check results/ directory for detailed outputs")
    print("3. Modify configuration for different experiments")


if __name__ == "__main__":
    main() 