#!/usr/bin/env python3
"""
End-to-End RLVR Example

This script demonstrates the complete Reinforcement Learning with Verifiable Rewards (RLVR) pipeline:
1. Data generation and preparation
2. Verifier testing and validation
3. RLVR training with hybrid rewards
4. Evaluation and results analysis

Usage:
    python example_end_to_end.py
"""

import sys
import os
import json
import time
import random
from typing import List, Dict, Any
from dataclasses import dataclass
import numpy as np
import pandas as pd
from pathlib import Path

# Add src to path
sys.path.append('src')

from src.config.training_config import TrainingConfig, get_fast_config
from src.data.dataset import RLVRDataset
from src.data.preprocessing import DataPreprocessor
from src.models.language_model import LanguageModel
from src.verifiers.code_verifier import CodeVerifier
from src.verifiers.math_verifier import MathVerifier
from src.verifiers.logic_verifier import LogicVerifier
from src.rewards.hybrid_reward import HybridReward
from src.rewards.reward_factory import RewardFactory
from src.training.rlvr_trainer import RLVRTrainer
from src.training.ppo_trainer import PPOTrainer
from src.utils.logging import setup_logging
from src.utils.metrics import MetricsTracker


@dataclass
class TrainingExample:
    """Represents a training example for RLVR."""
    instruction: str
    expected_output: str
    task_type: str
    difficulty: str
    context: Dict[str, Any] = None


class RLVRExample:
    """Complete RLVR example implementation."""
    
    def __init__(self, config: TrainingConfig = None):
        self.config = config or get_fast_config()
        self.logger = setup_logging(log_level="INFO")
        self.results_dir = Path("results")
        self.results_dir.mkdir(exist_ok=True)
        
        # Initialize components
        self.language_model = None
        self.verifiers = []
        self.reward_function = None
        self.trainer = None
        
        self.logger.info("Initialized RLVR Example")
    
    def generate_synthetic_data(self, num_samples: int = 500) -> List[TrainingExample]:
        """Generate synthetic training data."""
        self.logger.info(f"Generating {num_samples} synthetic training examples...")
        
        # Define task templates
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
            },
            {
                "instruction": "What is the square root of 144?",
                "expected_output": "√144 = 12",
                "task_type": "math_reasoning",
                "difficulty": "easy"
            }
        ]
        
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
            },
            {
                "instruction": "If all A are B and all B are C, can we conclude that all A are C?",
                "expected_output": "Yes, this is a valid syllogism. If all A are B and all B are C, then all A must be C.",
                "task_type": "logic_reasoning",
                "difficulty": "medium"
            }
        ]
        
        all_tasks = code_tasks + math_tasks + logic_tasks
        
        # Generate synthetic data
        synthetic_data = []
        for i in range(num_samples):
            task = random.choice(all_tasks)
            
            # Add variation to instructions
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
        
        self.logger.info(f"Generated {len(synthetic_data)} training examples")
        return synthetic_data
    
    def validate_data_quality(self, data: List[TrainingExample]) -> Dict[str, Any]:
        """Validate and analyze data quality."""
        self.logger.info("Validating data quality...")
        
        metrics = MetricsTracker()
        
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
        
        # Analyze dataset statistics
        quality_report = {
            "total_samples": len(data_dicts),
            "task_types": {},
            "difficulties": {},
            "instruction_lengths": [],
            "output_lengths": []
        }
        
        for item in data_dicts:
            # Count task types
            task_type = item.get("task_type", "unknown")
            quality_report["task_types"][task_type] = quality_report["task_types"].get(task_type, 0) + 1
            
            # Count difficulties
            difficulty = item.get("difficulty", "unknown")
            quality_report["difficulties"][difficulty] = quality_report["difficulties"].get(difficulty, 0) + 1
            
            # Track lengths
            quality_report["instruction_lengths"].append(len(item.get("instruction", "")))
            quality_report["output_lengths"].append(len(item.get("expected_output", "")))
        
        # Calculate quality score
        avg_instruction_length = np.mean(quality_report["instruction_lengths"])
        avg_output_length = np.mean(quality_report["output_lengths"])
        task_type_diversity = len(quality_report["task_types"])
        difficulty_diversity = len(quality_report["difficulties"])
        
        quality_score = min(1.0, (
            (avg_instruction_length / 100) * 0.3 +
            (avg_output_length / 200) * 0.3 +
            (task_type_diversity / 5) * 0.2 +
            (difficulty_diversity / 3) * 0.2
        ))
        
        quality_report["quality_score"] = quality_score
        quality_report["avg_instruction_length"] = avg_instruction_length
        quality_report["avg_output_length"] = avg_output_length
        quality_report["task_type_diversity"] = task_type_diversity
        quality_report["difficulty_diversity"] = difficulty_diversity
        
        self.logger.info(f"Data quality score: {quality_report['quality_score']:.2f}/1.0")
        return quality_report
    
    def initialize_verifiers(self):
        """Initialize verification components."""
        self.logger.info("Initializing verifiers...")
        
        self.verifiers = [
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
        
        self.logger.info(f"Initialized {len(self.verifiers)} verifiers")
    
    def test_verifiers(self, test_data: List[TrainingExample]) -> Dict[str, Any]:
        """Test verifiers on sample data."""
        self.logger.info("Testing verifiers...")
        
        verification_results = {}
        
        for verifier in self.verifiers:
            verifier_name = type(verifier).__name__
            self.logger.info(f"Testing {verifier_name}...")
            
            results = []
            for example in test_data[:20]:  # Test on subset
                try:
                    result = verifier.verify(
                        instruction=example.instruction,
                        model_output=example.expected_output,  # Use expected as model output for testing
                        expected_output=example.expected_output
                    )
                    results.append({
                        "success": result.result.value == "correct",
                        "score": result.score,
                        "details": result.details
                    })
                except Exception as e:
                    results.append({
                        "success": False,
                        "score": 0.0,
                        "error": str(e)
                    })
            
            success_rate = sum(1 for r in results if r["success"]) / len(results)
            avg_score = np.mean([r["score"] for r in results if "score" in r])
            
            verification_results[verifier_name] = {
                "success_rate": success_rate,
                "avg_score": avg_score,
                "results": results
            }
            
            self.logger.info(f"{verifier_name}: {success_rate:.2%} success rate, {avg_score:.3f} avg score")
        
        return verification_results
    
    def initialize_reward_function(self):
        """Initialize the hybrid reward function."""
        self.logger.info("Initializing reward function...")
        
        self.reward_function = HybridReward(config={
            "verification_weight": 0.7,
            "quality_weight": 0.2,
            "diversity_weight": 0.05,
            "efficiency_weight": 0.05
        })
        
        self.logger.info("Initialized hybrid reward function")
    
    def initialize_language_model(self):
        """Initialize the language model."""
        self.logger.info("Initializing language model...")
        
        try:
            self.language_model = LanguageModel(self.config.model)
            self.logger.info(f"Initialized language model: {self.config.model.model_name}")
        except Exception as e:
            self.logger.warning(f"Could not initialize language model: {e}")
            self.logger.info("Using mock language model for demonstration")
            # Create a simple mock model for demonstration
            self.language_model = MockLanguageModel(self.config.model)
    
    def prepare_training_data(self, data: List[TrainingExample]) -> RLVRDataset:
        """Prepare data for training."""
        self.logger.info("Preparing training data...")
        
        # Convert to dict format
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
        
        # Preprocess data
        preprocessor = DataPreprocessor()
        processed_data = preprocessor.preprocess_dataset(data_dicts)
        
        # Create dataset - save to temporary file first
        import tempfile
        import json
        
        with tempfile.NamedTemporaryFile(mode='w', suffix='.jsonl', delete=False) as f:
            for item in processed_data:
                f.write(json.dumps(item) + '\n')
            temp_file = f.name
        
        # Create dataset
        dataset = RLVRDataset(temp_file, self.logger)
        
        self.logger.info(f"Prepared {len(dataset)} training examples")
        return dataset
    
    def train(self, dataset: RLVRDataset) -> Dict[str, Any]:
        """Run the RLVR training process."""
        self.logger.info("Starting RLVR training...")
        
        # Convert dataset to list format for training
        train_data = []
        for i in range(len(dataset)):
            train_data.append(dataset[i])
        
        # Initialize trainer
        self.trainer = RLVRTrainer(
            config=self.config,
            language_model=self.language_model,
            verifiers=self.verifiers,
            reward_function=self.reward_function
        )
        
        # Start training
        self.trainer.train(train_data)
        
        # Get training summary
        training_results = self.trainer.get_training_summary()
        
        self.logger.info("Training completed")
        return training_results
    
    def evaluate_results(self, training_results: Dict[str, Any]) -> Dict[str, Any]:
        """Evaluate training results."""
        self.logger.info("Evaluating training results...")
        
        # Analyze training metrics
        metrics = MetricsTracker()
        
        # Create evaluation results
        evaluation_results = {
            "training_summary": training_results,
            "performance_metrics": {
                "total_episodes": training_results.get("total_episodes", 0),
                "best_reward": training_results.get("best_reward", 0.0),
                "final_reward_mean": training_results.get("final_reward_mean", 0.0),
                "final_verification_accuracy": training_results.get("final_verification_accuracy", 0.0),
                "training_time": training_results.get("training_time", 0.0)
            }
        }
        
        # Save results
        results_file = self.results_dir / "training_results.json"
        with open(results_file, 'w') as f:
            json.dump(training_results, f, indent=2, default=str)
        
        evaluation_file = self.results_dir / "evaluation_results.json"
        with open(evaluation_file, 'w') as f:
            json.dump(evaluation_results, f, indent=2, default=str)
        
        self.logger.info(f"Results saved to {self.results_dir}")
        return evaluation_results
    
    def run_complete_pipeline(self) -> Dict[str, Any]:
        """Run the complete RLVR pipeline."""
        self.logger.info("=== Starting Complete RLVR Pipeline ===")
        
        pipeline_results = {}
        
        # Step 1: Generate data
        data = self.generate_synthetic_data(num_samples=200)  # Smaller for demo
        pipeline_results["data_generation"] = {"num_samples": len(data)}
        
        # Step 2: Validate data quality
        quality_report = self.validate_data_quality(data)
        pipeline_results["data_quality"] = quality_report
        
        # Step 3: Initialize components
        self.initialize_verifiers()
        self.initialize_reward_function()
        self.initialize_language_model()
        
        # Step 4: Test verifiers
        verification_results = self.test_verifiers(data)
        pipeline_results["verification_testing"] = verification_results
        
        # Step 5: Prepare training data
        dataset = self.prepare_training_data(data)
        pipeline_results["data_preparation"] = {"dataset_size": len(dataset)}
        
        # Step 6: Train
        training_results = self.train(dataset)
        pipeline_results["training"] = training_results
        
        # Step 7: Evaluate
        evaluation_results = self.evaluate_results(training_results)
        pipeline_results["evaluation"] = evaluation_results
        
        # Save complete pipeline results
        pipeline_file = self.results_dir / "pipeline_results.json"
        with open(pipeline_file, 'w') as f:
            json.dump(pipeline_results, f, indent=2, default=str)
        
        self.logger.info("=== Complete RLVR Pipeline Finished ===")
        return pipeline_results


class MockLanguageModel:
    """Mock language model for demonstration purposes."""
    
    def __init__(self, config):
        self.config = config
        self.name = "mock_model"
        self.model = None  # Mock model attribute
    
    def generate(self, prompt: str, max_length: int = None, temperature: float = 1.0, return_logprobs: bool = False, **kwargs):
        """Generate a mock response."""
        from src.models.language_model import GenerationOutput
        
        # Simple mock responses based on instruction type
        if "factorial" in prompt.lower():
            text = "def factorial(n):\n    if n <= 1:\n        return 1\n    return n * factorial(n-1)"
        elif "percentage" in prompt.lower() or "15%" in prompt:
            text = "15% of 200 = 0.15 × 200 = 30"
        elif "palindrome" in prompt.lower():
            text = "def is_palindrome(s):\n    return s == s[::-1]"
        else:
            text = "This is a mock response for demonstration purposes."
        
        return GenerationOutput(
            text=text,
            logprobs=[0.1] * len(text.split()) if return_logprobs else None,
            tokens=text.split() if return_logprobs else None,
            scores=[0.1] * len(text.split()) if return_logprobs else None,
            generation_time=0.1,
            metadata={"model_type": "mock"}
        )
    
    def get_model_info(self) -> Dict[str, Any]:
        return {
            "name": self.name,
            "type": "mock",
            "parameters": "0"
        }
    
    def save_model(self, path: str) -> None:
        """Mock save model."""
        pass


def main():
    """Main function to run the complete RLVR example."""
    print("=== RLVR End-to-End Example ===\n")
    
    # Create and run RLVR example
    rlvr_example = RLVRExample()
    
    try:
        results = rlvr_example.run_complete_pipeline()
        
        print("\n=== Pipeline Results Summary ===")
        print(f"Data generation: {results['data_generation']['num_samples']} samples")
        print(f"Data quality score: {results['data_quality']['quality_score']:.2f}/1.0")
        print(f"Verification testing completed for {len(results['verification_testing'])} verifiers")
        print(f"Training completed with {len(results['training'])} episodes")
        print(f"Results saved to: {rlvr_example.results_dir}")
        
        print("\n=== Verification Performance ===")
        for verifier_name, verifier_results in results['verification_testing'].items():
            print(f"{verifier_name}: {verifier_results['success_rate']:.2%} success rate")
        
        print("\n=== Training Performance ===")
        if 'episode_rewards' in results['training']:
            rewards = results['training']['episode_rewards']
            print(f"Average reward: {np.mean(rewards):.3f}")
            print(f"Final reward: {rewards[-1]:.3f}")
        
        print("\n=== Example Completed Successfully ===")
        
    except Exception as e:
        print(f"Error running RLVR example: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main() 