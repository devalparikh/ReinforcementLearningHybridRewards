"""
Main RLVR trainer for Reinforcement Learning with Verifiable Rewards.

This module implements the main training loop for RLVR, orchestrating
model generation, verification, reward computation, and policy updates.
"""

import time
import traceback
from typing import Dict, Any, List, Optional, Union, Tuple
import logging
from pathlib import Path
import json
import torch
import numpy as np
from tqdm import tqdm
import wandb
from ..config.training_config import TrainingConfig
from ..models.language_model import LanguageModel
from ..verifiers.base_verifier import BaseVerifier, VerificationOutput
from ..rewards.base_reward import BaseReward, RewardOutput
from ..utils.logging import setup_logging, get_logger
from ..utils.metrics import MetricsTracker
from .ppo_trainer import PPOTrainer
from .experience_buffer import ExperienceBuffer
from .training_types import TrainingStep


class RLVRTrainer:
    """
    Main trainer for Reinforcement Learning with Verifiable Rewards.
    
    This trainer orchestrates the entire RLVR training process, including
    model generation, verification, reward computation, and policy updates.
    """
    
    def __init__(
        self,
        config: TrainingConfig,
        language_model: LanguageModel,
        verifiers: List[BaseVerifier],
        reward_function: BaseReward,
        logger: Optional[logging.Logger] = None
    ):
        """
        Initialize the RLVR trainer.
        
        Args:
            config: Training configuration
            language_model: Language model wrapper
            verifiers: List of verifiers
            reward_function: Reward function
            logger: Logger instance
        """
        self.config = config
        self.language_model = language_model
        self.verifiers = verifiers
        self.reward_function = reward_function
        self.logger = logger or get_logger(__name__)
        
        # Initialize components
        self.ppo_trainer = PPOTrainer(config, language_model, self.logger)
        self.experience_buffer = ExperienceBuffer(config.batch_size * 10)  # 10x batch size
        self.metrics_tracker = MetricsTracker()
        
        # Training state
        self.current_episode = 0
        self.best_reward = float('-inf')
        self.training_history = []
        
        # Setup logging
        self._setup_logging()
        
        self.logger.info("RLVR Trainer initialized successfully")
    
    def _setup_logging(self) -> None:
        """Setup logging and monitoring."""
        # Create output directories
        Path(self.config.output_dir).mkdir(parents=True, exist_ok=True)
        Path(self.config.checkpoint_dir).mkdir(parents=True, exist_ok=True)
        Path(self.config.log_dir).mkdir(parents=True, exist_ok=True)
        
        # Setup Weights & Biases if enabled
        if self.config.use_wandb:
            wandb.init(
                project="rlvr",
                name=self.config.experiment_name,
                config=self.config.dict()
            )
        
        # Setup TensorBoard if enabled
        if self.config.use_tensorboard:
            from torch.utils.tensorboard import SummaryWriter
            self.tensorboard_writer = SummaryWriter(log_dir=self.config.log_dir)
        else:
            self.tensorboard_writer = None
    
    def train(self, train_data: List[Dict[str, Any]], eval_data: Optional[List[Dict[str, Any]]] = None) -> None:
        """
        Main training loop.
        
        Args:
            train_data: Training data
            eval_data: Evaluation data (optional)
        """
        self.logger.info(f"Starting RLVR training for {self.config.num_episodes} episodes")
        
        try:
            for episode in range(self.config.num_episodes):
                self.current_episode = episode
                
                # Training step
                episode_metrics = self._train_episode(train_data)
                
                # Log metrics
                self._log_episode_metrics(episode_metrics)
                
                # Evaluation
                if eval_data and episode % self.config.eval_interval == 0:
                    eval_metrics = self._evaluate(eval_data)
                    self._log_evaluation_metrics(eval_metrics)
                
                # Save checkpoint
                if episode % self.config.save_interval == 0:
                    self._save_checkpoint(episode)
                
                # Early stopping check
                if self._should_stop_early():
                    self.logger.info("Early stopping triggered")
                    break
                
        except KeyboardInterrupt:
            self.logger.info("Training interrupted by user")
        except Exception as e:
            self.logger.error(f"Training failed: {e}")
            raise
        finally:
            self._cleanup()
    
    def _train_episode(self, train_data: List[Dict[str, Any]]) -> Dict[str, Any]:
        """
        Train for one episode.
        
        Args:
            train_data: Training data
            
        Returns:
            Episode metrics
        """
        episode_start_time = time.time()
        episode_steps = []
        
        # Sample batch of instructions
        batch_indices = np.random.choice(len(train_data), self.config.batch_size, replace=False)
        batch_data = [train_data[i] for i in batch_indices]
        
        for instruction_data in batch_data:
            instruction = instruction_data["instruction"]
            expected_output = instruction_data.get("expected_output")
            context = instruction_data.get("context", {})
            
            # Generate model output
            generation_output = self.language_model.generate(
                prompt=instruction,
                max_length=self.config.model.max_length,
                temperature=1.0,
                return_logprobs=True
            )
            
            # Verify outputs
            verification_outputs = []
            for verifier in self.verifiers:
                verification_output = verifier.verify(
                    instruction=instruction,
                    model_output=generation_output.text,
                    expected_output=expected_output,
                    context=context
                )
                verification_outputs.append(verification_output)
            
            # Compute reward
            reward_output = self.reward_function.compute_reward(
                instruction=instruction,
                model_output=generation_output.text,
                verification_outputs=verification_outputs,
                context=context
            )
            
            # Create training step
            training_step = TrainingStep(
                instruction=instruction,
                model_output=generation_output.text,
                verification_outputs=verification_outputs,
                reward_output=reward_output,
                logprobs=generation_output.logprobs,
                metadata={
                    "generation_time": generation_output.generation_time,
                    "expected_output": expected_output,
                    "context": context
                }
            )
            
            episode_steps.append(training_step)
        
        # Update policy using PPO
        policy_metrics = self.ppo_trainer.update_policy(episode_steps)
        
        # Compute episode metrics
        episode_metrics = self._compute_episode_metrics(episode_steps, policy_metrics)
        episode_metrics["episode_time"] = time.time() - episode_start_time
        
        return episode_metrics
    
    def _evaluate(self, eval_data: List[Dict[str, Any]]) -> Dict[str, Any]:
        """
        Evaluate the model on evaluation data.
        
        Args:
            eval_data: Evaluation data
            
        Returns:
            Evaluation metrics
        """
        self.logger.info("Starting evaluation...")
        
        eval_start_time = time.time()
        eval_steps = []
        
        # Use a subset for evaluation to save time
        eval_subset = eval_data[:min(len(eval_data), 100)]
        
        for instruction_data in tqdm(eval_subset, desc="Evaluating"):
            instruction = instruction_data["instruction"]
            expected_output = instruction_data.get("expected_output")
            context = instruction_data.get("context", {})
            
            # Generate model output
            generation_output = self.language_model.generate(
                prompt=instruction,
                max_length=self.config.model.max_length,
                temperature=0.7,  # Lower temperature for evaluation
                return_logprobs=True
            )
            
            # Verify outputs
            verification_outputs = []
            for verifier in self.verifiers:
                verification_output = verifier.verify(
                    instruction=instruction,
                    model_output=generation_output.text,
                    expected_output=expected_output,
                    context=context
                )
                verification_outputs.append(verification_output)
            
            # Compute reward
            reward_output = self.reward_function.compute_reward(
                instruction=instruction,
                model_output=generation_output.text,
                verification_outputs=verification_outputs,
                context=context
            )
            
            # Create evaluation step
            eval_step = TrainingStep(
                instruction=instruction,
                model_output=generation_output.text,
                verification_outputs=verification_outputs,
                reward_output=reward_output,
                logprobs=generation_output.logprobs,
                metadata={
                    "generation_time": generation_output.generation_time,
                    "expected_output": expected_output,
                    "context": context
                }
            )
            
            eval_steps.append(eval_step)
        
        # Compute evaluation metrics
        eval_metrics = self._compute_evaluation_metrics(eval_steps)
        eval_metrics["eval_time"] = time.time() - eval_start_time
        
        return eval_metrics
    
    def _compute_episode_metrics(self, episode_steps: List[TrainingStep], policy_metrics: Dict[str, Any]) -> Dict[str, Any]:
        """Compute metrics for an episode."""
        rewards = [step.reward_output.reward for step in episode_steps]
        confidences = [step.reward_output.confidence for step in episode_steps]
        
        # Verification metrics
        verification_results = []
        for step in episode_steps:
            for verification in step.verification_outputs:
                verification_results.append(verification.result.value)
        
        # Component metrics
        component_metrics = {}
        if episode_steps:
            all_components = set()
            for step in episode_steps:
                all_components.update(step.reward_output.components.keys())
            
            for component in all_components:
                component_values = [step.reward_output.components.get(component, 0.0) for step in episode_steps]
                component_metrics[f"{component}_mean"] = np.mean(component_values)
                component_metrics[f"{component}_std"] = np.std(component_values)
        
        return {
            "episode": self.current_episode,
            "reward_mean": np.mean(rewards),
            "reward_std": np.std(rewards),
            "reward_min": np.min(rewards),
            "reward_max": np.max(rewards),
            "confidence_mean": np.mean(confidences),
            "confidence_std": np.std(confidences),
            "verification_accuracy": verification_results.count("correct") / len(verification_results) if verification_results else 0.0,
            "verification_partial": verification_results.count("partial") / len(verification_results) if verification_results else 0.0,
            "verification_incorrect": verification_results.count("incorrect") / len(verification_results) if verification_results else 0.0,
            "verification_error": verification_results.count("error") / len(verification_results) if verification_results else 0.0,
            **component_metrics,
            **policy_metrics
        }
    
    def _compute_evaluation_metrics(self, eval_steps: List[TrainingStep]) -> Dict[str, Any]:
        """Compute metrics for evaluation."""
        rewards = [step.reward_output.reward for step in eval_steps]
        confidences = [step.reward_output.confidence for step in eval_steps]
        
        # Verification metrics
        verification_results = []
        for step in eval_steps:
            for verification in step.verification_outputs:
                verification_results.append(verification.result.value)
        
        # Per-verifier metrics
        verifier_metrics = {}
        for i, verifier in enumerate(self.verifiers):
            verifier_results = []
            for step in eval_steps:
                if i < len(step.verification_outputs):
                    verifier_results.append(step.verification_outputs[i].result.value)
            
            if verifier_results:
                verifier_metrics[f"verifier_{i}_accuracy"] = verifier_results.count("correct") / len(verifier_results)
                verifier_metrics[f"verifier_{i}_partial"] = verifier_results.count("partial") / len(verifier_results)
                verifier_metrics[f"verifier_{i}_incorrect"] = verifier_results.count("incorrect") / len(verifier_results)
                verifier_metrics[f"verifier_{i}_error"] = verifier_results.count("error") / len(verifier_results)
        
        return {
            "eval_reward_mean": np.mean(rewards),
            "eval_reward_std": np.std(rewards),
            "eval_confidence_mean": np.mean(confidences),
            "eval_verification_accuracy": verification_results.count("correct") / len(verification_results) if verification_results else 0.0,
            "eval_verification_partial": verification_results.count("partial") / len(verification_results) if verification_results else 0.0,
            "eval_verification_incorrect": verification_results.count("incorrect") / len(verification_results) if verification_results else 0.0,
            "eval_verification_error": verification_results.count("error") / len(verification_results) if verification_results else 0.0,
            **verifier_metrics
        }
    
    def _log_episode_metrics(self, metrics: Dict[str, Any]) -> None:
        """Log episode metrics."""
        # Update metrics tracker
        self.metrics_tracker.update(metrics)
        
        # Log to console
        self.logger.info(
            f"Episode {metrics['episode']}: "
            f"Reward={metrics['reward_mean']:.4f}±{metrics['reward_std']:.4f}, "
            f"Confidence={metrics['confidence_mean']:.4f}, "
            f"Verification Accuracy={metrics['verification_accuracy']:.4f}"
        )
        
        # Log to TensorBoard
        if self.tensorboard_writer:
            for key, value in metrics.items():
                if isinstance(value, (int, float)):
                    self.tensorboard_writer.add_scalar(f"episode/{key}", value, metrics['episode'])
        
        # Log to Weights & Biases
        if self.config.use_wandb:
            wandb.log(metrics, step=metrics['episode'])
        
        # Store in history
        self.training_history.append(metrics)
        
        # Update best reward
        if metrics['reward_mean'] > self.best_reward:
            self.best_reward = metrics['reward_mean']
    
    def _log_evaluation_metrics(self, metrics: Dict[str, Any]) -> None:
        """Log evaluation metrics."""
        self.logger.info(
            f"Evaluation: "
            f"Reward={metrics['eval_reward_mean']:.4f}±{metrics['eval_reward_std']:.4f}, "
            f"Verification Accuracy={metrics['eval_verification_accuracy']:.4f}"
        )
        
        # Log to TensorBoard
        if self.tensorboard_writer:
            for key, value in metrics.items():
                if isinstance(value, (int, float)):
                    self.tensorboard_writer.add_scalar(f"eval/{key}", value, self.current_episode)
        
        # Log to Weights & Biases
        if self.config.use_wandb:
            wandb.log(metrics, step=self.current_episode)
    
    def _save_checkpoint(self, episode: int) -> None:
        """Save training checkpoint."""
        checkpoint_path = Path(self.config.checkpoint_dir) / f"checkpoint_episode_{episode}.pt"
        
        checkpoint = {
            "episode": episode,
            "model_state": self.language_model.model.state_dict(),
            "optimizer_state": self.ppo_trainer.optimizer.state_dict(),
            "config": self.config.dict(),
            "best_reward": self.best_reward,
            "training_history": self.training_history
        }
        
        torch.save(checkpoint, checkpoint_path)
        self.logger.info(f"Checkpoint saved: {checkpoint_path}")
    
    def _should_stop_early(self) -> bool:
        """Check if training should stop early."""
        # Simple early stopping based on reward plateau
        if len(self.training_history) < 50:
            return False
        
        recent_rewards = [h['reward_mean'] for h in self.training_history[-50:]]
        if np.std(recent_rewards) < 0.01:  # Very small improvement
            return True
        
        return False
    
    def _cleanup(self) -> None:
        """Cleanup resources."""
        if self.tensorboard_writer:
            self.tensorboard_writer.close()
        
        if self.config.use_wandb:
            wandb.finish()
        
        # Save final model
        final_model_path = Path(self.config.output_dir) / "final_model"
        self.language_model.save_model(str(final_model_path))
        
        # Save training history
        history_path = Path(self.config.output_dir) / "training_history.json"
        with open(history_path, 'w') as f:
            json.dump(self.training_history, f, indent=2)
        
        self.logger.info("Training completed and resources cleaned up")
    
    def load_checkpoint(self, checkpoint_path: Union[str, Path]) -> None:
        """Load training checkpoint."""
        checkpoint_path = Path(checkpoint_path)
        
        if not checkpoint_path.exists():
            raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")
        
        checkpoint = torch.load(checkpoint_path, map_location='cpu')
        
        # Load model state
        self.language_model.model.load_state_dict(checkpoint["model_state"])
        
        # Load optimizer state
        self.ppo_trainer.optimizer.load_state_dict(checkpoint["optimizer_state"])
        
        # Load training state
        self.current_episode = checkpoint["episode"]
        self.best_reward = checkpoint["best_reward"]
        self.training_history = checkpoint["training_history"]
        
        self.logger.info(f"Checkpoint loaded from: {checkpoint_path}")
    
    def get_training_summary(self) -> Dict[str, Any]:
        """Get training summary."""
        if not self.training_history:
            return {}
        
        recent_metrics = self.training_history[-100:]  # Last 100 episodes
        
        return {
            "total_episodes": len(self.training_history),
            "best_reward": self.best_reward,
            "final_reward_mean": np.mean([h['reward_mean'] for h in recent_metrics]),
            "final_reward_std": np.std([h['reward_mean'] for h in recent_metrics]),
            "final_verification_accuracy": np.mean([h['verification_accuracy'] for h in recent_metrics]),
            "training_time": sum(h.get('episode_time', 0) for h in self.training_history)
        } 