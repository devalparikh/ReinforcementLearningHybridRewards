"""
Reinforcement Learning with Verifiable Rewards (RLVR)

A production-grade implementation for training language models with objective,
verifiable feedback using reinforcement learning techniques.
"""

__version__ = "0.1.0"
__author__ = "Your Name"
__email__ = "your.email@example.com"

# Core imports for easy access
from .config.training_config import TrainingConfig
from .verifiers.base_verifier import BaseVerifier
from .rewards.base_reward import BaseReward
from .models.language_model import LanguageModel
from .training.rlvr_trainer import RLVRTrainer

__all__ = [
    "TrainingConfig",
    "BaseVerifier", 
    "BaseReward",
    "LanguageModel",
    "RLVRTrainer",
] 