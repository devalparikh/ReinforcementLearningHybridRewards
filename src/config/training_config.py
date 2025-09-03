"""
Training configuration for RLVR.

This module provides a centralized configuration system for all RLVR training
parameters, with validation and type safety using Pydantic.
"""

from typing import Dict, List, Optional, Union, Any
from dataclasses import dataclass
from pathlib import Path
import yaml
from pydantic import BaseModel, Field, validator


class ModelConfig(BaseModel):
    """Configuration for the language model."""
    
    model_name: str = Field(
        default="gpt2-medium",
        description="HuggingFace model name or path"
    )
    tokenizer_name: Optional[str] = Field(
        default=None,
        description="Tokenizer name (if different from model_name)"
    )
    max_length: int = Field(
        default=512,
        ge=1,
        description="Maximum sequence length"
    )
    device: str = Field(
        default="auto",
        description="Device to use (auto, cpu, cuda, mps)"
    )
    dtype: str = Field(
        default="auto",
        description="Data type (auto, float32, float16, bfloat16)"
    )
    load_in_8bit: bool = Field(
        default=False,
        description="Whether to load model in 8-bit precision"
    )
    load_in_4bit: bool = Field(
        default=False,
        description="Whether to load model in 4-bit precision"
    )
    
    @validator('device')
    def validate_device(cls, v):
        """Validate device specification."""
        valid_devices = ['auto', 'cpu', 'cuda', 'mps']
        if v not in valid_devices:
            raise ValueError(f"Device must be one of {valid_devices}")
        return v
    
    @validator('dtype')
    def validate_dtype(cls, v):
        """Validate data type specification."""
        valid_dtypes = ['auto', 'float32', 'float16', 'bfloat16']
        if v not in valid_dtypes:
            raise ValueError(f"Data type must be one of {valid_dtypes}")
        return v


class TrainingConfig(BaseModel):
    """Configuration for RLVR training."""
    
    # Model configuration
    model: ModelConfig = Field(default_factory=ModelConfig)
    
    # Training hyperparameters
    learning_rate: float = Field(
        default=1e-5,
        gt=0,
        description="Learning rate for optimization"
    )
    batch_size: int = Field(
        default=4,
        ge=1,
        description="Training batch size"
    )
    gradient_accumulation_steps: int = Field(
        default=1,
        ge=1,
        description="Number of gradient accumulation steps"
    )
    max_grad_norm: float = Field(
        default=1.0,
        gt=0,
        description="Maximum gradient norm for clipping"
    )
    
    # PPO specific parameters
    ppo_epochs: int = Field(
        default=4,
        ge=1,
        description="Number of PPO epochs per update"
    )
    clip_epsilon: float = Field(
        default=0.2,
        gt=0,
        description="PPO clip epsilon parameter"
    )
    value_coef: float = Field(
        default=0.5,
        description="Value function coefficient"
    )
    entropy_coef: float = Field(
        default=0.01,
        description="Entropy coefficient for exploration"
    )
    gamma: float = Field(
        default=1.0,
        ge=0,
        le=1,
        description="Discount factor"
    )
    gae_lambda: float = Field(
        default=0.95,
        ge=0,
        le=1,
        description="GAE lambda parameter"
    )
    
    # Training duration
    num_episodes: int = Field(
        default=1000,
        ge=1,
        description="Total number of training episodes"
    )
    eval_interval: int = Field(
        default=100,
        ge=1,
        description="Evaluation interval in episodes"
    )
    save_interval: int = Field(
        default=500,
        ge=1,
        description="Model save interval in episodes"
    )
    
    # Data configuration
    train_data_path: str = Field(
        default="data/train.jsonl",
        description="Path to training data"
    )
    eval_data_path: str = Field(
        default="data/eval.jsonl",
        description="Path to evaluation data"
    )
    
    # Verification and reward configuration
    verifier_config: Dict[str, Any] = Field(
        default_factory=dict,
        description="Configuration for verifiers"
    )
    reward_config: Dict[str, Any] = Field(
        default_factory=dict,
        description="Configuration for reward functions"
    )
    
    # Logging and monitoring
    log_dir: str = Field(
        default="logs",
        description="Directory for logging"
    )
    experiment_name: str = Field(
        default="rlvr_experiment",
        description="Name of the experiment"
    )
    use_wandb: bool = Field(
        default=False,
        description="Whether to use Weights & Biases logging"
    )
    use_tensorboard: bool = Field(
        default=True,
        description="Whether to use TensorBoard logging"
    )
    
    # Output configuration
    output_dir: str = Field(
        default="outputs",
        description="Directory for model outputs"
    )
    checkpoint_dir: str = Field(
        default="checkpoints",
        description="Directory for model checkpoints"
    )
    
    # Remove the validators since they're validating non-existent fields
    # The validation should be done in ModelConfig instead
    
    def save(self, path: Union[str, Path]) -> None:
        """Save configuration to YAML file."""
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        
        with open(path, 'w') as f:
            yaml.dump(self.dict(), f, default_flow_style=False, indent=2)
    
    @classmethod
    def load(cls, path: Union[str, Path]) -> 'TrainingConfig':
        """Load configuration from YAML file."""
        path = Path(path)
        
        if not path.exists():
            raise FileNotFoundError(f"Configuration file not found: {path}")
        
        with open(path, 'r') as f:
            config_dict = yaml.safe_load(f)
        
        return cls(**config_dict)
    
    def get_device(self) -> str:
        """Get the actual device to use."""
        if self.model.device == "auto":
            import torch
            if torch.cuda.is_available():
                return "cuda"
            elif hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
                return "mps"
            else:
                return "cpu"
        return self.model.device
    
    def get_dtype(self) -> str:
        """Get the actual data type to use."""
        if self.model.dtype == "auto":
            import torch
            if torch.cuda.is_available():
                return "float16"
            else:
                return "float32"
        return self.model.dtype


# Default configuration
DEFAULT_CONFIG = TrainingConfig()

# Example configurations for different scenarios
def get_fast_config() -> TrainingConfig:
    """Get a configuration optimized for fast experimentation."""
    config = TrainingConfig()
    config.model.model_name = "gpt2"
    config.batch_size = 2
    config.num_episodes = 100
    config.eval_interval = 20
    config.save_interval = 50
    return config

def get_production_config() -> TrainingConfig:
    """Get a configuration optimized for production training."""
    config = TrainingConfig()
    config.model.model_name = "gpt2-large"
    config.batch_size = 8
    config.learning_rate = 5e-6
    config.num_episodes = 10000
    config.use_wandb = True
    return config 