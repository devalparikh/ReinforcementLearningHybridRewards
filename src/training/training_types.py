"""
Training types for RLVR.

This module contains shared data structures used across training modules
to avoid circular imports.
"""

from dataclasses import dataclass
from typing import Dict, Any, List, Optional

from ..verifiers.base_verifier import VerificationOutput
from ..rewards.base_reward import RewardOutput


@dataclass
class TrainingStep:
    """Data structure for a single training step."""
    
    instruction: str
    model_output: str
    verification_outputs: List[VerificationOutput]
    reward_output: RewardOutput
    logprobs: Optional[List[float]] = None
    value: Optional[float] = None
    metadata: Optional[Dict[str, Any]] = None 