"""
Experience buffer for RLVR.

This module implements an experience replay buffer for storing and sampling
training experiences during RLVR training.
"""

from typing import List, Dict, Any, Optional
from collections import deque
import random
import numpy as np
from dataclasses import dataclass


@dataclass
class Experience:
    """A single training experience."""
    instruction: str
    model_output: str
    reward: float
    logprobs: Optional[List[float]] = None
    metadata: Optional[Dict[str, Any]] = None


class ExperienceBuffer:
    """
    Experience replay buffer for RLVR training.
    
    This buffer stores training experiences and provides methods for sampling
    them during training.
    """
    
    def __init__(self, max_size: int = 10000):
        """
        Initialize experience buffer.
        
        Args:
            max_size: Maximum number of experiences to store
        """
        self.max_size = max_size
        self.buffer = deque(maxlen=max_size)
        self.total_experiences = 0
    
    def add_experience(self, experience: Experience) -> None:
        """
        Add an experience to the buffer.
        
        Args:
            experience: Experience to add
        """
        self.buffer.append(experience)
        self.total_experiences += 1
    
    def add_batch(self, experiences: List[Experience]) -> None:
        """
        Add multiple experiences to the buffer.
        
        Args:
            experiences: List of experiences to add
        """
        for experience in experiences:
            self.add_experience(experience)
    
    def sample(self, batch_size: int) -> List[Experience]:
        """
        Sample a batch of experiences from the buffer.
        
        Args:
            batch_size: Number of experiences to sample
            
        Returns:
            List of sampled experiences
        """
        if len(self.buffer) < batch_size:
            return list(self.buffer)
        
        return random.sample(self.buffer, batch_size)
    
    def sample_by_reward(self, batch_size: int, high_reward_threshold: float = 0.7) -> List[Experience]:
        """
        Sample experiences with preference for high-reward experiences.
        
        Args:
            batch_size: Number of experiences to sample
            high_reward_threshold: Threshold for high-reward experiences
            
        Returns:
            List of sampled experiences
        """
        if len(self.buffer) < batch_size:
            return list(self.buffer)
        
        # Separate high and low reward experiences
        high_reward_experiences = [exp for exp in self.buffer if exp.reward >= high_reward_threshold]
        low_reward_experiences = [exp for exp in self.buffer if exp.reward < high_reward_threshold]
        
        # Sample with preference for high-reward experiences
        high_reward_sample_size = min(batch_size // 2, len(high_reward_experiences))
        low_reward_sample_size = batch_size - high_reward_sample_size
        
        sampled_experiences = []
        
        if high_reward_experiences:
            sampled_experiences.extend(random.sample(high_reward_experiences, high_reward_sample_size))
        
        if low_reward_experiences and low_reward_sample_size > 0:
            sampled_experiences.extend(random.sample(low_reward_experiences, low_reward_sample_size))
        
        # If we don't have enough experiences, sample from the remaining
        if len(sampled_experiences) < batch_size:
            remaining_experiences = [exp for exp in self.buffer if exp not in sampled_experiences]
            additional_needed = batch_size - len(sampled_experiences)
            if remaining_experiences:
                sampled_experiences.extend(random.sample(remaining_experiences, min(additional_needed, len(remaining_experiences))))
        
        return sampled_experiences
    
    def get_statistics(self) -> Dict[str, Any]:
        """
        Get statistics about the experiences in the buffer.
        
        Returns:
            Dictionary containing buffer statistics
        """
        if not self.buffer:
            return {"total_experiences": 0, "buffer_size": 0}
        
        rewards = [exp.reward for exp in self.buffer]
        
        return {
            "total_experiences": self.total_experiences,
            "buffer_size": len(self.buffer),
            "reward_mean": np.mean(rewards),
            "reward_std": np.std(rewards),
            "reward_min": np.min(rewards),
            "reward_max": np.max(rewards),
            "reward_median": np.median(rewards)
        }
    
    def clear(self) -> None:
        """Clear all experiences from the buffer."""
        self.buffer.clear()
    
    def __len__(self) -> int:
        """Return the number of experiences in the buffer."""
        return len(self.buffer) 