"""
Reward factory for RLVR.

This module provides a factory pattern for creating and managing different
types of reward functions. It allows easy configuration and instantiation
of reward functions based on their type and configuration.
"""

from typing import Dict, Any, List, Optional, Union, Type
import logging
from pathlib import Path
import yaml
import json

from .base_reward import BaseReward, RewardType
from .hybrid_reward import HybridReward


class RewardFactory:
    """
    Factory for creating and managing reward functions.
    
    This factory provides a centralized way to create different types of
    reward functions with their configurations. It supports loading
    configurations from files and provides easy access to reward functions.
    """
    
    def __init__(self, logger: Optional[logging.Logger] = None):
        """
        Initialize the reward factory.
        
        Args:
            logger: Logger instance
        """
        self.logger = logger or logging.getLogger(__name__)
        self._reward_registry: Dict[str, Type[BaseReward]] = {}
        self._config_cache: Dict[str, Dict[str, Any]] = {}
        
        # Register default reward functions
        self._register_default_rewards()
    
    def _register_default_rewards(self) -> None:
        """Register default reward functions."""
        self.register_reward("hybrid", HybridReward)
    
    def register_reward(self, name: str, reward_class: Type[BaseReward]) -> None:
        """
        Register a reward function class.
        
        Args:
            name: Name for the reward function
            reward_class: The reward function class to register
        """
        if not issubclass(reward_class, BaseReward):
            raise ValueError(f"Reward class must inherit from BaseReward")
        
        self._reward_registry[name] = reward_class
        self.logger.info(f"Registered reward function: {name}")
    
    def create_reward(
        self,
        reward_type: str,
        name: Optional[str] = None,
        config: Optional[Dict[str, Any]] = None,
        config_file: Optional[Union[str, Path]] = None
    ) -> BaseReward:
        """
        Create a reward function instance.
        
        Args:
            reward_type: Type of reward function to create
            name: Name for the reward function instance
            config: Configuration dictionary
            config_file: Path to configuration file
            
        Returns:
            Reward function instance
            
        Raises:
            ValueError: If reward type is not registered
            FileNotFoundError: If config file is not found
        """
        if reward_type not in self._reward_registry:
            available_types = list(self._reward_registry.keys())
            raise ValueError(f"Unknown reward type '{reward_type}'. Available types: {available_types}")
        
        # Load configuration from file if provided
        if config_file:
            config = self.load_config(config_file)
        
        # Use default name if not provided
        if name is None:
            name = f"{reward_type}_reward"
        
        # Get the reward class
        reward_class = self._reward_registry[reward_type]
        
        # Create the reward instance
        reward = reward_class(name=name, config=config, logger=self.logger)
        
        self.logger.info(f"Created reward function: {name} (type: {reward_type})")
        return reward
    
    def create_hybrid_reward(
        self,
        name: Optional[str] = None,
        config: Optional[Dict[str, Any]] = None,
        config_file: Optional[Union[str, Path]] = None
    ) -> HybridReward:
        """
        Create a hybrid reward function instance.
        
        Args:
            name: Name for the reward function instance
            config: Configuration dictionary
            config_file: Path to configuration file
            
        Returns:
            Hybrid reward function instance
        """
        reward = self.create_reward("hybrid", name, config, config_file)
        return reward
    
    def load_config(self, config_file: Union[str, Path]) -> Dict[str, Any]:
        """
        Load configuration from a file.
        
        Args:
            config_file: Path to the configuration file
            
        Returns:
            Configuration dictionary
            
        Raises:
            FileNotFoundError: If config file is not found
            ValueError: If config file format is not supported
        """
        config_file = Path(config_file)
        
        if not config_file.exists():
            raise FileNotFoundError(f"Configuration file not found: {config_file}")
        
        # Check if config is already cached
        cache_key = str(config_file.absolute())
        if cache_key in self._config_cache:
            return self._config_cache[cache_key]
        
        # Load configuration based on file extension
        if config_file.suffix.lower() in ['.yaml', '.yml']:
            with open(config_file, 'r') as f:
                config = yaml.safe_load(f)
        elif config_file.suffix.lower() == '.json':
            with open(config_file, 'r') as f:
                config = json.load(f)
        else:
            raise ValueError(f"Unsupported configuration file format: {config_file.suffix}")
        
        # Cache the configuration
        self._config_cache[cache_key] = config
        
        self.logger.info(f"Loaded configuration from: {config_file}")
        return config
    
    def save_config(self, config: Dict[str, Any], config_file: Union[str, Path]) -> None:
        """
        Save configuration to a file.
        
        Args:
            config: Configuration dictionary to save
            config_file: Path to save the configuration
        """
        config_file = Path(config_file)
        config_file.parent.mkdir(parents=True, exist_ok=True)
        
        # Save configuration based on file extension
        if config_file.suffix.lower() in ['.yaml', '.yml']:
            with open(config_file, 'w') as f:
                yaml.dump(config, f, default_flow_style=False, indent=2)
        elif config_file.suffix.lower() == '.json':
            with open(config_file, 'w') as f:
                json.dump(config, f, indent=2)
        else:
            raise ValueError(f"Unsupported configuration file format: {config_file.suffix}")
        
        self.logger.info(f"Saved configuration to: {config_file}")
    
    def get_available_rewards(self) -> List[str]:
        """
        Get list of available reward function types.
        
        Returns:
            List of available reward function names
        """
        return list(self._reward_registry.keys())
    
    def get_reward_info(self, reward_type: str) -> Dict[str, Any]:
        """
        Get information about a reward function type.
        
        Args:
            reward_type: Type of reward function
            
        Returns:
            Dictionary containing reward function information
            
        Raises:
            ValueError: If reward type is not registered
        """
        if reward_type not in self._reward_registry:
            raise ValueError(f"Unknown reward type: {reward_type}")
        
        reward_class = self._reward_registry[reward_type]
        
        return {
            "name": reward_type,
            "class": reward_class.__name__,
            "module": reward_class.__module__,
            "description": getattr(reward_class, '__doc__', ''),
            "reward_type": getattr(reward_class, 'reward_type', RewardType.CUSTOM)
        }
    
    def create_reward_from_preset(self, preset_name: str) -> BaseReward:
        """
        Create a reward function from a preset configuration.
        
        Args:
            preset_name: Name of the preset
            
        Returns:
            Reward function instance
            
        Raises:
            ValueError: If preset is not found
        """
        presets = self._get_presets()
        
        if preset_name not in presets:
            available_presets = list(presets.keys())
            raise ValueError(f"Unknown preset '{preset_name}'. Available presets: {available_presets}")
        
        preset = presets[preset_name]
        return self.create_reward(
            reward_type=preset["type"],
            name=preset.get("name", f"{preset_name}_reward"),
            config=preset.get("config", {})
        )
    
    def _get_presets(self) -> Dict[str, Dict[str, Any]]:
        """
        Get predefined reward function presets.
        
        Returns:
            Dictionary of preset configurations
        """
        return {
            "default_hybrid": {
                "type": "hybrid",
                "name": "default_hybrid_reward",
                "config": {
                    "verification_weight": 0.7,
                    "quality_weight": 0.2,
                    "diversity_weight": 0.05,
                    "efficiency_weight": 0.05,
                    "correct_score": 1.0,
                    "partial_score": 0.7,
                    "incorrect_score": 0.0,
                    "error_score": -0.1
                }
            },
            "verification_focused": {
                "type": "hybrid",
                "name": "verification_focused_reward",
                "config": {
                    "verification_weight": 0.9,
                    "quality_weight": 0.05,
                    "diversity_weight": 0.025,
                    "efficiency_weight": 0.025,
                    "correct_score": 1.0,
                    "partial_score": 0.5,
                    "incorrect_score": -0.5,
                    "error_score": -0.2
                }
            },
            "quality_focused": {
                "type": "hybrid",
                "name": "quality_focused_reward",
                "config": {
                    "verification_weight": 0.4,
                    "quality_weight": 0.5,
                    "diversity_weight": 0.05,
                    "efficiency_weight": 0.05,
                    "correct_score": 1.0,
                    "partial_score": 0.8,
                    "incorrect_score": 0.0,
                    "error_score": -0.1,
                    "length_penalty": 0.2,
                    "repetition_penalty": 0.3,
                    "coherence_bonus": 0.2
                }
            },
            "balanced": {
                "type": "hybrid",
                "name": "balanced_reward",
                "config": {
                    "verification_weight": 0.5,
                    "quality_weight": 0.3,
                    "diversity_weight": 0.1,
                    "efficiency_weight": 0.1,
                    "correct_score": 1.0,
                    "partial_score": 0.6,
                    "incorrect_score": 0.0,
                    "error_score": -0.1,
                    "length_penalty": 0.15,
                    "repetition_penalty": 0.25,
                    "coherence_bonus": 0.15
                }
            }
        }
    
    def get_preset_names(self) -> List[str]:
        """
        Get list of available preset names.
        
        Returns:
            List of preset names
        """
        return list(self._get_presets().keys())
    
    def validate_config(self, reward_type: str, config: Dict[str, Any]) -> bool:
        """
        Validate a configuration for a reward function type.
        
        Args:
            reward_type: Type of reward function
            config: Configuration to validate
            
        Returns:
            True if configuration is valid, False otherwise
        """
        try:
            if reward_type not in self._reward_registry:
                return False
            
            reward_class = self._reward_registry[reward_type]
            # Create a temporary instance to validate config
            temp_reward = reward_class(name="temp", config=config)
            return True
        except Exception as e:
            self.logger.warning(f"Configuration validation failed: {e}")
            return False
    
    def clear_cache(self) -> None:
        """Clear the configuration cache."""
        self._config_cache.clear()
        self.logger.info("Configuration cache cleared") 