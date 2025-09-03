"""
Data preprocessing for RLVR.

This module provides utilities for preprocessing and cleaning training data
for RLVR training.
"""

import re
from typing import List, Dict, Any, Optional
import logging


class DataPreprocessor:
    """
    Data preprocessor for RLVR training data.
    
    This class provides methods for cleaning and preprocessing training data
    to ensure quality and consistency.
    """
    
    def __init__(self, logger: Optional[logging.Logger] = None):
        """
        Initialize data preprocessor.
        
        Args:
            logger: Logger instance
        """
        self.logger = logger or logging.getLogger(__name__)
    
    def clean_text(self, text: str) -> str:
        """
        Clean and normalize text.
        
        Args:
            text: Input text to clean
            
        Returns:
            Cleaned text
        """
        if not text:
            return ""
        
        # Remove extra whitespace
        text = re.sub(r'\\s+', ' ', text.strip())
        
        # Normalize quotes
        text = text.replace('"', '"').replace('"', '"')
        text = text.replace(''', "'").replace(''', "'")
        
        # Remove control characters
        text = re.sub(r'[\\x00-\\x1f\\x7f-\\x9f]', '', text)
        
        return text
    
    def validate_sample(self, sample: Dict[str, Any]) -> Dict[str, Any]:
        """
        Validate and clean a single data sample.
        
        Args:
            sample: Data sample to validate
            
        Returns:
            Validated and cleaned sample
        """
        cleaned_sample = sample.copy()
        
        # Clean instruction
        if "instruction" in cleaned_sample:
            cleaned_sample["instruction"] = self.clean_text(cleaned_sample["instruction"])
        
        # Clean expected output
        if "expected_output" in cleaned_sample:
            cleaned_sample["expected_output"] = self.clean_text(cleaned_sample["expected_output"])
        
        # Validate context
        if "context" not in cleaned_sample:
            cleaned_sample["context"] = {}
        
        # Ensure verification type is present
        if "verification_type" not in cleaned_sample["context"]:
            cleaned_sample["context"]["verification_type"] = "unknown"
        
        return cleaned_sample
    
    def preprocess_dataset(self, data: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        Preprocess entire dataset.
        
        Args:
            data: List of data samples
            
        Returns:
            Preprocessed data
        """
        preprocessed_data = []
        
        for i, sample in enumerate(data):
            try:
                cleaned_sample = self.validate_sample(sample)
                preprocessed_data.append(cleaned_sample)
            except Exception as e:
                self.logger.warning(f"Failed to preprocess sample {i}: {e}")
                continue
        
        self.logger.info(f"Preprocessed {len(preprocessed_data)} samples from {len(data)} total")
        return preprocessed_data
    
    def filter_by_length(self, data: List[Dict[str, Any]], min_length: int = 10, max_length: int = 1000) -> List[Dict[str, Any]]:
        """
        Filter samples by instruction length.
        
        Args:
            data: List of data samples
            min_length: Minimum instruction length
            max_length: Maximum instruction length
            
        Returns:
            Filtered data
        """
        filtered_data = []
        
        for sample in data:
            instruction = sample.get("instruction", "")
            if min_length <= len(instruction) <= max_length:
                filtered_data.append(sample)
        
        self.logger.info(f"Filtered to {len(filtered_data)} samples by length")
        return filtered_data
    
    def filter_by_verification_type(self, data: List[Dict[str, Any]], allowed_types: List[str]) -> List[Dict[str, Any]]:
        """
        Filter samples by verification type.
        
        Args:
            data: List of data samples
            allowed_types: List of allowed verification types
            
        Returns:
            Filtered data
        """
        filtered_data = []
        
        for sample in data:
            context = sample.get("context", {})
            verification_type = context.get("verification_type", "unknown")
            if verification_type in allowed_types:
                filtered_data.append(sample)
        
        self.logger.info(f"Filtered to {len(filtered_data)} samples by verification type")
        return filtered_data
    
    def balance_dataset(self, data: List[Dict[str, Any]], target_samples_per_type: Optional[int] = None) -> List[Dict[str, Any]]:
        """
        Balance dataset by verification type.
        
        Args:
            data: List of data samples
            target_samples_per_type: Target number of samples per verification type
            
        Returns:
            Balanced dataset
        """
        # Group by verification type
        type_groups = {}
        for sample in data:
            context = sample.get("context", {})
            verification_type = context.get("verification_type", "unknown")
            if verification_type not in type_groups:
                type_groups[verification_type] = []
            type_groups[verification_type].append(sample)
        
        # Determine target samples per type
        if target_samples_per_type is None:
            target_samples_per_type = min(len(group) for group in type_groups.values())
        
        # Sample from each group
        balanced_data = []
        for verification_type, group in type_groups.items():
            if len(group) > target_samples_per_type:
                import random
                sampled_group = random.sample(group, target_samples_per_type)
            else:
                sampled_group = group
            balanced_data.extend(sampled_group)
        
        self.logger.info(f"Balanced dataset to {len(balanced_data)} samples")
        return balanced_data 