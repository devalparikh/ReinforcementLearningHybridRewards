"""
Dataset handling for RLVR.

This module provides classes for loading and managing training data
for RLVR training.
"""

import json
from typing import List, Dict, Any, Optional
from pathlib import Path
import logging


class RLVRDataset:
    """
    Dataset class for RLVR training data.
    
    This class handles loading, preprocessing, and accessing training data
    for RLVR training.
    """
    
    def __init__(self, data_path: str, logger: Optional[logging.Logger] = None):
        """
        Initialize RLVR dataset.
        
        Args:
            data_path: Path to the data file
            logger: Logger instance
        """
        self.data_path = Path(data_path)
        self.logger = logger or logging.getLogger(__name__)
        self.data = []
        
        self._load_data()
    
    def _load_data(self) -> None:
        """Load data from file."""
        if not self.data_path.exists():
            raise FileNotFoundError(f"Data file not found: {self.data_path}")
        
        if self.data_path.suffix.lower() == '.jsonl':
            self._load_jsonl()
        elif self.data_path.suffix.lower() == '.json':
            self._load_json()
        else:
            raise ValueError(f"Unsupported file format: {self.data_path.suffix}")
        
        self.logger.info(f"Loaded {len(self.data)} samples from {self.data_path}")
    
    def _load_jsonl(self) -> None:
        """Load data from JSONL file."""
        with open(self.data_path, 'r') as f:
            for line in f:
                if line.strip():
                    self.data.append(json.loads(line.strip()))
    
    def _load_json(self) -> None:
        """Load data from JSON file."""
        with open(self.data_path, 'r') as f:
            self.data = json.load(f)
    
    def __len__(self) -> int:
        """Return the number of samples in the dataset."""
        return len(self.data)
    
    def __getitem__(self, index: int) -> Dict[str, Any]:
        """Get a sample by index."""
        return self.data[index]
    
    def get_sample(self, index: int) -> Dict[str, Any]:
        """Get a sample by index with validation."""
        if index < 0 or index >= len(self.data):
            raise IndexError(f"Index {index} out of range")
        return self.data[index]
    
    def get_batch(self, indices: List[int]) -> List[Dict[str, Any]]:
        """Get a batch of samples by indices."""
        return [self.get_sample(i) for i in indices]
    
    def filter_by_verification_type(self, verification_type: str) -> List[Dict[str, Any]]:
        """Filter samples by verification type."""
        filtered = []
        for sample in self.data:
            context = sample.get("context", {})
            if context.get("verification_type") == verification_type:
                filtered.append(sample)
        return filtered
    
    def filter_by_difficulty(self, difficulty: str) -> List[Dict[str, Any]]:
        """Filter samples by difficulty level."""
        return [sample for sample in self.data if sample.get("difficulty") == difficulty]
    
    def get_statistics(self) -> Dict[str, Any]:
        """Get dataset statistics."""
        if not self.data:
            return {"total_samples": 0}
        
        verification_types = {}
        difficulties = {}
        instruction_lengths = []
        output_lengths = []
        
        for sample in self.data:
            # Verification types
            context = sample.get("context", {})
            vtype = context.get("verification_type", "unknown")
            verification_types[vtype] = verification_types.get(vtype, 0) + 1
            
            # Difficulties
            difficulty = sample.get("difficulty", "unknown")
            difficulties[difficulty] = difficulties.get(difficulty, 0) + 1
            
            # Lengths
            instruction_lengths.append(len(sample.get("instruction", "")))
            output_lengths.append(len(sample.get("expected_output", "")))
        
        return {
            "total_samples": len(self.data),
            "verification_types": verification_types,
            "difficulties": difficulties,
            "instruction_length_stats": {
                "mean": sum(instruction_lengths) / len(instruction_lengths),
                "min": min(instruction_lengths),
                "max": max(instruction_lengths)
            },
            "output_length_stats": {
                "mean": sum(output_lengths) / len(output_lengths),
                "min": min(output_lengths),
                "max": max(output_lengths)
            }
        } 