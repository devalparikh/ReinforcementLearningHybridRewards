"""
Language model wrapper for RLVR.

This module provides a unified interface for different language models,
abstracting away the differences between various model implementations
and providing consistent methods for generation and evaluation.
"""

import time
import traceback
from typing import Dict, Any, List, Optional, Union, Tuple
import logging
import torch
from transformers import (
    AutoTokenizer, AutoModelForCausalLM, AutoModelForSeq2SeqLM,
    GenerationConfig, PreTrainedTokenizer, PreTrainedModel
)
from dataclasses import dataclass
import numpy as np

from ..config.training_config import ModelConfig


@dataclass
class GenerationOutput:
    """Structured output from language model generation."""
    
    text: str  # Generated text
    logprobs: Optional[List[float]] = None  # Token log probabilities
    tokens: Optional[List[str]] = None  # Generated tokens
    scores: Optional[List[float]] = None  # Token scores
    generation_time: Optional[float] = None  # Generation time in seconds
    metadata: Optional[Dict[str, Any]] = None  # Additional metadata


class LanguageModel:
    """
    Unified wrapper for language models.
    
    This class provides a consistent interface for different language models,
    handling tokenization, generation, and evaluation in a standardized way.
    """
    
    def __init__(
        self,
        model_config: ModelConfig,
        logger: Optional[logging.Logger] = None
    ):
        """
        Initialize the language model wrapper.
        
        Args:
            model_config: Model configuration
            logger: Logger instance
        """
        self.config = model_config
        self.logger = logger or logging.getLogger(__name__)
        
        # Initialize model and tokenizer
        self.model = None
        self.tokenizer = None
        self.device = None
        self.dtype = None
        
        self._load_model()
    
    def _load_model(self) -> None:
        """Load the model and tokenizer."""
        try:
            # Determine device
            if self.config.device == "auto":
                if torch.cuda.is_available():
                    self.device = "cuda"
                elif hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
                    self.device = "mps"
                else:
                    self.device = "cpu"
            else:
                self.device = self.config.device
            
            # Determine dtype
            if self.config.dtype == "auto":
                if self.device == "cuda":
                    self.dtype = torch.float16
                else:
                    self.dtype = torch.float32
            else:
                dtype_map = {
                    "float32": torch.float32,
                    "float16": torch.float16,
                    "bfloat16": torch.bfloat16
                }
                self.dtype = dtype_map.get(self.config.dtype, torch.float32)
            
            self.logger.info(f"Loading model: {self.config.model_name}")
            self.logger.info(f"Device: {self.device}, Dtype: {self.dtype}")
            
            # Load tokenizer
            tokenizer_name = self.config.tokenizer_name or self.config.model_name
            self.tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)
            
            # Add padding token if not present
            if self.tokenizer.pad_token is None:
                self.tokenizer.pad_token = self.tokenizer.eos_token
            
            # Load model
            model_kwargs = {
                "torch_dtype": self.dtype,
                "device_map": "auto" if self.device == "cuda" else None
            }
            
            # Add quantization if specified
            if self.config.load_in_8bit:
                model_kwargs["load_in_8bit"] = True
            elif self.config.load_in_4bit:
                model_kwargs["load_in_4bit"] = True
            
            # Try to load as causal LM first, then as seq2seq
            try:
                self.model = AutoModelForCausalLM.from_pretrained(
                    self.config.model_name, **model_kwargs
                )
                self.model_type = "causal"
            except Exception:
                self.logger.info("Failed to load as causal LM, trying seq2seq...")
                self.model = AutoModelForSeq2SeqLM.from_pretrained(
                    self.config.model_name, **model_kwargs
                )
                self.model_type = "seq2seq"
            
            # Move model to device if not using device_map
            if self.device != "cuda" or not model_kwargs.get("device_map"):
                self.model = self.model.to(self.device)
            
            self.model.eval()
            
            self.logger.info(f"Model loaded successfully. Type: {self.model_type}")
            
        except Exception as e:
            self.logger.error(f"Failed to load model: {e}")
            raise
    
    def generate(
        self,
        prompt: str,
        max_length: Optional[int] = None,
        temperature: float = 1.0,
        top_p: float = 1.0,
        top_k: int = 50,
        do_sample: bool = True,
        num_return_sequences: int = 1,
        return_logprobs: bool = False,
        **kwargs
    ) -> Union[GenerationOutput, List[GenerationOutput]]:
        """
        Generate text from a prompt.
        
        Args:
            prompt: Input prompt
            max_length: Maximum generation length
            temperature: Sampling temperature
            top_p: Top-p sampling parameter
            top_k: Top-k sampling parameter
            do_sample: Whether to use sampling
            num_return_sequences: Number of sequences to return
            return_logprobs: Whether to return log probabilities
            **kwargs: Additional generation parameters
            
        Returns:
            Generated output(s)
        """
        start_time = time.time()
        
        try:
            # Set default max_length
            if max_length is None:
                max_length = self.config.max_length
            
            # Tokenize input
            inputs = self.tokenizer(
                prompt,
                return_tensors="pt",
                padding=True,
                truncation=True,
                max_length=max_length
            ).to(self.device)
            
            # Prepare generation config
            generation_config = GenerationConfig(
                max_length=max_length,
                temperature=temperature,
                top_p=top_p,
                top_k=top_k,
                do_sample=do_sample,
                num_return_sequences=num_return_sequences,
                pad_token_id=self.tokenizer.pad_token_id,
                eos_token_id=self.tokenizer.eos_token_id,
                **kwargs
            )
            
            # Generate
            with torch.no_grad():
                outputs = self.model.generate(
                    **inputs,
                    generation_config=generation_config,
                    return_dict_in_generate=True,
                    output_scores=return_logprobs
                )
            
            generation_time = time.time() - start_time
            
            # Process outputs
            if num_return_sequences == 1:
                return self._process_single_output(
                    outputs, inputs, return_logprobs, generation_time
                )
            else:
                return self._process_multiple_outputs(
                    outputs, inputs, return_logprobs, generation_time
                )
            
        except Exception as e:
            generation_time = time.time() - start_time
            self.logger.error(f"Generation failed: {e}")
            raise
    
    def _process_single_output(
        self,
        outputs,
        inputs,
        return_logprobs: bool,
        generation_time: float
    ) -> GenerationOutput:
        """Process a single generation output."""
        # Extract generated tokens
        generated_tokens = outputs.sequences[0][inputs.input_ids.shape[1]:]
        generated_text = self.tokenizer.decode(generated_tokens, skip_special_tokens=True)
        
        # Extract log probabilities if requested
        logprobs = None
        scores = None
        if return_logprobs and hasattr(outputs, 'scores'):
            logprobs = []
            scores = []
            for score in outputs.scores:
                probs = torch.softmax(score[0], dim=-1)
                logprobs.append(torch.log(probs).max().item())
                scores.append(score[0].max().item())
        
        # Extract individual tokens
        tokens = [self.tokenizer.decode([token], skip_special_tokens=True) 
                 for token in generated_tokens]
        
        return GenerationOutput(
            text=generated_text,
            logprobs=logprobs,
            tokens=tokens,
            scores=scores,
            generation_time=generation_time,
            metadata={
                "model_type": self.model_type,
                "input_length": inputs.input_ids.shape[1],
                "output_length": len(generated_tokens)
            }
        )
    
    def _process_multiple_outputs(
        self,
        outputs,
        inputs,
        return_logprobs: bool,
        generation_time: float
    ) -> List[GenerationOutput]:
        """Process multiple generation outputs."""
        results = []
        
        for i in range(len(outputs.sequences)):
            # Extract generated tokens for this sequence
            generated_tokens = outputs.sequences[i][inputs.input_ids.shape[1]:]
            generated_text = self.tokenizer.decode(generated_tokens, skip_special_tokens=True)
            
            # Extract log probabilities if requested
            logprobs = None
            scores = None
            if return_logprobs and hasattr(outputs, 'scores'):
                logprobs = []
                scores = []
                for score in outputs.scores:
                    probs = torch.softmax(score[i], dim=-1)
                    logprobs.append(torch.log(probs).max().item())
                    scores.append(score[i].max().item())
            
            # Extract individual tokens
            tokens = [self.tokenizer.decode([token], skip_special_tokens=True) 
                     for token in generated_tokens]
            
            results.append(GenerationOutput(
                text=generated_text,
                logprobs=logprobs,
                tokens=tokens,
                scores=scores,
                generation_time=generation_time / len(outputs.sequences),
                metadata={
                    "model_type": self.model_type,
                    "input_length": inputs.input_ids.shape[1],
                    "output_length": len(generated_tokens),
                    "sequence_index": i
                }
            ))
        
        return results
    
    def get_logprobs(
        self,
        text: str,
        return_tokens: bool = False
    ) -> Union[List[float], Tuple[List[float], List[str]]]:
        """
        Get log probabilities for a given text.
        
        Args:
            text: Input text
            return_tokens: Whether to return tokens as well
            
        Returns:
            Log probabilities (and tokens if requested)
        """
        try:
            # Tokenize
            inputs = self.tokenizer(
                text,
                return_tensors="pt",
                padding=True,
                truncation=True
            ).to(self.device)
            
            # Get logits
            with torch.no_grad():
                outputs = self.model(**inputs)
                logits = outputs.logits
            
            # Compute log probabilities
            logprobs = torch.log_softmax(logits, dim=-1)
            
            # Extract log probabilities for the actual tokens
            token_logprobs = []
            for i in range(logits.shape[1] - 1):  # -1 because we predict next token
                token_id = inputs.input_ids[0, i + 1]
                token_logprob = logprobs[0, i, token_id].item()
                token_logprobs.append(token_logprob)
            
            if return_tokens:
                tokens = self.tokenizer.convert_ids_to_tokens(
                    inputs.input_ids[0, 1:], skip_special_tokens=True
                )
                return token_logprobs, tokens
            else:
                return token_logprobs
                
        except Exception as e:
            self.logger.error(f"Failed to get log probabilities: {e}")
            raise
    
    def compute_perplexity(self, text: str) -> float:
        """
        Compute perplexity for a given text.
        
        Args:
            text: Input text
            
        Returns:
            Perplexity value
        """
        try:
            logprobs = self.get_logprobs(text)
            avg_logprob = np.mean(logprobs)
            perplexity = np.exp(-avg_logprob)
            return perplexity
        except Exception as e:
            self.logger.error(f"Failed to compute perplexity: {e}")
            raise
    
    def get_model_info(self) -> Dict[str, Any]:
        """
        Get information about the loaded model.
        
        Returns:
            Dictionary containing model information
        """
        return {
            "model_name": self.config.model_name,
            "model_type": self.model_type,
            "device": self.device,
            "dtype": str(self.dtype),
            "max_length": self.config.max_length,
            "vocab_size": self.tokenizer.vocab_size,
            "model_params": sum(p.numel() for p in self.model.parameters()),
            "trainable_params": sum(p.numel() for p in self.model.parameters() if p.requires_grad)
        }
    
    def save_model(self, path: str) -> None:
        """
        Save the model and tokenizer.
        
        Args:
            path: Path to save the model
        """
        try:
            self.model.save_pretrained(path)
            self.tokenizer.save_pretrained(path)
            self.logger.info(f"Model saved to: {path}")
        except Exception as e:
            self.logger.error(f"Failed to save model: {e}")
            raise
    
    def load_model(self, path: str) -> None:
        """
        Load a model and tokenizer from path.
        
        Args:
            path: Path to load the model from
        """
        try:
            self.model = AutoModelForCausalLM.from_pretrained(path)
            self.tokenizer = AutoTokenizer.from_pretrained(path)
            self.model = self.model.to(self.device)
            self.model.eval()
            self.logger.info(f"Model loaded from: {path}")
        except Exception as e:
            self.logger.error(f"Failed to load model: {e}")
            raise 