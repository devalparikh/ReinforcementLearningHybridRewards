"""
Logging utilities for RLVR.

This module provides centralized logging configuration and utilities
for the RLVR project.
"""

import logging
import sys
from pathlib import Path
from typing import Optional, Dict, Any
import json
from datetime import datetime


def setup_logging(
    log_level: str = "INFO",
    log_file: Optional[str] = None,
    log_format: str = "%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    date_format: str = "%Y-%m-%d %H:%M:%S"
) -> None:
    """
    Setup logging configuration for the RLVR project.
    
    Args:
        log_level: Logging level (DEBUG, INFO, WARNING, ERROR, CRITICAL)
        log_file: Path to log file (optional)
        log_format: Format string for log messages
        date_format: Format string for timestamps
    """
    # Create logger
    logger = logging.getLogger()
    logger.setLevel(getattr(logging, log_level.upper()))
    
    # Clear existing handlers
    logger.handlers.clear()
    
    # Create formatter
    formatter = logging.Formatter(log_format, date_format)
    
    # Console handler
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(getattr(logging, log_level.upper()))
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)
    
    # File handler (if specified)
    if log_file:
        log_path = Path(log_file)
        log_path.parent.mkdir(parents=True, exist_ok=True)
        
        file_handler = logging.FileHandler(log_file)
        file_handler.setLevel(getattr(logging, log_level.upper()))
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)
    
    # Set specific loggers to avoid duplicate messages
    logging.getLogger("transformers").setLevel(logging.WARNING)
    logging.getLogger("torch").setLevel(logging.WARNING)
    logging.getLogger("wandb").setLevel(logging.WARNING)


def get_logger(name: str) -> logging.Logger:
    """
    Get a logger instance for a specific module.
    
    Args:
        name: Logger name (usually __name__)
        
    Returns:
        Logger instance
    """
    return logging.getLogger(name)


class StructuredLogger:
    """
    Structured logger for RLVR that supports structured logging.
    
    This logger can output logs in JSON format for better parsing
    and analysis.
    """
    
    def __init__(
        self,
        name: str,
        log_file: Optional[str] = None,
        use_json: bool = False
    ):
        """
        Initialize structured logger.
        
        Args:
            name: Logger name
            log_file: Path to log file
            use_json: Whether to use JSON format
        """
        self.logger = logging.getLogger(name)
        self.use_json = use_json
        self.log_file = log_file
        
        if log_file:
            self._setup_file_handler()
    
    def _setup_file_handler(self) -> None:
        """Setup file handler for structured logging."""
        log_path = Path(self.log_file)
        log_path.parent.mkdir(parents=True, exist_ok=True)
        
        if self.use_json:
            handler = JSONFileHandler(self.log_file)
        else:
            handler = logging.FileHandler(self.log_file)
            formatter = logging.Formatter(
                "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
            )
            handler.setFormatter(formatter)
        
        self.logger.addHandler(handler)
    
    def info(self, message: str, **kwargs) -> None:
        """Log info message with additional context."""
        if self.use_json:
            self._log_json("INFO", message, **kwargs)
        else:
            if kwargs:
                message = f"{message} - {kwargs}"
            self.logger.info(message)
    
    def warning(self, message: str, **kwargs) -> None:
        """Log warning message with additional context."""
        if self.use_json:
            self._log_json("WARNING", message, **kwargs)
        else:
            if kwargs:
                message = f"{message} - {kwargs}"
            self.logger.warning(message)
    
    def error(self, message: str, **kwargs) -> None:
        """Log error message with additional context."""
        if self.use_json:
            self._log_json("ERROR", message, **kwargs)
        else:
            if kwargs:
                message = f"{message} - {kwargs}"
            self.logger.error(message)
    
    def debug(self, message: str, **kwargs) -> None:
        """Log debug message with additional context."""
        if self.use_json:
            self._log_json("DEBUG", message, **kwargs)
        else:
            if kwargs:
                message = f"{message} - {kwargs}"
            self.logger.debug(message)
    
    def _log_json(self, level: str, message: str, **kwargs) -> None:
        """Log message in JSON format."""
        log_entry = {
            "timestamp": datetime.now().isoformat(),
            "level": level,
            "message": message,
            "logger": self.logger.name,
            **kwargs
        }
        
        # Find JSON handler
        for handler in self.logger.handlers:
            if isinstance(handler, JSONFileHandler):
                handler.emit_json(log_entry)
                break


class JSONFileHandler(logging.FileHandler):
    """File handler that outputs JSON formatted logs."""
    
    def emit_json(self, log_entry: Dict[str, Any]) -> None:
        """Emit a JSON log entry."""
        try:
            json_line = json.dumps(log_entry) + "\n"
            self.stream.write(json_line)
            self.flush()
        except Exception:
            self.handleError(log_entry)


class TrainingLogger:
    """
    Specialized logger for training events.
    
    This logger provides methods for logging training-specific events
    with structured data.
    """
    
    def __init__(self, name: str = "training"):
        """
        Initialize training logger.
        
        Args:
            name: Logger name
        """
        self.logger = StructuredLogger(name)
    
    def log_episode(self, episode: int, metrics: Dict[str, Any]) -> None:
        """Log episode metrics."""
        self.logger.info(
            "Episode completed",
            episode=episode,
            **metrics
        )
    
    def log_evaluation(self, episode: int, metrics: Dict[str, Any]) -> None:
        """Log evaluation metrics."""
        self.logger.info(
            "Evaluation completed",
            episode=episode,
            **metrics
        )
    
    def log_checkpoint(self, episode: int, path: str, metrics: Dict[str, Any]) -> None:
        """Log checkpoint save."""
        self.logger.info(
            "Checkpoint saved",
            episode=episode,
            checkpoint_path=path,
            **metrics
        )
    
    def log_error(self, error: Exception, context: Optional[Dict[str, Any]] = None) -> None:
        """Log training error."""
        error_info = {
            "error_type": type(error).__name__,
            "error_message": str(error),
            "traceback": str(error.__traceback__)
        }
        
        if context:
            error_info.update(context)
        
        self.logger.error("Training error occurred", **error_info)
    
    def log_config(self, config: Dict[str, Any]) -> None:
        """Log training configuration."""
        self.logger.info("Training configuration", **config)
    
    def log_model_info(self, model_info: Dict[str, Any]) -> None:
        """Log model information."""
        self.logger.info("Model loaded", **model_info) 