"""
Metrics tracking for RLVR.

This module provides utilities for tracking and analyzing training metrics
and performance indicators.
"""

import time
from typing import Dict, Any, List, Optional, Union
import numpy as np
import pandas as pd
from collections import defaultdict, deque
import json
from pathlib import Path
import matplotlib.pyplot as plt
import seaborn as sns


class MetricsTracker:
    """
    Tracks and analyzes training metrics.
    
    This class provides functionality to track various metrics during training,
    compute statistics, and generate visualizations.
    """
    
    def __init__(self, window_size: int = 100):
        """
        Initialize metrics tracker.
        
        Args:
            window_size: Size of sliding window for moving averages
        """
        self.window_size = window_size
        self.metrics_history = defaultdict(list)
        self.moving_averages = defaultdict(lambda: deque(maxlen=window_size))
        self.start_time = time.time()
    
    def update(self, metrics: Dict[str, Any]) -> None:
        """
        Update metrics with new values.
        
        Args:
            metrics: Dictionary of metric values
        """
        current_time = time.time() - self.start_time
        
        for key, value in metrics.items():
            if isinstance(value, (int, float)):
                self.metrics_history[key].append(value)
                self.moving_averages[key].append(value)
        
        # Add timestamp
        self.metrics_history["timestamp"].append(current_time)
    
    def get_latest(self, metric_name: str) -> Optional[float]:
        """
        Get the latest value for a metric.
        
        Args:
            metric_name: Name of the metric
            
        Returns:
            Latest value or None if not found
        """
        if metric_name in self.metrics_history and self.metrics_history[metric_name]:
            return self.metrics_history[metric_name][-1]
        return None
    
    def get_moving_average(self, metric_name: str) -> Optional[float]:
        """
        Get moving average for a metric.
        
        Args:
            metric_name: Name of the metric
            
        Returns:
            Moving average or None if not enough data
        """
        if metric_name in self.moving_averages and self.moving_averages[metric_name]:
            return np.mean(self.moving_averages[metric_name])
        return None
    
    def get_statistics(self, metric_name: str) -> Dict[str, float]:
        """
        Get statistics for a metric.
        
        Args:
            metric_name: Name of the metric
            
        Returns:
            Dictionary containing statistics
        """
        if metric_name not in self.metrics_history or not self.metrics_history[metric_name]:
            return {}
        
        values = self.metrics_history[metric_name]
        
        return {
            "count": len(values),
            "mean": np.mean(values),
            "std": np.std(values),
            "min": np.min(values),
            "max": np.max(values),
            "median": np.median(values),
            "latest": values[-1],
            "moving_average": self.get_moving_average(metric_name)
        }
    
    def get_summary(self) -> Dict[str, Dict[str, float]]:
        """
        Get summary statistics for all metrics.
        
        Returns:
            Dictionary of metric summaries
        """
        summary = {}
        
        for metric_name in self.metrics_history:
            if metric_name != "timestamp":
                summary[metric_name] = self.get_statistics(metric_name)
        
        return summary
    
    def plot_metric(self, metric_name: str, save_path: Optional[str] = None) -> None:
        """
        Plot a single metric over time.
        
        Args:
            metric_name: Name of the metric to plot
            save_path: Path to save the plot (optional)
        """
        if metric_name not in self.metrics_history:
            print(f"Metric '{metric_name}' not found")
            return
        
        values = self.metrics_history[metric_name]
        timestamps = self.metrics_history["timestamp"][:len(values)]
        
        plt.figure(figsize=(12, 6))
        
        # Plot raw values
        plt.subplot(1, 2, 1)
        plt.plot(timestamps, values, alpha=0.7, label="Raw Values")
        plt.xlabel("Time (seconds)")
        plt.ylabel(metric_name)
        plt.title(f"{metric_name} Over Time")
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        # Plot moving average
        plt.subplot(1, 2, 2)
        if len(values) >= self.window_size:
            moving_avg = list(self.moving_averages[metric_name])
            moving_timestamps = timestamps[-len(moving_avg):]
            plt.plot(moving_timestamps, moving_avg, color='red', label="Moving Average")
        plt.xlabel("Time (seconds)")
        plt.ylabel(metric_name)
        plt.title(f"{metric_name} Moving Average")
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        
        plt.show()
    
    def plot_metrics_comparison(self, metric_names: List[str], save_path: Optional[str] = None) -> None:
        """
        Plot multiple metrics for comparison.
        
        Args:
            metric_names: List of metric names to plot
            save_path: Path to save the plot (optional)
        """
        plt.figure(figsize=(15, 10))
        
        for i, metric_name in enumerate(metric_names):
            if metric_name not in self.metrics_history:
                continue
            
            values = self.metrics_history[metric_name]
            timestamps = self.metrics_history["timestamp"][:len(values)]
            
            plt.subplot(2, 2, i + 1)
            plt.plot(timestamps, values, alpha=0.7, label="Raw Values")
            
            # Add moving average
            if len(values) >= self.window_size:
                moving_avg = list(self.moving_averages[metric_name])
                moving_timestamps = timestamps[-len(moving_avg):]
                plt.plot(moving_timestamps, moving_avg, color='red', label="Moving Average")
            
            plt.xlabel("Time (seconds)")
            plt.ylabel(metric_name)
            plt.title(f"{metric_name}")
            plt.legend()
            plt.grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        
        plt.show()
    
    def create_dashboard(self, save_path: Optional[str] = None) -> None:
        """
        Create a comprehensive dashboard of all metrics.
        
        Args:
            save_path: Path to save the dashboard (optional)
        """
        metrics = [name for name in self.metrics_history.keys() if name != "timestamp"]
        
        if not metrics:
            print("No metrics to display")
            return
        
        n_metrics = len(metrics)
        n_cols = min(3, n_metrics)
        n_rows = (n_metrics + n_cols - 1) // n_cols
        
        fig, axes = plt.subplots(n_rows, n_cols, figsize=(15, 5 * n_rows))
        if n_metrics == 1:
            axes = [axes]
        elif n_rows == 1:
            axes = axes.reshape(1, -1)
        
        for i, metric_name in enumerate(metrics):
            row = i // n_cols
            col = i % n_cols
            ax = axes[row, col]
            
            values = self.metrics_history[metric_name]
            timestamps = self.metrics_history["timestamp"][:len(values)]
            
            # Plot raw values
            ax.plot(timestamps, values, alpha=0.7, label="Raw Values")
            
            # Add moving average
            if len(values) >= self.window_size:
                moving_avg = list(self.moving_averages[metric_name])
                moving_timestamps = timestamps[-len(moving_avg):]
                ax.plot(moving_timestamps, moving_avg, color='red', label="Moving Average")
            
            ax.set_xlabel("Time (seconds)")
            ax.set_ylabel(metric_name)
            ax.set_title(f"{metric_name}")
            ax.legend()
            ax.grid(True, alpha=0.3)
        
        # Hide empty subplots
        for i in range(n_metrics, n_rows * n_cols):
            row = i // n_cols
            col = i % n_cols
            axes[row, col].set_visible(False)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        
        plt.show()
    
    def export_to_csv(self, filepath: str) -> None:
        """
        Export metrics to CSV file.
        
        Args:
            filepath: Path to save the CSV file
        """
        df = pd.DataFrame(self.metrics_history)
        df.to_csv(filepath, index=False)
        print(f"Metrics exported to {filepath}")
    
    def export_to_json(self, filepath: str) -> None:
        """
        Export metrics to JSON file.
        
        Args:
            filepath: Path to save the JSON file
        """
        # Convert numpy types to native Python types for JSON serialization
        json_data = {}
        for key, values in self.metrics_history.items():
            json_data[key] = [float(v) if isinstance(v, (np.integer, np.floating)) else v for v in values]
        
        with open(filepath, 'w') as f:
            json.dump(json_data, f, indent=2)
        
        print(f"Metrics exported to {filepath}")
    
    def reset(self) -> None:
        """Reset all metrics."""
        self.metrics_history.clear()
        self.moving_averages.clear()
        self.start_time = time.time()
    
    def get_training_progress(self) -> Dict[str, Any]:
        """
        Get training progress summary.
        
        Returns:
            Dictionary containing training progress information
        """
        if not self.metrics_history:
            return {}
        
        total_time = time.time() - self.start_time
        
        # Get latest values for key metrics
        key_metrics = ["reward_mean", "verification_accuracy", "confidence_mean"]
        latest_values = {}
        
        for metric in key_metrics:
            latest = self.get_latest(metric)
            if latest is not None:
                latest_values[metric] = latest
        
        # Get moving averages
        moving_avgs = {}
        for metric in key_metrics:
            moving_avg = self.get_moving_average(metric)
            if moving_avg is not None:
                moving_avgs[f"{metric}_moving_avg"] = moving_avg
        
        return {
            "total_time": total_time,
            "total_episodes": len(self.metrics_history.get("reward_mean", [])),
            "latest_values": latest_values,
            "moving_averages": moving_avgs
        }


class PerformanceMonitor:
    """
    Monitor for tracking system performance during training.
    """
    
    def __init__(self):
        """Initialize performance monitor."""
        self.metrics = MetricsTracker()
        self.start_time = time.time()
    
    def log_memory_usage(self) -> Dict[str, float]:
        """
        Log current memory usage.
        
        Returns:
            Dictionary containing memory usage information
        """
        try:
            import psutil
            process = psutil.Process()
            memory_info = process.memory_info()
            
            memory_metrics = {
                "memory_rss_mb": memory_info.rss / 1024 / 1024,
                "memory_vms_mb": memory_info.vms / 1024 / 1024,
                "memory_percent": process.memory_percent(),
                "cpu_percent": process.cpu_percent()
            }
            
            self.metrics.update(memory_metrics)
            return memory_metrics
            
        except ImportError:
            print("psutil not available for memory monitoring")
            return {}
    
    def log_gpu_usage(self) -> Dict[str, float]:
        """
        Log GPU usage if available.
        
        Returns:
            Dictionary containing GPU usage information
        """
        try:
            import torch
            if torch.cuda.is_available():
                gpu_metrics = {}
                for i in range(torch.cuda.device_count()):
                    gpu_metrics[f"gpu_{i}_memory_allocated_mb"] = torch.cuda.memory_allocated(i) / 1024 / 1024
                    gpu_metrics[f"gpu_{i}_memory_reserved_mb"] = torch.cuda.memory_reserved(i) / 1024 / 1024
                
                self.metrics.update(gpu_metrics)
                return gpu_metrics
            else:
                return {}
                
        except Exception as e:
            print(f"Error monitoring GPU usage: {e}")
            return {}
    
    def get_performance_summary(self) -> Dict[str, Any]:
        """
        Get performance summary.
        
        Returns:
            Dictionary containing performance summary
        """
        return {
            "system_metrics": self.metrics.get_summary(),
            "uptime": time.time() - self.start_time
        } 