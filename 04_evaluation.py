#!/usr/bin/env python3
"""
RLVR Evaluation Script

This script evaluates the results of RLVR training and generates comprehensive reports.
It analyzes training metrics, verification performance, and generates visualizations.

Usage:
    python 04_evaluation.py
"""

import sys
import json
import time
from typing import List, Dict, Any
import numpy as np
import pandas as pd
from pathlib import Path
import matplotlib.pyplot as plt
import seaborn as sns

# Add src to path
sys.path.append('src')

from config.training_config import get_fast_config
from utils.logging import setup_logging
from utils.metrics import TrainingMetrics, VerificationMetrics, DataQualityMetrics


def load_results():
    """Load training and verification results."""
    print("Loading results...")
    
    results = {}
    
    # Load training results
    try:
        with open('results/training_results.json', 'r') as f:
            results['training'] = json.load(f)
        print("✓ Loaded training results")
    except FileNotFoundError:
        print("⚠ No training results found")
        results['training'] = {}
    
    # Load training analysis
    try:
        with open('results/training_analysis.json', 'r') as f:
            results['analysis'] = json.load(f)
        print("✓ Loaded training analysis")
    except FileNotFoundError:
        print("⚠ No training analysis found")
        results['analysis'] = {}
    
    # Load verification results
    try:
        with open('results/verification_results.json', 'r') as f:
            results['verification'] = json.load(f)
        print("✓ Loaded verification results")
    except FileNotFoundError:
        print("⚠ No verification results found")
        results['verification'] = {}
    
    return results


def analyze_training_performance(results):
    """Analyze training performance metrics."""
    print("\n=== Training Performance Analysis ===")
    
    training_data = results.get('training', {})
    analysis_data = results.get('analysis', {})
    
    if not training_data:
        print("No training data available for analysis")
        return {}
    
    # Basic metrics
    print(f"Total episodes: {analysis_data.get('total_episodes', 'N/A')}")
    print(f"Average reward: {analysis_data.get('avg_reward', 'N/A')}")
    print(f"Final reward: {analysis_data.get('final_reward', 'N/A')}")
    print(f"Reward improvement: {analysis_data.get('reward_improvement', 'N/A')}")
    
    # Episode rewards analysis
    if 'episode_rewards' in training_data:
        rewards = training_data['episode_rewards']
        print(f"\nReward Statistics:")
        print(f"  Min: {min(rewards):.3f}")
        print(f"  Max: {max(rewards):.3f}")
        print(f"  Mean: {np.mean(rewards):.3f}")
        print(f"  Std: {np.std(rewards):.3f}")
        print(f"  Median: {np.median(rewards):.3f}")
        
        # Trend analysis
        if len(rewards) > 10:
            first_quarter = np.mean(rewards[:len(rewards)//4])
            last_quarter = np.mean(rewards[-len(rewards)//4:])
            trend = "improving" if last_quarter > first_quarter else "declining"
            print(f"  Trend: {trend} (first quarter: {first_quarter:.3f}, last quarter: {last_quarter:.3f})")
    
    # Verification accuracy analysis
    if 'verification_accuracy' in training_data:
        accuracies = training_data['verification_accuracy']
        print(f"\nVerification Accuracy:")
        print(f"  Average: {np.mean(accuracies):.3f}")
        print(f"  Final: {accuracies[-1]:.3f}")
        print(f"  Improvement: {accuracies[-1] - accuracies[0]:.3f}")
    
    return analysis_data


def analyze_verification_performance(results):
    """Analyze verification performance metrics."""
    print("\n=== Verification Performance Analysis ===")
    
    verification_data = results.get('verification', {})
    
    if not verification_data:
        print("No verification data available for analysis")
        return {}
    
    # Individual test results
    individual_tests = verification_data.get('individual_tests', {})
    for verifier_type, test_results in individual_tests.items():
        if test_results:
            success_count = sum(1 for r in test_results if r.get('success', False))
            total_count = len(test_results)
            success_rate = success_count / total_count if total_count > 0 else 0
            print(f"{verifier_type.title()} Verifier: {success_rate:.2%} ({success_count}/{total_count})")
    
    # Batch test results
    batch_tests = verification_data.get('batch_tests', {})
    for verifier_type, batch_data in batch_tests.items():
        if 'results' in batch_data:
            results_list = batch_data['results']
            success_count = sum(1 for r in results_list if r.get('success', False))
            total_count = len(results_list)
            success_rate = success_count / total_count if total_count > 0 else 0
            avg_time = batch_data.get('time', 0) / total_count if total_count > 0 else 0
            print(f"{verifier_type.title()} Batch: {success_rate:.2%} success rate, {avg_time:.3f}s avg time")
    
    # Summary statistics
    summary = verification_data.get('summary', {})
    if summary:
        print(f"\nOverall Summary:")
        print(f"  Total tests: {summary.get('total_tests', 'N/A')}")
        print(f"  Code success rate: {summary.get('code_success_rate', 'N/A')}")
        print(f"  Math success rate: {summary.get('math_success_rate', 'N/A')}")
        print(f"  Logic success rate: {summary.get('logic_success_rate', 'N/A')}")
    
    return verification_data


def generate_visualizations(results):
    """Generate training and verification visualizations."""
    print("\n=== Generating Visualizations ===")
    
    # Create plots directory
    plots_dir = Path("results/plots")
    plots_dir.mkdir(exist_ok=True)
    
    training_data = results.get('training', {})
    
    # Training rewards plot
    if 'episode_rewards' in training_data:
        rewards = training_data['episode_rewards']
        
        plt.figure(figsize=(12, 6))
        plt.plot(rewards, alpha=0.7, linewidth=1)
        plt.title('Training Rewards Over Episodes')
        plt.xlabel('Episode')
        plt.ylabel('Reward')
        plt.grid(True, alpha=0.3)
        
        # Add moving average
        if len(rewards) > 10:
            window = min(50, len(rewards) // 10)
            moving_avg = pd.Series(rewards).rolling(window=window).mean()
            plt.plot(moving_avg, linewidth=2, label=f'{window}-episode moving average')
            plt.legend()
        
        plt.tight_layout()
        plt.savefig(plots_dir / 'training_rewards.png', dpi=300, bbox_inches='tight')
        plt.close()
        print("✓ Generated training rewards plot")
    
    # Verification accuracy plot
    if 'verification_accuracy' in training_data:
        accuracies = training_data['verification_accuracy']
        
        plt.figure(figsize=(12, 6))
        plt.plot(accuracies, alpha=0.7, linewidth=1)
        plt.title('Verification Accuracy Over Episodes')
        plt.xlabel('Episode')
        plt.ylabel('Accuracy')
        plt.grid(True, alpha=0.3)
        
        # Add moving average
        if len(accuracies) > 10:
            window = min(50, len(accuracies) // 10)
            moving_avg = pd.Series(accuracies).rolling(window=window).mean()
            plt.plot(moving_avg, linewidth=2, label=f'{window}-episode moving average')
            plt.legend()
        
        plt.tight_layout()
        plt.savefig(plots_dir / 'verification_accuracy.png', dpi=300, bbox_inches='tight')
        plt.close()
        print("✓ Generated verification accuracy plot")
    
    # Verification performance comparison
    verification_data = results.get('verification', {})
    batch_tests = verification_data.get('batch_tests', {})
    
    if batch_tests:
        verifier_names = []
        success_rates = []
        avg_times = []
        
        for verifier_type, batch_data in batch_tests.items():
            if 'results' in batch_data:
                results_list = batch_data['results']
                success_count = sum(1 for r in results_list if r.get('success', False))
                total_count = len(results_list)
                success_rate = success_count / total_count if total_count > 0 else 0
                avg_time = batch_data.get('time', 0) / total_count if total_count > 0 else 0
                
                verifier_names.append(verifier_type.title())
                success_rates.append(success_rate)
                avg_times.append(avg_time)
        
        if verifier_names:
            # Success rates comparison
            plt.figure(figsize=(10, 6))
            bars = plt.bar(verifier_names, success_rates, alpha=0.7)
            plt.title('Verifier Success Rates')
            plt.ylabel('Success Rate')
            plt.ylim(0, 1)
            
            # Add value labels on bars
            for bar, rate in zip(bars, success_rates):
                plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
                        f'{rate:.2%}', ha='center', va='bottom')
            
            plt.tight_layout()
            plt.savefig(plots_dir / 'verifier_success_rates.png', dpi=300, bbox_inches='tight')
            plt.close()
            print("✓ Generated verifier success rates plot")
            
            # Average time comparison
            plt.figure(figsize=(10, 6))
            bars = plt.bar(verifier_names, avg_times, alpha=0.7)
            plt.title('Verifier Average Processing Time')
            plt.ylabel('Time (seconds)')
            
            # Add value labels on bars
            for bar, time_val in zip(bars, avg_times):
                plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.001,
                        f'{time_val:.3f}s', ha='center', va='bottom')
            
            plt.tight_layout()
            plt.savefig(plots_dir / 'verifier_processing_times.png', dpi=300, bbox_inches='tight')
            plt.close()
            print("✓ Generated verifier processing times plot")
    
    print(f"✓ All visualizations saved to {plots_dir}")


def generate_evaluation_report(results):
    """Generate a comprehensive evaluation report."""
    print("\n=== Generating Evaluation Report ===")
    
    report = {
        "evaluation_timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        "summary": {},
        "training_performance": {},
        "verification_performance": {},
        "recommendations": []
    }
    
    # Training performance summary
    training_data = results.get('training', {})
    analysis_data = results.get('analysis', {})
    
    if training_data:
        report["training_performance"] = {
            "total_episodes": analysis_data.get('total_episodes', 'N/A'),
            "average_reward": analysis_data.get('avg_reward', 'N/A'),
            "final_reward": analysis_data.get('final_reward', 'N/A'),
            "reward_improvement": analysis_data.get('reward_improvement', 'N/A'),
            "verification_accuracy": np.mean(training_data.get('verification_accuracy', [0]))
        }
    
    # Verification performance summary
    verification_data = results.get('verification', {})
    if verification_data:
        summary = verification_data.get('summary', {})
        report["verification_performance"] = {
            "total_tests": summary.get('total_tests', 'N/A'),
            "code_success_rate": summary.get('code_success_rate', 'N/A'),
            "math_success_rate": summary.get('math_success_rate', 'N/A'),
            "logic_success_rate": summary.get('logic_success_rate', 'N/A')
        }
    
    # Generate recommendations
    recommendations = []
    
    if training_data and 'episode_rewards' in training_data:
        rewards = training_data['episode_rewards']
        if len(rewards) > 10:
            first_quarter = np.mean(rewards[:len(rewards)//4])
            last_quarter = np.mean(rewards[-len(rewards)//4:])
            
            if last_quarter > first_quarter * 1.1:
                recommendations.append("Training shows good improvement in rewards")
            elif last_quarter < first_quarter:
                recommendations.append("Training may need adjustment - rewards are declining")
            else:
                recommendations.append("Training shows stable performance")
    
    if verification_data:
        batch_tests = verification_data.get('batch_tests', {})
        for verifier_type, batch_data in batch_tests.items():
            if 'results' in batch_data:
                results_list = batch_data['results']
                success_rate = sum(1 for r in results_list if r.get('success', False)) / len(results_list)
                
                if success_rate < 0.5:
                    recommendations.append(f"{verifier_type.title()} verifier needs improvement (success rate: {success_rate:.2%})")
                elif success_rate > 0.8:
                    recommendations.append(f"{verifier_type.title()} verifier performing well (success rate: {success_rate:.2%})")
    
    report["recommendations"] = recommendations
    
    # Save report
    report_file = Path("results/evaluation_report.json")
    with open(report_file, 'w') as f:
        json.dump(report, f, indent=2, default=str)
    
    print(f"✓ Generated evaluation report: {report_file}")
    
    # Print summary
    print("\n=== Evaluation Summary ===")
    print(f"Training Performance:")
    for key, value in report["training_performance"].items():
        print(f"  {key}: {value}")
    
    print(f"\nVerification Performance:")
    for key, value in report["verification_performance"].items():
        print(f"  {key}: {value}")
    
    print(f"\nRecommendations:")
    for rec in recommendations:
        print(f"  • {rec}")
    
    return report


def compare_with_baselines(results):
    """Compare results with baseline metrics."""
    print("\n=== Baseline Comparison ===")
    
    # Define baseline metrics (these would typically come from literature or previous experiments)
    baselines = {
        "average_reward": 0.5,
        "verification_accuracy": 0.7,
        "code_success_rate": 0.6,
        "math_success_rate": 0.8,
        "logic_success_rate": 0.7
    }
    
    training_data = results.get('training', {})
    verification_data = results.get('verification', {})
    
    print("Comparison with Baselines:")
    
    # Training metrics comparison
    if 'episode_rewards' in training_data:
        avg_reward = np.mean(training_data['episode_rewards'])
        baseline_reward = baselines['average_reward']
        improvement = (avg_reward - baseline_reward) / baseline_reward * 100
        print(f"  Average Reward: {avg_reward:.3f} (baseline: {baseline_reward:.3f}, {improvement:+.1f}%)")
    
    if 'verification_accuracy' in training_data:
        avg_accuracy = np.mean(training_data['verification_accuracy'])
        baseline_accuracy = baselines['verification_accuracy']
        improvement = (avg_accuracy - baseline_accuracy) / baseline_accuracy * 100
        print(f"  Verification Accuracy: {avg_accuracy:.3f} (baseline: {baseline_accuracy:.3f}, {improvement:+.1f}%)")
    
    # Verification metrics comparison
    if verification_data:
        summary = verification_data.get('summary', {})
        for metric in ['code_success_rate', 'math_success_rate', 'logic_success_rate']:
            if metric in summary:
                actual_rate = summary[metric]
                baseline_rate = baselines[metric]
                improvement = (actual_rate - baseline_rate) / baseline_rate * 100
                print(f"  {metric.replace('_', ' ').title()}: {actual_rate:.3f} (baseline: {baseline_rate:.3f}, {improvement:+.1f}%)")


def main():
    """Main evaluation function."""
    print("=== RLVR Evaluation Pipeline ===\n")
    
    # Setup logging
    logger = setup_logging(log_level="INFO")
    
    # Step 1: Load results
    print("1. Loading results...")
    results = load_results()
    
    if not any(results.values()):
        print("❌ No results found. Please run training first.")
        return
    
    # Step 2: Analyze training performance
    print("\n2. Analyzing training performance...")
    training_analysis = analyze_training_performance(results)
    
    # Step 3: Analyze verification performance
    print("\n3. Analyzing verification performance...")
    verification_analysis = analyze_verification_performance(results)
    
    # Step 4: Generate visualizations
    print("\n4. Generating visualizations...")
    generate_visualizations(results)
    
    # Step 5: Generate evaluation report
    print("\n5. Generating evaluation report...")
    report = generate_evaluation_report(results)
    
    # Step 6: Compare with baselines
    print("\n6. Comparing with baselines...")
    compare_with_baselines(results)
    
    # Summary
    print("\n=== Evaluation Summary ===")
    print(f"✓ Analyzed training performance")
    print(f"✓ Analyzed verification performance")
    print(f"✓ Generated visualizations in results/plots/")
    print(f"✓ Generated evaluation report in results/evaluation_report.json")
    print(f"✓ Compared results with baselines")
    
    print("\nFiles generated:")
    print("  - results/plots/training_rewards.png")
    print("  - results/plots/verification_accuracy.png")
    print("  - results/plots/verifier_success_rates.png")
    print("  - results/plots/verifier_processing_times.png")
    print("  - results/evaluation_report.json")
    
    print("\nNext steps:")
    print("1. Review evaluation report for insights")
    print("2. Examine visualizations for trends")
    print("3. Adjust training parameters based on recommendations")
    print("4. Run additional experiments if needed")


if __name__ == "__main__":
    main() 