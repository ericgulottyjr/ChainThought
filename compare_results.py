#!/usr/bin/env python
# -*- coding: utf-8 -*-

import argparse
import json
import os
import matplotlib.pyplot as plt
import numpy as np
from tabulate import tabulate

def parse_args():
    parser = argparse.ArgumentParser(description="Compare baseline and fine-tuned model results")
    parser.add_argument(
        "--baseline_dir",
        type=str,
        default="./baseline_results",
        help="Directory containing baseline results",
    )
    parser.add_argument(
        "--finetuned_dir",
        type=str,
        default="./finetuned_results",
        help="Directory containing fine-tuned model results",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="./comparison_results",
        help="Directory to save comparison results",
    )
    return parser.parse_args()

def load_results(results_dir, file_name="baseline_results.json"):
    """Load results from a JSON file."""
    results_path = os.path.join(results_dir, file_name)
    if not os.path.exists(results_path):
        print(f"Warning: Results file not found at {results_path}")
        return []
    
    with open(results_path, "r") as f:
        results = json.load(f)
    
    return results

def load_summary(results_dir, file_name="baseline_summary.json"):
    """Load summary from a JSON file."""
    summary_path = os.path.join(results_dir, file_name)
    if not os.path.exists(summary_path):
        print(f"Warning: Summary file not found at {summary_path}")
        return {}
    
    with open(summary_path, "r") as f:
        summary = json.load(f)
    
    return summary

def compare_results(baseline_results, finetuned_results):
    """Compare baseline and fine-tuned model results."""
    # Create a mapping of question ID to results
    baseline_map = {result["id"]: result for result in baseline_results}
    finetuned_map = {result["id"]: result for result in finetuned_results}
    
    # Find common question IDs
    common_ids = set(baseline_map.keys()) & set(finetuned_map.keys())
    
    comparison = []
    for qid in common_ids:
        baseline = baseline_map[qid]
        finetuned = finetuned_map[qid]
        
        comparison.append({
            "id": qid,
            "question": baseline["question"],
            "ground_truth": baseline["ground_truth_value"],
            "baseline_predicted": baseline["predicted_value"],
            "baseline_correct": baseline["is_correct"],
            "finetuned_predicted": finetuned["predicted_value"],
            "finetuned_correct": finetuned["is_correct"],
            "improvement": finetuned["is_correct"] and not baseline["is_correct"],
            "regression": baseline["is_correct"] and not finetuned["is_correct"]
        })
    
    return comparison

def generate_comparison_report(comparison, baseline_summary, finetuned_summary, output_dir):
    """Generate a report comparing baseline and fine-tuned model performance."""
    # Extract stats
    total_samples = len(comparison)
    baseline_correct = sum(1 for item in comparison if item["baseline_correct"])
    finetuned_correct = sum(1 for item in comparison if item["finetuned_correct"])
    improvements = sum(1 for item in comparison if item["improvement"])
    regressions = sum(1 for item in comparison if item["regression"])
    
    baseline_accuracy = baseline_correct / total_samples * 100
    finetuned_accuracy = finetuned_correct / total_samples * 100
    improvement_percentage = (finetuned_accuracy - baseline_accuracy)
    
    # Create summary table
    summary_table = [
        ["Model", "Accuracy", "Correct", "Total"],
        ["Baseline (Deepseek-7B)", f"{baseline_accuracy:.2f}%", baseline_correct, total_samples],
        ["Fine-tuned", f"{finetuned_accuracy:.2f}%", finetuned_correct, total_samples],
        ["Difference", f"{improvement_percentage:+.2f}%", finetuned_correct - baseline_correct, ""]
    ]
    
    # Create detailed stats table
    stats_table = [
        ["Metric", "Count", "Percentage"],
        ["Improvements (B:✗ → FT:✓)", improvements, f"{improvements/total_samples*100:.2f}%"],
        ["Regressions (B:✓ → FT:✗)", regressions, f"{regressions/total_samples*100:.2f}%"],
        ["Both correct (B:✓, FT:✓)", baseline_correct - regressions, f"{(baseline_correct - regressions)/total_samples*100:.2f}%"],
        ["Both incorrect (B:✗, FT:✗)", total_samples - baseline_correct - improvements, f"{(total_samples - baseline_correct - improvements)/total_samples*100:.2f}%"]
    ]
    
    # Create examples of improvements and regressions
    improved_examples = [item for item in comparison if item["improvement"]][:5]  # First 5 improvements
    regressed_examples = [item for item in comparison if item["regression"]][:5]  # First 5 regressions
    
    # Format the report
    report = "# Comparison Report: Baseline vs. Fine-tuned Deepseek-7B on GSM8K\n\n"
    
    report += "## Summary\n\n"
    report += tabulate(summary_table, headers="firstrow", tablefmt="pipe") + "\n\n"
    
    report += "## Detailed Statistics\n\n"
    report += tabulate(stats_table, headers="firstrow", tablefmt="pipe") + "\n\n"
    
    if improved_examples:
        report += "## Examples of Improvements\n\n"
        for i, example in enumerate(improved_examples):
            report += f"### Example {i+1}\n\n"
            report += f"**Question:** {example['question']}\n\n"
            report += f"**Ground Truth:** {example['ground_truth']}\n\n"
            report += f"**Baseline prediction:** {example['baseline_predicted']}\n\n"
            report += f"**Fine-tuned prediction:** {example['finetuned_predicted']}\n\n"
            report += "---\n\n"
    
    if regressed_examples:
        report += "## Examples of Regressions\n\n"
        for i, example in enumerate(regressed_examples):
            report += f"### Example {i+1}\n\n"
            report += f"**Question:** {example['question']}\n\n"
            report += f"**Ground Truth:** {example['ground_truth']}\n\n"
            report += f"**Baseline prediction:** {example['baseline_predicted']}\n\n"
            report += f"**Fine-tuned prediction:** {example['finetuned_predicted']}\n\n"
            report += "---\n\n"
    
    # Save the report
    os.makedirs(output_dir, exist_ok=True)
    with open(os.path.join(output_dir, "comparison_report.md"), "w") as f:
        f.write(report)
    
    # Create visualization
    create_visualization(baseline_accuracy, finetuned_accuracy, improvements, regressions, total_samples, output_dir)
    
    return report

def create_visualization(baseline_acc, finetuned_acc, improvements, regressions, total, output_dir):
    """Create visualizations comparing baseline and fine-tuned performance."""
    # Create a figure with two subplots
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
    
    # Accuracy comparison bar chart
    models = ['Baseline', 'Fine-tuned']
    accuracies = [baseline_acc, finetuned_acc]
    bar_colors = ['#1f77b4', '#ff7f0e']
    
    bars = ax1.bar(models, accuracies, color=bar_colors)
    ax1.set_title('Model Accuracy Comparison', fontsize=14)
    ax1.set_ylabel('Accuracy (%)', fontsize=12)
    ax1.set_ylim(0, 100)
    
    # Add value labels on top of the bars
    for bar in bars:
        height = bar.get_height()
        ax1.text(bar.get_x() + bar.get_width()/2., height + 1,
                f'{height:.2f}%', ha='center', va='bottom', fontsize=11)
    
    # Add a horizontal line for the baseline
    ax1.axhline(y=baseline_acc, color='#1f77b4', linestyle='--', alpha=0.5)
    
    # Pie chart for error analysis
    both_correct = baseline_acc/100 * total - regressions
    both_incorrect = total - both_correct - improvements - regressions
    
    labels = ['Both Correct', 'Improvements', 'Regressions', 'Both Incorrect']
    sizes = [both_correct, improvements, regressions, both_incorrect]
    colors = ['#2ca02c', '#ff7f0e', '#d62728', '#7f7f7f']
    explode = (0.1, 0.1, 0.1, 0)  # explode the first three slices
    
    # Only include non-zero segments in the pie chart
    non_zero_labels = [label for label, size in zip(labels, sizes) if size > 0]
    non_zero_sizes = [size for size in sizes if size > 0]
    non_zero_colors = [color for color, size in zip(colors, sizes) if size > 0]
    non_zero_explode = [explode[i] for i, size in enumerate(sizes) if size > 0]
    
    ax2.pie(non_zero_sizes, explode=non_zero_explode, labels=non_zero_labels, colors=non_zero_colors,
           autopct='%1.1f%%', shadow=True, startangle=90)
    ax2.set_title('Performance Analysis', fontsize=14)
    
    # Adjust layout and save
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'performance_comparison.png'), dpi=300, bbox_inches='tight')
    plt.close()

def save_comparison_data(comparison, output_dir):
    """Save the detailed comparison data as JSON."""
    os.makedirs(output_dir, exist_ok=True)
    with open(os.path.join(output_dir, "detailed_comparison.json"), "w") as f:
        json.dump(comparison, f, indent=2)

def main():
    args = parse_args()
    
    # Load baseline results
    baseline_results = load_results(args.baseline_dir, "baseline_results.json")
    baseline_summary = load_summary(args.baseline_dir, "baseline_summary.json")
    
    # Load fine-tuned results
    finetuned_results = load_results(args.finetuned_dir, "baseline_results.json")
    finetuned_summary = load_summary(args.finetuned_dir, "baseline_summary.json")
    
    if not baseline_results or not finetuned_results:
        print("Error: Could not load results files.")
        return
    
    # Compare results
    comparison = compare_results(baseline_results, finetuned_results)
    
    # Save comparison data
    save_comparison_data(comparison, args.output_dir)
    
    # Generate report
    report = generate_comparison_report(comparison, baseline_summary, finetuned_summary, args.output_dir)
    
    print(f"Comparison completed. Results saved to {args.output_dir}")
    print(f"Summary: Baseline accuracy: {baseline_summary.get('accuracy', 'N/A')}%, Fine-tuned accuracy: {finetuned_summary.get('accuracy', 'N/A')}%")

if __name__ == "__main__":
    main() 