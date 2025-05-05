import json
import re
import argparse
import numpy as np
from collections import Counter
import os
from typing import List, Dict, Tuple, Any, Optional

def load_jsonl(file_path: str) -> List[Dict]:
    """Load data from a jsonl file."""
    data = []
    with open(file_path, 'r') as f:
        for line in f:
            data.append(json.loads(line))
    return data

def extract_answer(text: str) -> Optional[str]:
    """Extract the final answer from text using regex pattern matching."""
    # Look for patterns like "answer is 42" or "Answer: 42" or just the last number
    answer_patterns = [
        r"(?:answer|result)(?:\s+is)?(?:\s*:)?\s*(-?\d+(?:\.\d+)?)",  # "answer is 42" or "Answer: 42"
        r"(?:final answer|final result)(?:\s+is)?(?:\s*:)?\s*(-?\d+(?:\.\d+)?)",  # "final answer is 42"
        r"(?:the answer|the result)(?:\s+is)?(?:\s*:)?\s*(-?\d+(?:\.\d+)?)",  # "the answer is 42"
    ]
    
    for pattern in answer_patterns:
        matches = re.findall(pattern, text, re.IGNORECASE)
        if matches:
            return matches[-1].strip()  # Return the last match
    
    # If no structured answer found, try to find the last number in the text
    numbers = re.findall(r"(-?\d+(?:\.\d+)?)", text)
    if numbers:
        return numbers[-1].strip()
    
    return None

def tokenize_text(text: str) -> List[str]:
    """Simple whitespace-based tokenization."""
    return text.split()

def calculate_exact_match(predictions: List[str], targets: List[str]) -> float:
    """Calculate exact match accuracy."""
    correct = 0
    total = 0
    
    for pred, target in zip(predictions, targets):
        pred_answer = extract_answer(pred)
        target_answer = extract_answer(target)
        
        if pred_answer is not None and target_answer is not None:
            if pred_answer == target_answer:
                correct += 1
            total += 1
    
    if total == 0:
        return 0.0
    return correct / total

def calculate_per_token_accuracy(predictions: List[str], targets: List[str]) -> float:
    """Calculate per-token accuracy."""
    correct_tokens = 0
    total_tokens = 0
    
    for pred, target in zip(predictions, targets):
        pred_tokens = tokenize_text(pred)
        target_tokens = tokenize_text(target)
        
        # Simple token-level comparison (allows for different lengths)
        min_len = min(len(pred_tokens), len(target_tokens))
        for i in range(min_len):
            if pred_tokens[i] == target_tokens[i]:
                correct_tokens += 1
        total_tokens += len(target_tokens)
    
    if total_tokens == 0:
        return 0.0
    return correct_tokens / total_tokens

def calculate_token_level_metrics(predictions: List[str], targets: List[str]) -> Dict[str, float]:
    """Calculate token-level precision, recall, and F1 score."""
    total_precision = 0.0
    total_recall = 0.0
    total_f1 = 0.0
    valid_examples = 0
    
    for pred, target in zip(predictions, targets):
        pred_tokens = set(tokenize_text(pred))
        target_tokens = set(tokenize_text(target))
        
        if len(pred_tokens) == 0 and len(target_tokens) == 0:
            continue
        
        # Calculate intersection
        intersection = pred_tokens.intersection(target_tokens)
        
        # Calculate precision, recall, and F1
        precision = len(intersection) / len(pred_tokens) if len(pred_tokens) > 0 else 0
        recall = len(intersection) / len(target_tokens) if len(target_tokens) > 0 else 0
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
        
        total_precision += precision
        total_recall += recall
        total_f1 += f1
        valid_examples += 1
    
    if valid_examples == 0:
        return {
            "precision": 0.0,
            "recall": 0.0,
            "f1": 0.0
        }
    
    return {
        "precision": total_precision / valid_examples,
        "recall": total_recall / valid_examples,
        "f1": total_f1 / valid_examples
    }

def main():
    parser = argparse.ArgumentParser(description="Calculate evaluation metrics for LLM outputs")
    parser.add_argument("--eval_file", type=str, required=True, help="Path to the evaluation output JSONL file")
    
    args = parser.parse_args()
    
    # Load evaluation data
    print(f"Loading evaluation data from {args.eval_file}...")
    data = load_jsonl(args.eval_file)
    
    # Extract predictions and targets
    predictions = [item["prediction"] for item in data]
    targets = [item["target"] for item in data]
    
    # Calculate metrics
    print("Calculating metrics...")
    metrics = {}
    
    # 1. Exact-Match Accuracy
    metrics["exact_match_accuracy"] = calculate_exact_match(predictions, targets)
    
    # 2. Per-Token Accuracy
    metrics["per_token_accuracy"] = calculate_per_token_accuracy(predictions, targets)
    
    # 3. Token-Level Precision/Recall/F1
    token_metrics = calculate_token_level_metrics(predictions, targets)
    metrics.update(token_metrics)
    
    # Print results
    print("\n===== EVALUATION METRICS =====")
    print(f"Evaluated {len(data)} examples from {args.eval_file}")
    print("\nMetrics:")
    for metric_name, value in metrics.items():
        print(f"  {metric_name}: {value:.4f}")
    
    # Save metrics to file
    output_path = "../outputs/train2_evalmetrics.jsonl"
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, 'w') as f:
        json.dump(metrics, f, indent=2)
    print(f"\nMetrics saved to {output_path}")

if __name__ == "__main__":
    main() 