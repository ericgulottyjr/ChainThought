#!/usr/bin/env python
# -*- coding: utf-8 -*-

import argparse
import torch
import numpy as np
import json
import os
import re
from datasets import load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer, GenerationConfig
from tqdm import tqdm

def parse_args():
    parser = argparse.ArgumentParser(description="Evaluate LLM on GSM8K with simple settings")
    parser.add_argument(
        "--model_name_or_path",
        type=str,
        default="deepseek-ai/deepseek-llm-7b-base",
        help="Path to the base model"
    )
    parser.add_argument(
        "--num_samples",
        type=int,
        default=10,
        help="Number of samples to evaluate"
    )
    parser.add_argument(
        "--max_new_tokens",
        type=int,
        default=100,
        help="Maximum number of new tokens to generate"
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="./baseline_results",
        help="Directory to save results"
    )
    parser.add_argument(
        "--seed", 
        type=int, 
        default=42, 
        help="Random seed"
    )
    return parser.parse_args()

def format_prompt(example):
    """Format the GSM8K problem into a prompt for the model."""
    return f"""Below is a grade school math problem. Solve it step-by-step.

Problem: {example["question"]}

Solution:"""

def extract_answer(text):
    """Extract the numerical answer from the model's output."""
    # Remove commas from numbers
    text = text.replace(",", "")
    
    # First try to find answers formatted with GSM8K-style double brackets: <<2+2=4>>
    gsm8k_pattern = r'<<.*?=\s*(\d+(?:\.\d+)?)\s*>>'
    gsm8k_matches = re.findall(gsm8k_pattern, text)
    if gsm8k_matches:
        return float(gsm8k_matches[-1])
    
    # Look for explicit "The answer is X" pattern
    answer_pattern = r"(?:the\s+answer\s+is\s+|final\s+answer\s+is\s+|therefore\s+|thus\s+)(?:\$?|\\\$)?\s*(\d+(?:\.\d+)?)"
    answer_matches = re.findall(answer_pattern, text.lower())
    if answer_matches:
        return float(answer_matches[-1])
    
    # Look for answers after a hash/number sign
    hash_pattern = r"#+\s*(\d+(?:\.\d+)?)"
    hash_matches = re.findall(hash_pattern, text)
    if hash_matches:
        return float(hash_matches[-1])
        
    # Try to find the last number in the text
    number_pattern = r"(\d+(?:\.\d+)?)"
    number_matches = re.findall(number_pattern, text)
    if number_matches:
        return float(number_matches[-1])
    
    return None

def evaluate_gsm8k(model, tokenizer, dataset, num_samples, max_new_tokens, output_dir):
    """Evaluate the model on GSM8K dataset with simple initialization."""
    correct = 0
    total = min(num_samples, len(dataset))
    results = []
    
    # Get model device
    device = next(model.parameters()).device
    print(f"Using device: {device}")
    
    for i in tqdm(range(total), desc="Evaluating"):
        example = dataset[i]
        prompt = format_prompt(example)
        
        # Tokenize input
        inputs = tokenizer(prompt, return_tensors="pt").to(device)
        
        # Generate output
        with torch.no_grad():
            outputs = model.generate(
                **inputs, 
                max_new_tokens=max_new_tokens
            )
        
        # Extract output text
        output_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
        generated_text = output_text[len(prompt):]
        
        # Extract the predicted answer
        predicted_answer = extract_answer(generated_text)
        
        # Extract the ground truth answer
        answer_text = example["answer"]
        ground_truth = extract_answer(answer_text)
        
        # Check if the answer is correct
        is_correct = False
        if predicted_answer is not None and ground_truth is not None:
            if abs(predicted_answer - ground_truth) < 1e-6:
                correct += 1
                is_correct = True
        
        # Save results
        result = {
            "id": i,
            "question": example["question"],
            "ground_truth_answer": answer_text,
            "ground_truth_value": float(ground_truth) if ground_truth is not None else None,
            "model_response": generated_text,
            "predicted_value": float(predicted_answer) if predicted_answer is not None else None,
            "is_correct": is_correct
        }
        results.append(result)
        
        # Print example
        print(f"\nExample {i + 1}:")
        print(f"Question: {example['question']}")
        print(f"Model answer: {predicted_answer}")
        print(f"Ground truth: {ground_truth}")
        print(f"Correct: {is_correct}")
        print("-" * 50)
    
    accuracy = correct / total * 100
    print(f"\nBaseline Accuracy: {accuracy:.2f}% ({correct}/{total})")
    
    # Save detailed results to file
    os.makedirs(output_dir, exist_ok=True)
    with open(os.path.join(output_dir, "baseline_results.json"), "w") as f:
        json.dump(results, f, indent=2)
    
    # Save summary to file
    summary = {
        "model": "deepseek-7b-base",
        "dataset": "gsm8k",
        "samples_evaluated": total,
        "correct": correct,
        "accuracy": accuracy,
    }
    with open(os.path.join(output_dir, "baseline_summary.json"), "w") as f:
        json.dump(summary, f, indent=2)
    
    return accuracy

def main():
    args = parse_args()
    
    # Set random seed
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    
    # Load tokenizer and model exactly as in testing.py
    print(f"Loading tokenizer: {args.model_name_or_path}")
    tokenizer = AutoTokenizer.from_pretrained(args.model_name_or_path)
    
    print(f"Loading model: {args.model_name_or_path}")
    model = AutoModelForCausalLM.from_pretrained(
        args.model_name_or_path, 
        torch_dtype=torch.bfloat16, 
        device_map="auto"
    )
    
    # Set generation config
    model.generation_config = GenerationConfig.from_pretrained(args.model_name_or_path)
    model.generation_config.pad_token_id = model.generation_config.eos_token_id
    
    # Load GSM8K test dataset
    print("Loading the GSM8K test dataset")
    dataset = load_dataset("gsm8k", "main", split="test")
    
    # Evaluate the model
    accuracy = evaluate_gsm8k(model, tokenizer, dataset, args.num_samples, args.max_new_tokens, args.output_dir)
    
    return accuracy

if __name__ == "__main__":
    main() 