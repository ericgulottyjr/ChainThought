#!/usr/bin/env python
# -*- coding: utf-8 -*-

import os
import re
import argparse
import torch
import numpy as np
import json
from datasets import load_dataset
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
)
from peft import PeftModel
from tqdm import tqdm
import platform

def parse_args():
    parser = argparse.ArgumentParser(description="Evaluate fine-tuned Deepseek-7B on GSM8K")
    parser.add_argument(
        "--model_name_or_path",
        type=str,
        default="deepseek-ai/deepseek-llm-7b-base",
        help="Path to the base model",
    )
    parser.add_argument(
        "--peft_model_path",
        type=str,
        default="./outputs",
        help="Path to the fine-tuned PEFT model",
    )
    parser.add_argument(
        "--num_samples",
        type=int,
        default=100,
        help="Number of samples to evaluate",
    )
    parser.add_argument(
        "--max_new_tokens",
        type=int,
        default=512,
        help="Maximum number of new tokens to generate",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="./finetuned_results",
        help="Directory to save results",
    )
    parser.add_argument(
        "--offload_folder", 
        type=str, 
        default="./offload_folder",
        help="Folder for offloading model weights"
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
    
    # Try to find answers with dollar signs
    dollar_pattern = r"\$?(\d+(?:\.\d+)?)"
    dollar_matches = re.findall(dollar_pattern, text)
    
    # Look for "the answer is X" pattern
    answer_pattern = r"(?:the\s+answer\s+is\s+|equals\s+|=\s*)(\d+(?:\.\d+)?)"
    answer_matches = re.findall(answer_pattern, text.lower())
    
    # Try to find the last number in the text
    last_number_pattern = r"(\d+(?:\.\d+)?)"
    number_matches = re.findall(last_number_pattern, text)
    
    # Priority: "the answer is" pattern > dollar sign > last number
    if answer_matches:
        return float(answer_matches[-1])
    elif dollar_matches:
        return float(dollar_matches[-1])
    elif number_matches:
        return float(number_matches[-1])
    
    return None

def evaluate_gsm8k(model, tokenizer, dataset, num_samples, max_new_tokens, output_dir):
    """Evaluate the model on GSM8K dataset."""
    correct = 0
    total = min(num_samples, len(dataset))
    results = []
    
    # Set device
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")
    
    for i in tqdm(range(total), desc="Evaluating"):
        example = dataset[i]
        prompt = format_prompt(example)
        
        inputs = tokenizer(prompt, return_tensors="pt").to(device)
        
        # Generate output
        with torch.no_grad():
            outputs = model.generate(
                input_ids=inputs["input_ids"],
                attention_mask=inputs["attention_mask"],
                max_new_tokens=max_new_tokens,
                do_sample=False,  # Greedy decoding
                temperature=1.0,  # Not used with greedy decoding
                pad_token_id=tokenizer.eos_token_id,
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
        print(f"Example {i + 1}:")
        print(f"Question: {example['question']}")
        print(f"Generated answer: {predicted_answer}")
        print(f"Ground truth: {ground_truth}")
        print(f"Correct: {is_correct}")
        print("-" * 50)
    
    accuracy = correct / total * 100
    print(f"Fine-tuned Model Accuracy: {accuracy:.2f}% ({correct}/{total})")
    
    # Save detailed results to file
    os.makedirs(output_dir, exist_ok=True)
    with open(os.path.join(output_dir, "baseline_results.json"), "w") as f:
        json.dump(results, f, indent=2)
    
    # Save summary to file
    model_name = "deepseek-7b-finetuned" if os.path.exists(args.peft_model_path) else "deepseek-7b-base"
    summary = {
        "model": model_name,
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
    
    # Create offload folder if it doesn't exist
    os.makedirs(args.offload_folder, exist_ok=True)
    
    # Check if we're on Apple Silicon
    is_mac_arm = platform.system() == "Darwin" and platform.machine() == "arm64"
    
    # Configure loading based on platform
    if is_mac_arm:
        print("Detected Apple Silicon Mac. Using float16 precision instead of 4-bit quantization.")
        # Load base model
        print(f"Loading the base model: {args.model_name_or_path}")
        model = AutoModelForCausalLM.from_pretrained(
            args.model_name_or_path,
            torch_dtype=torch.float16,
            device_map="auto",
            trust_remote_code=True,
            offload_folder=args.offload_folder,
            offload_state_dict=True,  # Explicitly enable offloading
        )
    else:
        # Configure quantization
        bnb_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_use_double_quant=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.bfloat16
        )
        
        # Load base model
        print(f"Loading the base model: {args.model_name_or_path}")
        model = AutoModelForCausalLM.from_pretrained(
            args.model_name_or_path,
            quantization_config=bnb_config,
            device_map="auto",
            trust_remote_code=True,
            offload_folder=args.offload_folder,
        )
    
    # Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained(
        args.model_name_or_path,
        trust_remote_code=True,
        use_fast=True
    )
    
    # Load PEFT model
    if os.path.exists(args.peft_model_path):
        print(f"Loading the fine-tuned PEFT model from {args.peft_model_path}")
        model = PeftModel.from_pretrained(model, args.peft_model_path)
    else:
        print(f"PEFT model not found at {args.peft_model_path}. Using the base model.")
    
    # Set evaluation mode
    model.eval()
    
    # Load GSM8K test dataset
    print("Loading the GSM8K test dataset")
    dataset = load_dataset("gsm8k", "main", split="test")
    
    # Evaluate the model
    accuracy = evaluate_gsm8k(model, tokenizer, dataset, args.num_samples, args.max_new_tokens, args.output_dir)
    
    return accuracy

if __name__ == "__main__":
    main() 