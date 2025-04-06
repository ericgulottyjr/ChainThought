#!/usr/bin/env python
# -*- coding: utf-8 -*-

import argparse
import torch
import numpy as np
import json
import os
import re
from datasets import load_dataset
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    StoppingCriteria,
    StoppingCriteriaList,
    GenerationConfig
)
from tqdm import tqdm

# Custom stopping criteria for math problems - adapted to detect EOT or final answer
class MathStoppingCriteria(StoppingCriteria):
    def __init__(self, tokenizer, stop_tokens=None):
        self.tokenizer = tokenizer
        # Default stop tokens include "therefore", "thus", "the answer is", "boxed"
        self.stop_tokens = stop_tokens or ["therefore", "thus", "the answer is", "boxed", "####"]
        
    def __call__(self, input_ids, scores, **kwargs):
        # Get the generated text so far
        generated_text = self.tokenizer.decode(input_ids[0])
        
        # Check if any stop token is in the last portion of text
        last_chunk = generated_text[-100:].lower()
        for stop_token in self.stop_tokens:
            if stop_token in last_chunk:
                # Allow a few more tokens to complete the answer
                return False
                
        # Also check for repetition patterns
        if len(input_ids[0]) > 50:
            last_tokens = input_ids[0, -20:].tolist()
            prev_tokens = input_ids[0, -40:-20].tolist()
            similarity = sum(1 for a, b in zip(last_tokens, prev_tokens) if a == b) / 20
            if similarity > 0.8:
                return True
                
        return False
        
    def __repr__(self):
        return f"MathStoppingCriteria(stop_tokens={self.stop_tokens})"

def parse_args():
    parser = argparse.ArgumentParser(description="Evaluate LLM on GSM8K with improved settings")
    parser.add_argument(
        "--model_name_or_path",
        type=str,
        default="deepseek-ai/deepseek-llm-7b-base",
        help="Path to the base model"
    )
    parser.add_argument(
        "--num_samples",
        type=int,
        default=10,  # Reduced default for testing
        help="Number of samples to evaluate"
    )
    parser.add_argument(
        "--max_new_tokens",
        type=int,
        default=1024,  # Increased for more complete reasoning
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
    parser.add_argument(
        "--temperature",
        type=float,
        default=0.1,  # Lower temperature for more focused generation
        help="Temperature for generation"
    )
    parser.add_argument(
        "--precision",
        type=str,
        choices=["bf16", "fp16", "fp32", "int8", "int4"],
        default="bf16",
        help="Precision for model loading"
    )
    parser.add_argument(
        "--prompt_template",
        type=str,
        choices=["basic", "detailed", "cot", "gsm8k_format"],
        default="gsm8k_format",
        help="Prompt template to use"
    )
    return parser.parse_args()

def format_prompt(example, prompt_template="gsm8k_format"):
    """Format the GSM8K problem using different prompting strategies."""
    question = example["question"].strip()
    
    if prompt_template == "basic":
        return f"""Below is a grade school math problem.

Problem: {question}

Solution:"""
    
    elif prompt_template == "detailed":
        return f"""Below is a grade school math problem. Solve it step-by-step, explaining each calculation. After solving, clearly indicate your final answer.

Problem: {question}

Solution:"""
    
    elif prompt_template == "cot":
        return f"""Below is a grade school math problem. Let's think through it step-by-step.

Problem: {question}

Solution: Let's solve this step-by-step.
"""
    
    elif prompt_template == "gsm8k_format":
        # This format more closely matches GSM8K dataset format and includes double-checking
        return f"""Below is a grade school math problem. 

Problem: {question}

Solve the problem carefully. Show all your work step-by-step. Put your final answer on the last line with a double angle bracket notation.

Solution:"""
    
    else:
        # Default fallback
        return f"""Below is a grade school math problem.

Problem: {question}

Solution:"""

def extract_answer(text):
    """Extract the numerical answer from the model's output with improved pattern matching."""
    # Remove commas from numbers
    text = text.replace(",", "")
    
    # First try to find answers formatted with GSM8K-style double brackets: <<2+2=4>> or boxed notation
    gsm8k_pattern = r'<<.*?=\s*(\d+(?:\.\d+)?)\s*>>|\\\boxed{(\d+(?:\.\d+)?)}'
    gsm8k_matches = re.findall(gsm8k_pattern, text)
    if gsm8k_matches:
        for match in gsm8k_matches:
            # Take the first non-empty group
            for group in match:
                if group:
                    return float(group)
    
    # Look for explicit "The answer is X" pattern (strengthened)
    answer_pattern = r"(?:the\s+answer\s+is\s+|final\s+answer\s+is\s+|therefore\s+|thus\s+|hence\s+)(?:\$?|\\\$)?\s*(\d+(?:\.\d+)?)"
    answer_matches = re.findall(answer_pattern, text.lower())
    if answer_matches:
        return float(answer_matches[-1])
    
    # Look for answers after a hash/number sign
    hash_pattern = r"#+\s*(\d+(?:\.\d+)?)"
    hash_matches = re.findall(hash_pattern, text)
    if hash_matches:
        return float(hash_matches[-1])
        
    # Try to find the last number in the text that appears after "=" or "is"
    eq_pattern = r"=\s*(\d+(?:\.\d+)?)\s*$|is\s+(\d+(?:\.\d+)?)\s*$"
    eq_matches = re.findall(eq_pattern, text.lower())
    if eq_matches:
        for match in eq_matches:
            for group in match:
                if group:
                    return float(group)
    
    # As a last resort, try to find the last number in the text
    number_pattern = r"(\d+(?:\.\d+)?)"
    number_matches = re.findall(number_pattern, text)
    if number_matches:
        return float(number_matches[-1])
    
    return None

def evaluate_gsm8k(model, tokenizer, dataset, args):
    """Evaluate the model on GSM8K dataset with improved generation settings."""
    correct = 0
    total = min(args.num_samples, len(dataset))
    results = []
    
    # Set device
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")
    
    # Initialize stopping criteria
    stopping_criteria = MathStoppingCriteria(tokenizer)
    
    # Prepare generation config
    gen_config = GenerationConfig(
        max_new_tokens=args.max_new_tokens,
        temperature=args.temperature,
        top_p=0.95,
        top_k=50,
        repetition_penalty=1.2,
        do_sample=args.temperature > 0.05,  # Turn off sampling if temperature is very low
        pad_token_id=tokenizer.eos_token_id,
    )
    
    for i in tqdm(range(total), desc="Evaluating"):
        example = dataset[i]
        prompt = format_prompt(example, args.prompt_template)
        
        inputs = tokenizer(prompt, return_tensors="pt").to(device)
        
        # Generate output with better parameters
        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                generation_config=gen_config,
                stopping_criteria=StoppingCriteriaList([stopping_criteria]),
                # Allow longer sequences and avoid cutting off mid-reasoning
                min_new_tokens=100,
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
    os.makedirs(args.output_dir, exist_ok=True)
    with open(os.path.join(args.output_dir, "baseline_results.json"), "w") as f:
        json.dump(results, f, indent=2)
    
    # Save summary to file
    summary = {
        "model": args.model_name_or_path,
        "dataset": "gsm8k",
        "samples_evaluated": total,
        "correct": correct,
        "accuracy": accuracy,
        "temperature": args.temperature,
        "prompt_template": args.prompt_template,
        "precision": args.precision
    }
    with open(os.path.join(args.output_dir, "baseline_summary.json"), "w") as f:
        json.dump(summary, f, indent=2)
    
    return accuracy

def main():
    args = parse_args()
    
    # Set random seed
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    
    # Load tokenizer first
    print(f"Loading tokenizer: {args.model_name_or_path}")
    tokenizer = AutoTokenizer.from_pretrained(
        args.model_name_or_path,
        trust_remote_code=True,
        use_fast=True,
        padding_side="left"  # This helps with better batch processing
    )
    
    # Set up proper padding token
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    # Configure model loading based on precision
    print(f"Loading model with {args.precision} precision")
    model_kwargs = {
        "trust_remote_code": True,
        "device_map": "auto",
    }
    
    # Configure precision
    if args.precision == "bf16":
        model_kwargs["torch_dtype"] = torch.bfloat16
    elif args.precision == "fp16":
        model_kwargs["torch_dtype"] = torch.float16
    elif args.precision == "fp32":
        model_kwargs["torch_dtype"] = torch.float32
    elif args.precision == "int8":
        model_kwargs["load_in_8bit"] = True
    elif args.precision == "int4":
        from transformers import BitsAndBytesConfig
        model_kwargs["quantization_config"] = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_compute_dtype=torch.float16,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_use_double_quant=False  # Disabling double quantization for better quality
        )
    
    # Load model
    model = AutoModelForCausalLM.from_pretrained(
        args.model_name_or_path,
        **model_kwargs
    )
    
    # Set evaluation mode
    model.eval()
    
    # Load GSM8K test dataset
    print("Loading the GSM8K test dataset")
    dataset = load_dataset("gsm8k", "main", split="test")
    
    # Evaluate the model
    accuracy = evaluate_gsm8k(model, tokenizer, dataset, args)
    
    return accuracy

if __name__ == "__main__":
    main() 