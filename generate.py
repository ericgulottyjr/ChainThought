#!/usr/bin/env python
# -*- coding: utf-8 -*-

import os
import argparse
import torch
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
)
from peft import PeftModel
import platform

def parse_args():
    parser = argparse.ArgumentParser(description="Generate responses from the fine-tuned Deepseek-7B model")
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
        "--question",
        type=str,
        default=None,
        help="The math question to answer",
    )
    parser.add_argument(
        "--max_new_tokens",
        type=int,
        default=512,
        help="Maximum number of new tokens to generate",
    )
    parser.add_argument(
        "--temperature",
        type=float,
        default=0.7,
        help="Sampling temperature",
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

def format_prompt(question):
    """Format the GSM8K problem into a prompt for the model."""
    return f"""Below is a grade school math problem. Solve it step-by-step.

Problem: {question}

Solution:"""

def main():
    args = parse_args()
    
    # Set random seed
    torch.manual_seed(args.seed)
    
    # Create offload folder if it doesn't exist
    os.makedirs(args.offload_folder, exist_ok=True)
    
    # Check if we're on Apple Silicon
    is_mac_arm = platform.system() == "Darwin" and platform.machine() == "arm64"
    
    if is_mac_arm:
        print("Detected Apple Silicon Mac. Using float16 precision instead of 4-bit quantization.")
        # Load model with float16 on Apple Silicon
        print(f"Loading the base model: {args.model_name_or_path}")
        model = AutoModelForCausalLM.from_pretrained(
            args.model_name_or_path,
            torch_dtype=torch.float16,
            device_map="auto",
            trust_remote_code=True,
            offload_folder=args.offload_folder,
            offload_state_dict=True,
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
    
    # Load PEFT model if it exists
    if os.path.exists(args.peft_model_path):
        print(f"Loading the fine-tuned PEFT model from {args.peft_model_path}")
        model = PeftModel.from_pretrained(model, args.peft_model_path)
    else:
        print(f"PEFT model not found at {args.peft_model_path}. Using the base model.")
    
    # Set evaluation mode
    model.eval()
    
    # Set device
    device = "cuda" if torch.cuda.is_available() else "cpu"
    
    # Get the question from user if not provided
    question = args.question
    if question is None:
        question = input("Enter a math problem: ")
    
    # Format the prompt
    prompt = format_prompt(question)
    
    # Tokenize the prompt
    inputs = tokenizer(prompt, return_tensors="pt").to(device)
    
    # Generate output
    with torch.no_grad():
        outputs = model.generate(
            input_ids=inputs["input_ids"],
            attention_mask=inputs["attention_mask"],
            max_new_tokens=args.max_new_tokens,
            do_sample=True,
            temperature=args.temperature,
            pad_token_id=tokenizer.eos_token_id,
        )
    
    # Extract output text
    output_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
    
    # Print the complete response
    print("\nModel Response:")
    print(output_text)

if __name__ == "__main__":
    main() 