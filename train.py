#!/usr/bin/env python
# -*- coding: utf-8 -*-

import os
import argparse
import torch
from datasets import load_dataset
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
    TrainingArguments,
    set_seed,
    Trainer
)
from peft import (
    LoraConfig,
    get_peft_model,
    prepare_model_for_kbit_training,
)

def parse_args():
    parser = argparse.ArgumentParser(description="Fine-tune Deepseek-7B on GSM8K")
    parser.add_argument(
        "--model_name_or_path",
        type=str,
        default="deepseek-ai/deepseek-llm-7b-base",
        help="Path to the pretrained model or model identifier from huggingface.co/models",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="./outputs",
        help="The output directory where the model predictions and checkpoints will be written",
    )
    parser.add_argument(
        "--per_device_train_batch_size",
        type=int,
        default=4,
        help="Batch size per GPU/TPU core/CPU for training",
    )
    parser.add_argument(
        "--learning_rate",
        type=float,
        default=2e-5,
        help="The initial learning rate for AdamW optimizer",
    )
    parser.add_argument(
        "--num_train_epochs",
        type=int,
        default=3,
        help="Total number of training epochs to perform",
    )
    parser.add_argument(
        "--seed", 
        type=int, 
        default=42, 
        help="A seed for reproducible training"
    )
    parser.add_argument(
        "--lora_r",
        type=int,
        default=16,
        help="Lora attention dimension",
    )
    parser.add_argument(
        "--lora_alpha",
        type=int,
        default=32,
        help="Lora alpha parameter",
    )
    parser.add_argument(
        "--lora_dropout",
        type=float,
        default=0.05,
        help="Lora dropout parameter",
    )
    parser.add_argument(
        "--gradient_accumulation_steps",
        type=int,
        default=8,
        help="Number of updates steps to accumulate before performing a backward/update pass",
    )
    parser.add_argument(
        "--max_seq_length",
        type=int,
        default=2048,
        help="The maximum total input sequence length after tokenization",
    )
    parser.add_argument(
        "--offload_folder", 
        type=str, 
        default="./offload_folder",
        help="Folder for offloading model weights"
    )
    return parser.parse_args()

def format_prompt(example):
    """Format the GSM8K problem into a prompt for the model."""
    return f"""Below is a grade school math problem. Solve it step-by-step.

Problem: {example["question"]}

Solution:"""

def process_data(example, tokenizer, max_seq_length):
    """Process and tokenize the data."""
    prompt = format_prompt(example)
    answer = example["answer"]
    
    # Combine prompt and answer for training
    full_text = f"{prompt}\n{answer}"
    
    # Tokenize
    tokenized = tokenizer(
        full_text,
        max_length=max_seq_length,
        padding="max_length",
        truncation=True,
    )
    
    # Prepare the labels
    tokenized["labels"] = tokenized["input_ids"].copy()
    
    return tokenized

def main():
    args = parse_args()
    set_seed(args.seed)
    
    # Create offload folder if it doesn't exist
    os.makedirs(args.offload_folder, exist_ok=True)
    
    # Check if we're on Apple Silicon
    import platform
    is_mac_arm = platform.system() == "Darwin" and platform.machine() == "arm64"
    
    print(f"Loading the model: {args.model_name_or_path}")
    
    if is_mac_arm:
        print("Detected Apple Silicon Mac. Using float16 precision instead of 4-bit quantization.")
        # Load model with float16 on Apple Silicon
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
        
        # Load model and tokenizer
        model = AutoModelForCausalLM.from_pretrained(
            args.model_name_or_path,
            quantization_config=bnb_config,
            device_map="auto",
            trust_remote_code=True,
            offload_folder=args.offload_folder,
        )
    
    tokenizer = AutoTokenizer.from_pretrained(
        args.model_name_or_path, 
        trust_remote_code=True,
        use_fast=True
    )
    tokenizer.pad_token = tokenizer.eos_token
    
    # Prepare model for training
    if is_mac_arm:
        # On Apple Silicon, we don't need to prepare for k-bit training since we're using float16
        pass
    else:
        # Prepare model for k-bit training
        model = prepare_model_for_kbit_training(model)
    
    # Configure LoRA
    peft_config = LoraConfig(
        r=args.lora_r,
        lora_alpha=args.lora_alpha,
        lora_dropout=args.lora_dropout,
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
        bias="none",
        task_type="CAUSAL_LM",
    )
    
    # Get PEFT model
    model = get_peft_model(model, peft_config)
    
    # Load the GSM8K dataset
    print("Loading the GSM8K dataset")
    dataset = load_dataset("gsm8k", "main")
    
    # Process the dataset
    train_dataset = dataset["train"].map(
        lambda x: process_data(x, tokenizer, args.max_seq_length),
        remove_columns=dataset["train"].column_names,
    )
    
    # Set training parameters
    training_args = TrainingArguments(
        output_dir=args.output_dir,
        per_device_train_batch_size=args.per_device_train_batch_size,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        learning_rate=args.learning_rate,
        num_train_epochs=args.num_train_epochs,
        warmup_ratio=0.03,
        save_strategy="epoch",
        logging_dir=f"{args.output_dir}/logs",
        logging_steps=10,
        save_total_limit=2,
        remove_unused_columns=False,
        push_to_hub=False,
        report_to="tensorboard",
        fp16=True,  # Use FP16 precision
    )
    
    # Initialize the Trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        tokenizer=tokenizer,
        data_collator=lambda data: {"input_ids": torch.stack([torch.tensor(f["input_ids"]) for f in data]),
                                   "attention_mask": torch.stack([torch.tensor(f["attention_mask"]) for f in data]),
                                   "labels": torch.stack([torch.tensor(f["labels"]) for f in data])},
    )
    
    # Train the model
    print("Starting training...")
    trainer.train()
    
    # Save the final model
    trainer.save_model()
    

if __name__ == "__main__":
    main() 