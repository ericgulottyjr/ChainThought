import os
import argparse
import json
import torch
import numpy as np
from datasets import Dataset
from transformers import AutoTokenizer, AutoModelForCausalLM, Trainer, TrainingArguments, DataCollatorForSeq2Seq
from peft import LoraConfig, get_peft_model
import wandb
from utils import set_seed, load_config


def load_jsonl(path: str):
    with open(path) as f:
        return [json.loads(line) for line in f]


def sample_dataset(dataset, sample_size, seed=42):
    """Sample a subset of the dataset with stratified sampling if possible"""
    if len(dataset) <= sample_size:
        return dataset
    
    import random
    random.seed(seed)
    # Simple random sampling for now
    return random.sample(dataset, sample_size)


def main(args):
    # Verify we're using the right GPU
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # Get GPU info
    if torch.cuda.is_available():
        print(f"CUDA device: {torch.cuda.get_device_name(0)}")
        print(f"CUDA memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.2f} GB")
    
    # Load config
    cfg = load_config(args.config)
    print(f"Loaded configuration from: {args.config}")
    
    # Set output directory
    output_dir = "outputs/basic_train/"
    print(f"Output directory: {output_dir}")
    os.makedirs(output_dir, exist_ok=True)

    # Set seed for reproducibility
    set_seed(cfg['training']['seed'])
    
    # Create run name
    run_name = f"basic_lora_r{cfg['model']['lora']['r']}_a{cfg['model']['lora']['alpha']}_lr{cfg['training']['learning_rate']}_bs{cfg['training']['per_device_train_batch_size']}"
    
    if 'dropout' in cfg['model']:
        run_name += f"_do{cfg['model']['dropout']}"
    else:
        cfg['model']['dropout'] = 0.1
        run_name += f"_do{cfg['model']['dropout']}"
        
    if args.small:
        run_name = f"small_{run_name}"
        
    # Initialize wandb
    wandb.init(
        project=cfg['wandb']['project'],
        entity=cfg['wandb']['entity'],
        config=cfg,
        name=run_name
    )
    print(f"W&B Run Name: {run_name}")

    # Load data
    train_list = load_jsonl('data/formatted/train.jsonl')
    val_list = load_jsonl('data/formatted/validation.jsonl')
    
    # Use small subset if requested
    if args.small:
        original_train_size = len(train_list)
        original_val_size = len(val_list)
        
        train_list = sample_dataset(train_list, 100, seed=cfg['training']['seed'])
        val_list = sample_dataset(val_list, 20, seed=cfg['training']['seed'])
        
        print(f"SMALL DATASET MODE: Using {len(train_list)}/{original_train_size} train examples")
        print(f"SMALL DATASET MODE: Using {len(val_list)}/{original_val_size} validation examples")
    else:
        print(f"FULL DATASET MODE: Using all {len(train_list)} train examples")
        print(f"FULL DATASET MODE: Using all {len(val_list)} validation examples")
    
    train_ds = Dataset.from_list(train_list)
    val_ds = Dataset.from_list(val_list)

    # Get tokenizer
    base_model = cfg['model']['base_model']
    cache_dir = cfg.get('cache', {}).get('hf_home', None)
    
    print(f"Loading tokenizer from {base_model}...")
    tokenizer = AutoTokenizer.from_pretrained(
        base_model, 
        use_fast=True, 
        cache_dir=cache_dir
    )
    
    # Set pad token if missing
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        print("Set tokenizer pad_token to eos_token")

    # Tokenization
    max_length = 512  # Use a smaller max_length to save memory
    print(f"Using sequence length: {max_length}")

    def tokenize_fn(ex):
        # Format as a single text string (input + target)
        full_text = ex['input'] + " " + ex['target']
        
        # Tokenize with explicit padding and truncation
        tokenized = tokenizer(
            full_text,
            padding='max_length',
            truncation=True,
            max_length=max_length,
            return_tensors=None  # Don't convert to tensors yet
        )
        
        # Set up labels as the input_ids (for causal LM)
        tokenized["labels"] = tokenized["input_ids"].copy()
        
        return tokenized

    # Tokenize datasets
    print("Tokenizing training data...")
    train_tokens = train_ds.map(
        tokenize_fn, 
        batched=False,
        remove_columns=['input', 'target']
    )
    print("Tokenizing validation data...")
    val_tokens = val_ds.map(
        tokenize_fn, 
        batched=False,
        remove_columns=['input', 'target']
    )
    
    print(f"Train dataset features: {train_tokens.features}")

    # Load model with memory-efficient settings
    print(f"Loading model {base_model}...")
    model = AutoModelForCausalLM.from_pretrained(
        base_model, 
        cache_dir=cache_dir,
        # Load in 8-bit to save memory
        load_in_8bit=True,
        device_map="auto",
        torch_dtype=torch.float16,  # Use fp16 precision
    )

    # Apply LoRA
    lora_r = cfg['model']['lora']['r']
    lora_alpha = cfg['model']['lora']['alpha']
    lora_dropout = cfg['model'].get('dropout', 0.1)
    
    print(f"Applying LoRA with r={lora_r}, alpha={lora_alpha}, dropout={lora_dropout}")
    
    # Target specific modules to save memory
    target_modules = ['q_proj', 'v_proj']
    print(f"Targeting modules: {', '.join(target_modules)}")
    
    lora_config = LoraConfig(
        r=lora_r,
        lora_alpha=lora_alpha,
        target_modules=target_modules,
        lora_dropout=lora_dropout,
        bias='none',
        task_type="CAUSAL_LM"
    )
    
    model = get_peft_model(model, lora_config)
    
    # Double-check trainable parameters
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    all_params = sum(p.numel() for p in model.parameters())
    print(f"Model has {trainable_params:,} trainable parameters out of {all_params:,} total parameters")
    
    # Enable gradient checkpointing if available
    if hasattr(model, "gradient_checkpointing_enable"):
        model.gradient_checkpointing_enable()
        print("Gradient checkpointing enabled")

    # Create Trainer
    print("Setting up Trainer...")
    
    # Data collator for padding
    data_collator = DataCollatorForSeq2Seq(
        tokenizer=tokenizer,
        padding=True,
        return_tensors="pt",
    )
    
    # Training arguments
    training_args = TrainingArguments(
        output_dir=output_dir,
        per_device_train_batch_size=1,  # Use small batch size
        gradient_accumulation_steps=16,  # Accumulate more steps
        learning_rate=float(cfg['training']['learning_rate']),
        num_train_epochs=cfg['training']['num_train_epochs'],
        
        # Evaluation and logging
        logging_steps=25,
        eval_steps=100,
        save_steps=200,
        save_total_limit=2,
        
        # Run configuration
        run_name=run_name,
        
        # Avoid OOM errors
        fp16=True,  # Use mixed precision
        gradient_checkpointing=True,
        
        # Other trainer settings
        remove_unused_columns=False,
        report_to="wandb",
        label_names=["labels"],
        
        # Optimizer settings
        weight_decay=0.01,
        optim="adamw_torch",
        adam_beta1=0.9,
        adam_beta2=0.999,
        adam_epsilon=1e-8,
        
        # Learning rate scheduler
        lr_scheduler_type="cosine",
        warmup_ratio=0.1,
        
        # Model selection
        load_best_model_at_end=False,
        metric_for_best_model="eval_loss",
        greater_is_better=False,
        
        # Do NOT use DataParallel to save memory
        dataloader_pin_memory=False,
        dataloader_drop_last=False,
    )
    
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_tokens,
        eval_dataset=val_tokens,
        data_collator=data_collator,
    )
    
    # Start training
    print("Starting training...")
    trainer.train()
    print("Training finished.")
    
    # Final evaluation
    print("Running final evaluation...")
    eval_results = trainer.evaluate()
    print(f"Final evaluation results: {eval_results}")
    
    # Save final model
    print("Saving final model...")
    final_model_path = os.path.join(output_dir, "final_model")
    trainer.save_model(final_model_path)
    tokenizer.save_pretrained(final_model_path)
    print(f"Model saved to {final_model_path}")
    
    wandb.finish()
    print("Done!")


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, default='default.yaml', help='Path to config file')
    parser.add_argument('--small', action='store_true', help='Use small dataset for quick testing')
    args = parser.parse_args()
    main(args) 