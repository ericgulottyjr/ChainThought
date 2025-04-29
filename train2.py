import os
import argparse
import json
import torch
import numpy as np
import evaluate
from datasets import Dataset
from transformers import Trainer, TrainingArguments
from utils import set_seed, load_config
import wandb
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import LoraConfig, get_peft_model
import torch.nn.functional as F
from torch.optim.lr_scheduler import CosineAnnealingLR, OneCycleLR
from accelerate import Accelerator


def load_jsonl(path: str):
    with open(path) as f:
        return [json.loads(line) for line in f]


def compute_metrics(eval_pred):
    # Load multiple metrics
    accuracy_metric = evaluate.load("accuracy")
    precision_metric = evaluate.load("precision")
    recall_metric = evaluate.load("recall")
    f1_metric = evaluate.load("f1")
    
    # Predictions are logits, labels are the true IDs
    logits, labels = eval_pred
    
    # Get the predicted token IDs (highest logit)
    predictions = np.argmax(logits, axis=-1)
    
    # Filter out padding tokens (-100 is the default label ignore index)
    mask = labels != -100
    filtered_predictions = predictions[mask]
    filtered_labels = labels[mask]
    
    # Compute accuracy only on non-padding tokens
    results = {}
    results.update(accuracy_metric.compute(predictions=filtered_predictions, references=filtered_labels))
    
    # Convert to binary for other metrics (correct/incorrect predictions)
    binary_preds = (filtered_predictions == filtered_labels).astype(int)
    binary_labels = np.ones_like(binary_preds)  # Ideal case is all correct predictions
    
    # Compute additional metrics
    results.update(precision_metric.compute(predictions=binary_preds, references=binary_labels, average='macro'))
    results.update(recall_metric.compute(predictions=binary_preds, references=binary_labels, average='macro'))
    results.update(f1_metric.compute(predictions=binary_preds, references=binary_labels, average='macro'))
    
    return results


# Custom function to get tokenizer and model with enhanced config options
def get_enhanced_tokenizer_and_model(cfg: dict):
    base = cfg['model']['base_model']
    cache_dir = cfg.get('cache', {}).get('hf_home', None)

    # Initialize tokenizer
    tokenizer = AutoTokenizer.from_pretrained(base, use_fast=True, cache_dir=cache_dir)
    
    # Set pad token if missing (common for Llama-based models)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        print("Set tokenizer pad_token to eos_token")

    # Load model with more efficient settings
    model = AutoModelForCausalLM.from_pretrained(
        base, 
        cache_dir=cache_dir,
        # Use full precision to avoid FP16 gradient issues
        # torch_dtype=torch.float16 if cfg.get('training', {}).get('fp16', False) else torch.float32
    )

    # Get the dropout value from config or use default
    lora_dropout = cfg['model'].get('dropout', 0.1)  # Higher dropout than default
    
    # Apply enhanced LoRA configuration
    lora_cfg = LoraConfig(
        r=cfg['model']['lora']['r'],
        lora_alpha=cfg['model']['lora']['alpha'],
        # Target more modules for better fine-tuning
        target_modules=['q_proj', 'k_proj', 'v_proj', 'o_proj'],
        lora_dropout=lora_dropout,
        bias='none',
        task_type="CAUSAL_LM"
    )
    
    print(f"Using LoRA with dropout={lora_dropout}, r={cfg['model']['lora']['r']}, alpha={cfg['model']['lora']['alpha']}")
    print(f"Targeting modules: q_proj, k_proj, v_proj, o_proj")
    
    model = get_peft_model(model, lora_cfg)
    return tokenizer, model


def sample_dataset(dataset, sample_size, seed=42):
    """Sample a subset of the dataset with stratified sampling if possible"""
    if len(dataset) <= sample_size:
        return dataset
    
    import random
    random.seed(seed)
    # Simple random sampling for now
    return random.sample(dataset, sample_size)


def main(args):
    # Initialize accelerator first
    accelerator = Accelerator()
    
    # Get device info - use accelerator's device instead
    device = accelerator.device
    if accelerator.is_main_process:
        print(f"Using device: {device}")
        print(f"Number of processes: {accelerator.num_processes}")
        print(f"Distributed type: {accelerator.distributed_type}")
    
    # Rest of device detection can stay for info purposes
    n_gpus = torch.cuda.device_count()
    if accelerator.is_main_process:
        print(f"Found {n_gpus} GPUs")
    
    # Load config
    cfg = load_config(args.config)
    if accelerator.is_main_process:
        print(f"Loaded configuration from: {args.config}")
    
    # Override the output directory to avoid overwriting existing checkpoints
    original_output_dir = cfg['training']['output_dir']
    cfg['training']['output_dir'] = "outputs/train2/"
    if accelerator.is_main_process:
        print(f"IMPORTANT: Changed output directory from {original_output_dir} to {cfg['training']['output_dir']}")
    
    # Make sure the output directory exists
    if accelerator.is_main_process:
        os.makedirs(cfg['training']['output_dir'], exist_ok=True)

    # Set seed for reproducibility
    set_seed(cfg['training']['seed'])
    
    # Enhanced run name with more hyperparameter details
    run_name = f"train2_lora_r{cfg['model']['lora']['r']}_a{cfg['model']['lora']['alpha']}_lr{cfg['training']['learning_rate']}_bs{cfg['training']['per_device_train_batch_size']}"
    
    # Add dropout to run name if it exists in config
    if 'dropout' in cfg['model']:
        run_name += f"_do{cfg['model']['dropout']}"
    else:
        # Set a default dropout value if not provided
        cfg['model']['dropout'] = 0.1
        run_name += f"_do{cfg['model']['dropout']}"
        
    # Add small flag to run name
    if args.small:
        run_name = f"small_{run_name}"
        
    # Initialize wandb - only on main process
    if accelerator.is_main_process:
        run = wandb.init(
            project=cfg['wandb']['project'],
            entity=cfg['wandb']['entity'],
            config=cfg,  # Log the config used for this run
            name=run_name
        )
        print(f"W&B Run Name: {run_name}")
    else:
        run = None

    # Load data
    train_list = load_jsonl('data/formatted/train.jsonl')
    val_list = load_jsonl('data/formatted/validation.jsonl')
    
    # Use small subset if requested (for quick testing)
    if args.small:
        original_train_size = len(train_list)
        original_val_size = len(val_list)
        
        # Sample 100 examples for training
        train_list = sample_dataset(train_list, 100, seed=cfg['training']['seed'])
        # Sample 20 examples for validation
        val_list = sample_dataset(val_list, 20, seed=cfg['training']['seed'])
        
        if accelerator.is_main_process:
            print(f"SMALL DATASET MODE: Using {len(train_list)}/{original_train_size} train examples")
            print(f"SMALL DATASET MODE: Using {len(val_list)}/{original_val_size} validation examples")
    else:
        if accelerator.is_main_process:
            print(f"FULL DATASET MODE: Using all {len(train_list)} train examples")
            print(f"FULL DATASET MODE: Using all {len(val_list)} validation examples")
    
    train_ds = Dataset.from_list(train_list)
    val_ds = Dataset.from_list(val_list)

    # Get enhanced model & tokenizer with improved dropout and configuration
    tokenizer, model = get_enhanced_tokenizer_and_model(cfg)
    
    # DISABLE gradient checkpointing as it's causing issues with gradient propagation
    # model.gradient_checkpointing_enable()
    if accelerator.is_main_process:
        print("Gradient Checkpointing DISABLED to avoid gradient computation issues")
    
    # Ensure parameters require gradients
    for param in model.parameters():
        param.requires_grad = True
    
    # No DataParallel - accelerate handles distributed training
    # Don't place on device manually - accelerate will handle it
    model.train()
    
    # Print parameter grad status for debugging
    if accelerator.is_main_process:
        trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        all_params = sum(p.numel() for p in model.parameters())
        print(f"Model has {trainable_params:,} trainable parameters out of {all_params:,} total parameters")

    # Reduced max_length to help with repetition issues
    # Based on analysis of example outputs that showed repetition
    max_length = 768  # Reduced from 1024
    if accelerator.is_main_process:
        print(f"Using reduced max_length: {max_length} (was 1024 in original)")

    def tokenize_fn(ex):
        # Format as a single text string (input + target)
        full_text = ex['input'] + " " + ex['target']
        
        # Enhanced tokenization with better padding and truncation
        tokenized = tokenizer(
            full_text,
            padding='max_length',
            truncation=True,
            max_length=max_length,
            return_tensors="pt"
        )
        
        # Set up labels as the input_ids (for causal LM)
        tokenized["labels"] = tokenized["input_ids"].clone()
        
        # Remove batch dimension
        return {k: v.squeeze(0) for k, v in tokenized.items()}

    # Tokenize datasets
    if accelerator.is_main_process:
        print("Tokenizing training data...")
    train_tokens = train_ds.map(tokenize_fn, batched=False)
    if accelerator.is_main_process:
        print("Tokenizing validation data...")
    val_tokens = val_ds.map(tokenize_fn, batched=False)
    
    if accelerator.is_main_process:
        print(f"Train dataset features: {train_tokens.features}")

    # Enhanced training arguments - keep all existing parameters
    training_args = TrainingArguments(
        output_dir=cfg['training']['output_dir'],
        per_device_train_batch_size=cfg['training']['per_device_train_batch_size'],
        gradient_accumulation_steps=cfg['training']['gradient_accumulation_steps'],
        learning_rate=float(cfg['training']['learning_rate']),
        num_train_epochs=cfg['training']['num_train_epochs'],
        
        # More frequent logging and evaluation
        logging_steps=25,
        eval_steps=100,
        save_steps=200,
        save_total_limit=3,
        
        # Run configuration
        run_name=run_name,
        
        # Disable mixed precision to avoid FP16 gradient issues
        fp16=False,
        
        # Other improvements
        remove_unused_columns=False,
        report_to='wandb' if accelerator.is_main_process else "none",
        label_names=["labels"],
        
        # Weight decay for regularization
        weight_decay=0.01,
        
        # Use AdamW optimizer with better parameters
        optim="adamw_torch",
        adam_beta1=0.9,
        adam_beta2=0.999,
        adam_epsilon=1e-8,
        
        # Learning rate scheduler
        lr_scheduler_type="cosine",
        warmup_ratio=0.1,  # Warm up for 10% of training
        
        # Better model selection - setting to False due to strategy mismatch error
        load_best_model_at_end=False,
        metric_for_best_model="eval_loss",
        greater_is_better=False,
    )

    # Initialize trainer with enhanced metrics
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_tokens,
        eval_dataset=val_tokens,
        compute_metrics=compute_metrics,
        tokenizer=tokenizer,  # Add tokenizer for better integration with accelerate
    )

    if accelerator.is_main_process:
        print("Starting training with enhanced techniques...")
    trainer.train()
    if accelerator.is_main_process:
        print("Training finished.")

    # Final evaluation - only log on main process
    if accelerator.is_main_process:
        print("Starting final evaluation...")
    eval_results = trainer.evaluate()
    if accelerator.is_main_process:
        print(f"Final evaluation results: {eval_results}")
        # Log final eval metrics to W&B
        if run:
            wandb.log(eval_results)

    # Save final model - only on main process
    if accelerator.is_main_process:
        print("Saving final model...")
        final_model_path = os.path.join(cfg['training']['output_dir'], "final_model")
        trainer.save_model(final_model_path)
        print(f"Model saved to {final_model_path}")
        
        # Save training configuration for reproducibility
        config_path = os.path.join(cfg['training']['output_dir'], "train2_config.yaml")
        with open(config_path, 'w') as f:
            import yaml
            yaml.dump(cfg, f)
        print(f"Configuration saved to {config_path}")

        if run:
            run.finish()
            print("W&B run finished.")

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, default='default.yaml', help='Path to config file')
    parser.add_argument('--small', action='store_true', help='Use small dataset for quick testing')
    args = parser.parse_args()
    main(args) 