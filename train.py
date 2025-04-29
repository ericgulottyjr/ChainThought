import os
import argparse
import json
import torch
import numpy as np
import evaluate  # Import evaluate library
from datasets import Dataset
from transformers import Trainer, TrainingArguments
from utils import set_seed, load_config, get_tokenizer_and_model
import wandb


def load_jsonl(path: str):
    with open(path) as f:
        return [json.loads(line) for line in f]


def compute_metrics(eval_pred):
    # Load the accuracy metric
    accuracy_metric = evaluate.load("accuracy")
    
    # Predictions are logits, labels are the true IDs
    logits, labels = eval_pred
    
    # Get the predicted token IDs (highest logit)
    predictions = np.argmax(logits, axis=-1)
    
    # Filter out padding tokens (-100 is the default label ignore index)
    mask = labels != -100
    
    # Compute accuracy only on non-padding tokens
    results = accuracy_metric.compute(predictions=predictions[mask], references=labels[mask])
    return results


def main(args):
    # Get device info
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    n_gpus = torch.cuda.device_count()
    print(f"Using device: {device}, Found {n_gpus} GPUs")
    
    # Load config directly from the specified file
    cfg = load_config(args.config)
    print(f"Loaded configuration from: {args.config}")
    print(f"Running with parameters: {cfg}")

    set_seed(cfg['training']['seed'])
    
    # Initialize wandb for logging this single run
    run_name = f"direct_lora_r{cfg['model']['lora']['r']}_lr{cfg['training']['learning_rate']}_bs{cfg['training']['per_device_train_batch_size']}"
    if args.small:
        run_name = f"small_{run_name}"
        
    run = wandb.init(
        project=cfg['wandb']['project'],
        entity=cfg['wandb']['entity'],
        config=cfg, # Log the config used for this run
        name=run_name
    )
    print(f"W&B Run Name: {run_name}")

    # Load data
    train_list = load_jsonl('data/formatted/train.jsonl')
    val_list = load_jsonl('data/formatted/validation.jsonl')
    
    # Use small subset if requested (for quick testing)
    if args.small:
        original_train_size = len(train_list)
        original_val_size = len(val_list)
        
        train_list = train_list[:100]
        val_list = val_list[:20]
        
        print(f"SMALL DATASET MODE: Using {len(train_list)}/{original_train_size} train examples")
        print(f"SMALL DATASET MODE: Using {len(val_list)}/{original_val_size} validation examples")
    else:
        print(f"FULL DATASET MODE: Using all {len(train_list)} train examples")
        print(f"FULL DATASET MODE: Using all {len(val_list)} validation examples")
    
    train_ds = Dataset.from_list(train_list)
    val_ds = Dataset.from_list(val_list)

    # Get model & tokenizer using config from file
    tokenizer, model = get_tokenizer_and_model(cfg)
    
    # Let Trainer handle FP16 via TrainingArguments
    # model.gradient_checkpointing_enable() - Disabled
    print("Gradient Checkpointing DISABLED")
    
    # DataParallel for multi-GPU (if applicable)
    if n_gpus > 1:
        print(f"Using DataParallel across {n_gpus} GPUs")
        model = torch.nn.DataParallel(model)
    
    # Move model to device (GPU/CPU)
    model = model.to(device)

    # Max length
    max_length = 1024
    print(f"Using max_length: {max_length}")

    def tokenize_fn(ex):
        # Format as a single text string (input + target)
        # For causal language modeling, we want to predict the target given the input
        full_text = ex['input'] + " " + ex['target']
        
        # Tokenize without specifying text_target (which isn't working correctly)
        tokenized = tokenizer(
            full_text,
            padding='max_length',
            truncation=True,
            max_length=max_length,
            return_tensors="pt"
        )
        
        # Set up labels as the input_ids (for causal LM, labels are the same as inputs)
        tokenized["labels"] = tokenized["input_ids"].clone()
        
        # Remove batch dimension added by return_tensors="pt"
        return {k: v.squeeze(0) for k, v in tokenized.items()}

    # Tokenize
    print("Tokenizing training data...")
    train_tokens = train_ds.map(tokenize_fn, batched=False)  # Process one at a time for safety
    print("Tokenizing validation data...")
    val_tokens = val_ds.map(tokenize_fn, batched=False)
    
    print(f"Train dataset features: {train_tokens.features}")
    # print(f"Sample train data: {train_tokens[0]}") # Avoid printing large tensors

    # Configure training arguments using config from file
    training_args = TrainingArguments(
        output_dir=cfg['training']['output_dir'],
        per_device_train_batch_size=cfg['training']['per_device_train_batch_size'],
        gradient_accumulation_steps=cfg['training']['gradient_accumulation_steps'],
        learning_rate=float(cfg['training']['learning_rate']),
        num_train_epochs=cfg['training']['num_train_epochs'],
        logging_steps=50,  # Log more frequently for full dataset
        eval_steps=200,  # Evaluate less frequently for full dataset
        save_steps=200,  # Save less frequently for full dataset
        save_total_limit=3, # Keep a few more checkpoints
        run_name=run_name, # Use the specific run_name
        fp16=cfg['training'].get('fp16', False),
        remove_unused_columns=False,
        report_to='wandb', 
        # Ensure evaluation is enabled for compute_metrics to run
        # Use evaluation_strategy if your transformers version supports it
        # evaluation_strategy="steps", 
        # Otherwise, rely on eval_steps (implicitly enables evaluation in older versions)
        label_names=["labels"] # Explicitly tell Trainer the label column
    )

    # Callbacks (Early stopping disabled)
    callbacks = []
    print("Early stopping disabled due to library version incompatibility.")

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_tokens,
        eval_dataset=val_tokens,
        callbacks=callbacks,
        compute_metrics=compute_metrics  # Pass the metrics function
    )

    print("Starting training...")
    trainer.train()
    print("Training finished.")

    # Final evaluation
    print("Starting final evaluation...")
    eval_results = trainer.evaluate()
    print(f"Final evaluation results: {eval_results}")
    # Log final eval metrics to W&B
    wandb.log(eval_results)

    print("Saving final model...")
    trainer.save_model(os.path.join(training_args.output_dir, "final_checkpoint"))
    print("Model saved.")

    run.finish()
    print("W&B run finished.")

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, default='default.yaml', help='Path to config file')
    parser.add_argument('--small', action='store_true', help='Use small dataset for testing (NOT for final training)')
    args = parser.parse_args()
    main(args)