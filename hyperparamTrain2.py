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
from sklearn.model_selection import ParameterGrid
import copy
import random
from datetime import datetime
import gc  # Garbage collection for memory efficiency
import yaml
import time
import transformers

def load_jsonl(path: str):
    """Load data from a jsonl file"""
    with open(path) as f:
        return [json.loads(line) for line in f]

def compute_metrics(eval_pred):
    """Compute multiple metrics for evaluation"""
    # Load metrics
    accuracy_metric = evaluate.load("accuracy")
    precision_metric = evaluate.load("precision")
    recall_metric = evaluate.load("recall")
    f1_metric = evaluate.load("f1")
    
    # Extract predictions and labels
    logits, labels = eval_pred
    predictions = np.argmax(logits, axis=-1)
    
    # Filter out padding tokens (-100 is the default label ignore index)
    mask = labels != -100
    filtered_predictions = predictions[mask]
    filtered_labels = labels[mask]
    
    # Compute accuracy
    results = {}
    results.update(accuracy_metric.compute(predictions=filtered_predictions, references=filtered_labels))
    
    # Convert to binary for other metrics (correct/incorrect predictions)
    binary_preds = (filtered_predictions == filtered_labels).astype(int)
    binary_labels = np.ones_like(binary_preds)  # Ideal case is all correct predictions
    
    # Compute additional metrics (precision, recall, f1)
    results.update(precision_metric.compute(predictions=binary_preds, references=binary_labels, average='macro'))
    results.update(recall_metric.compute(predictions=binary_preds, references=binary_labels, average='macro'))
    results.update(f1_metric.compute(predictions=binary_preds, references=binary_labels, average='macro'))
    
    return results

def sample_dataset(dataset, sample_size, seed=42):
    """Sample a subset of the dataset with stratified sampling if possible"""
    if len(dataset) <= sample_size:
        return dataset
    
    random.seed(seed)
    # Simple random sampling for now
    return random.sample(dataset, sample_size)

def get_enhanced_tokenizer_and_model(base_model, hyperparams):
    """Get tokenizer and model with enhanced configuration from hyperparameters"""
    cache_dir = None  # Set appropriately if needed

    # Initialize tokenizer
    tokenizer = AutoTokenizer.from_pretrained(base_model, use_fast=True, cache_dir=cache_dir)
    
    # Set pad token if missing (common for Llama-based models)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        print("Set tokenizer pad_token to eos_token")

    # Load model
    model = AutoModelForCausalLM.from_pretrained(
        base_model, 
        cache_dir=cache_dir,
    )

    # Get LoRA hyperparameters
    lora_r = hyperparams['lora_r']
    lora_alpha = hyperparams['lora_alpha']
    lora_dropout = hyperparams['dropout']
    
    # Apply enhanced LoRA configuration
    lora_cfg = LoraConfig(
        r=lora_r,
        lora_alpha=lora_alpha,
        # Target more modules for better fine-tuning (from train2.py)
        target_modules=['q_proj', 'k_proj', 'v_proj', 'o_proj'],
        lora_dropout=lora_dropout,
        bias='none',
        task_type="CAUSAL_LM"
    )
    
    print(f"Using LoRA with r={lora_r}, alpha={lora_alpha}, dropout={lora_dropout}")
    print(f"Targeting modules: q_proj, k_proj, v_proj, o_proj")
    
    # Get PEFT model
    model = get_peft_model(model, lora_cfg)
    
    # Important: Enable requires_grad on the trainable parameters
    for param in model.parameters():
        if param.requires_grad:
            if not param.requires_grad:
                param.requires_grad = True
    
    return tokenizer, model

def free_memory():
    """Free up memory by clearing cache and running garbage collection"""
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    gc.collect()

def main(args):
    """Main function to run hyperparameter search"""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    n_gpus = torch.cuda.device_count()
    print(f"Using device: {device}, Found {n_gpus} GPUs")
    
    # Load config
    cfg = load_config(args.config)
    print(f"Loaded configuration from: {args.config}")
    
    # Define base output directory
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    base_output_dir = f"outputs/hyperparamTrain2_{timestamp}"
    os.makedirs(base_output_dir, exist_ok=True)
    print(f"Saving all outputs to: {base_output_dir}")
    
    # Define hyperparameter grid
    hyperparameter_grid = {
        'lora_r': [8, 16, 32],
        'lora_alpha': [16, 32, 64],
        'learning_rate': [1e-4, 2e-4, 3e-4],
        'batch_size': [1, 2, 4],
        'dropout': [0.0, 0.1, 0.2],
        'weight_decay': [0.0, 0.01, 0.05],
        'max_length': [512, 768]
    }
    
    # For quick testing with smaller grid
    if args.small_grid:
        small_grid = {k: [v[1]] for k, v in hyperparameter_grid.items()}  # Use middle values
        hyperparameter_grid = small_grid
        print("Using small hyperparameter grid for quick testing")
    
    # Create parameter grid
    param_grid = list(ParameterGrid(hyperparameter_grid))
    num_configs = len(param_grid)
    print(f"Testing {num_configs} hyperparameter combinations")
    
    # Load datasets and sample
    train_list = load_jsonl('data/formatted/train.jsonl')
    val_list = load_jsonl('data/formatted/validation.jsonl')
    
    # Sample dataset based on args
    train_sample_size = args.sample_size if args.sample_size > 0 else 100
    val_sample_size = max(20, int(train_sample_size * 0.2))
    
    print(f"SAMPLED DATASET: Using {train_sample_size}/{len(train_list)} train examples")
    train_list = sample_dataset(train_list, train_sample_size, seed=cfg['training']['seed'])
    
    print(f"SAMPLED VALIDATION: Using {val_sample_size}/{len(val_list)} validation examples")
    val_list = sample_dataset(val_list, val_sample_size, seed=cfg['training']['seed'])
    
    train_ds = Dataset.from_list(train_list)
    val_ds = Dataset.from_list(val_list)
    
    # Define training epochs based on sample size (fewer epochs for larger samples)
    if train_sample_size <= 100:
        num_epochs = 5  # Use more epochs for very small samples
    elif train_sample_size <= 500:
        num_epochs = 3  # Moderate number of epochs for medium samples
    else:
        num_epochs = 2  # Fewer epochs for larger samples
    
    # Track best model performance
    best_model_info = {
        'eval_loss': float('inf'),
        'eval_accuracy': 0.0,
        'hyperparams': {},
        'output_dir': ''
    }
    
    # Save the hyperparameter grid and configuration
    hyperparams_file = os.path.join(base_output_dir, "hyperparameter_grid.json")
    with open(hyperparams_file, 'w') as f:
        json.dump({
            'grid': hyperparameter_grid,
            'num_configs': num_configs,
            'train_sample_size': train_sample_size,
            'val_sample_size': val_sample_size,
            'num_epochs': num_epochs
        }, f, indent=2)
    
    # Initialize W&B for the entire search
    if not args.no_wandb:
        project_name = f"{cfg['wandb']['project']}-hyperparam2"
        search_run = wandb.init(
            project=project_name,
            entity=cfg['wandb']['entity'],
            name=f"hyperparam2-search-{timestamp}",
            job_type="hyperparameter-search"
        )
    
    # Run grid search
    start_time = time.time()
    for hp_idx, hyperparams in enumerate(param_grid):
        print(f"\n{'='*80}")
        print(f"Training configuration {hp_idx+1}/{num_configs}")
        print(f"Hyperparameters: {hyperparams}")
        print(f"{'='*80}\n")
        
        config_start_time = time.time()
        
        # Clear memory before each run
        free_memory()
        
        # Set the max_length for this run
        max_length = hyperparams['max_length']
        print(f"Using max_length: {max_length}")
        
        # Set seed for reproducibility
        seed = cfg['training']['seed'] + hp_idx  # Use different seed for each config
        set_seed(seed)
        
        # Prepare run name and output directory
        hp_dir = f"r{hyperparams['lora_r']}_a{hyperparams['lora_alpha']}_lr{hyperparams['learning_rate']}_bs{hyperparams['batch_size']}_do{hyperparams['dropout']}_wd{hyperparams['weight_decay']}_ml{hyperparams['max_length']}"
        output_dir = os.path.join(base_output_dir, f"hp_search/{hp_dir}")
        os.makedirs(output_dir, exist_ok=True)
        
        # Initialize W&B for this specific run
        if not args.no_wandb:
            run_name = f"hp2_{hp_dir}"
            run = wandb.init(
                project=cfg['wandb']['project'],
                entity=cfg['wandb']['entity'],
                config=hyperparams,
                name=run_name,
                group="hyperparameter-search-train2",
                reinit=True  # Allow reinitializing for multiple runs
            )
        
        # Get tokenizer and model
        tokenizer, model = get_enhanced_tokenizer_and_model(cfg['model']['base_model'], hyperparams)
        model = model.to(device)
        
        # Explicitly set use_cache to False for training stability
        if hasattr(model.config, "use_cache"):
            model.config.use_cache = False
            print("Disabled model.config.use_cache for training stability")
        
        # Print trainable parameters
        trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        all_params = sum(p.numel() for p in model.parameters())
        print(f"Model has {trainable_params:,} trainable parameters out of {all_params:,} total parameters")
        
        # Tokenization function
        def tokenize_fn(ex):
            full_text = ex['input'] + " " + ex['target']
            tokenized = tokenizer(
                full_text,
                padding='max_length',
                truncation=True,
                max_length=max_length,
                return_tensors=None
            )
            tokenized["labels"] = tokenized["input_ids"].copy()
            return tokenized
        
        # Tokenize datasets
        print("Tokenizing datasets...")
        train_tokens = train_ds.map(
            tokenize_fn, 
            batched=False,
            remove_columns=['input', 'target']
        )
        val_tokens = val_ds.map(
            tokenize_fn, 
            batched=False,
            remove_columns=['input', 'target']
        )
        
        # Create data collator
        data_collator = transformers.DataCollatorForSeq2Seq(
            tokenizer=tokenizer,
            padding=True,
            return_tensors="pt",
        )
        
        # Training arguments - optimized for efficiency
        training_args = TrainingArguments(
            output_dir=output_dir,
            per_device_train_batch_size=hyperparams['batch_size'],
            gradient_accumulation_steps=1,  # Reduced for faster iterations
            learning_rate=hyperparams['learning_rate'],
            num_train_epochs=num_epochs,
            logging_steps=10,  # More frequent logging for small datasets
            eval_steps=50,  # More frequent evaluation 
            save_steps=100,
            save_total_limit=1,  # Only keep best checkpoint
            run_name=f"hp2_{hp_dir}" if not args.no_wandb else None,
            fp16=False,  # Avoid FP16 gradient issues
            remove_unused_columns=False,
            report_to='wandb' if not args.no_wandb else "none",
            label_names=["labels"],
            weight_decay=hyperparams['weight_decay'],
            optim="adamw_torch",
            adam_beta1=0.9,
            adam_beta2=0.999,
            adam_epsilon=1e-8,
            lr_scheduler_type="cosine",
            warmup_ratio=0.1,
            load_best_model_at_end=False,  # Avoid library compatibility issues
            metric_for_best_model="eval_loss",
            greater_is_better=False,
            dataloader_drop_last=True,
            gradient_checkpointing=False,  # Disabled to avoid requires_grad issues
            torch_compile=False,
        )
        
        # Create trainer
        trainer = Trainer(
            model=model,
            args=training_args,
            train_dataset=train_tokens,
            eval_dataset=val_tokens,
            compute_metrics=compute_metrics,
            data_collator=data_collator,
        )
        
        print(f"Starting training for hyperparameter configuration {hp_idx+1}/{num_configs}...")
        trainer.train()
        
        # Final evaluation
        print("Running final evaluation...")
        eval_results = trainer.evaluate()
        print(f"Evaluation results: {eval_results}")
        
        # Calculate training time
        config_end_time = time.time()
        config_duration = config_end_time - config_start_time
        print(f"Configuration {hp_idx+1} training time: {config_duration:.2f} seconds")
        
        # Save config, hyperparams and results as JSON
        results_path = os.path.join(output_dir, "results.json")
        with open(results_path, "w") as f:
            json.dump({
                "hyperparameters": hyperparams,
                "results": eval_results,
                "training_time": config_duration
            }, f, indent=2)
        
        # Log to W&B
        if not args.no_wandb:
            wandb.log({
                **eval_results,
                **{f"hp_{k}": v for k, v in hyperparams.items()},
                "training_time": config_duration
            })
        
        # Track best model
        if eval_results["eval_loss"] < best_model_info["eval_loss"]:
            best_model_info["eval_loss"] = eval_results["eval_loss"]
            best_model_info["eval_accuracy"] = eval_results.get("eval_accuracy", 0.0)
            best_model_info["hyperparams"] = hyperparams
            best_model_info["output_dir"] = output_dir
            
            # Create a symlink to the best model for easy access
            best_model_path = os.path.join(base_output_dir, "best_model")
            if os.path.exists(best_model_path):
                if os.path.islink(best_model_path):
                    os.unlink(best_model_path)
                else:
                    # It's a directory, remove it
                    import shutil
                    shutil.rmtree(best_model_path)
            os.symlink(output_dir, best_model_path)
            
            print(f"New best model found: {best_model_info}")
        
        # Delete the model to free up memory
        del model, tokenizer, trainer, training_args
        free_memory()  # Explicit memory cleanup
        
        # Finish this W&B run if enabled
        if not args.no_wandb:
            run.finish()
        
        # Save updated best model info after each configuration
        best_config_path = os.path.join(base_output_dir, "current_best_hyperparams.json")
        with open(best_config_path, "w") as f:
            json.dump(best_model_info, f, indent=2)
    
    # Total search time
    end_time = time.time()
    total_duration = end_time - start_time
    hours, remainder = divmod(total_duration, 3600)
    minutes, seconds = divmod(remainder, 60)
    
    print(f"Total search time: {int(hours)}h {int(minutes)}m {seconds:.2f}s")
    
    # Log best model information to the project-level run
    if not args.no_wandb:
        wandb.log({
            "best_eval_loss": best_model_info["eval_loss"],
            "best_eval_accuracy": best_model_info["eval_accuracy"],
            "total_search_time": total_duration,
            **{f"best_hp_{k}": v for k, v in best_model_info["hyperparams"].items()}
        })
    
    # Save the final best hyperparameter configuration
    best_config_path = os.path.join(base_output_dir, "best_hyperparams.json")
    with open(best_config_path, "w") as f:
        json.dump(best_model_info, f, indent=2)
    
    # Create a train2.yaml config file with the best hyperparameters
    best_hp = best_model_info["hyperparams"]
    train2_config = copy.deepcopy(cfg)
    train2_config["model"]["lora"] = {
        "r": best_hp["lora_r"],
        "alpha": best_hp["lora_alpha"]
    }
    train2_config["model"]["dropout"] = best_hp["dropout"]
    train2_config["training"]["learning_rate"] = best_hp["learning_rate"]
    train2_config["training"]["per_device_train_batch_size"] = best_hp["batch_size"]
    train2_config["training"]["weight_decay"] = best_hp["weight_decay"]
    
    # Save the best config for easy use with train2.py
    best_config_yaml_path = os.path.join(base_output_dir, "train2_best_config.yaml")
    with open(best_config_yaml_path, 'w') as f:
        yaml.dump(train2_config, f, default_flow_style=False)
    
    print("\n" + "="*80)
    print("Hyperparameter search completed!")
    print(f"Best configuration: {best_model_info['hyperparams']}")
    print(f"Best validation loss: {best_model_info['eval_loss']}")
    print(f"Best validation accuracy: {best_model_info['eval_accuracy']}")
    print(f"Best model saved at: {best_model_info['output_dir']}")
    print(f"Ready-to-use config for train2.py saved at: {best_config_yaml_path}")
    print("="*80 + "\n")
    
    if not args.no_wandb:
        search_run.finish()

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Hyperparameter search for train2.py')
    parser.add_argument('--config', type=str, default='default.yaml', help='Path to base config file')
    parser.add_argument('--sample_size', type=int, default=100, help='Number of examples to sample from the dataset (default: 100)')
    parser.add_argument('--small_grid', action='store_true', help='Use small hyperparameter grid for testing')
    parser.add_argument('--no_wandb', action='store_true', help='Disable WandB logging')
    
    args = parser.parse_args()
    main(args)