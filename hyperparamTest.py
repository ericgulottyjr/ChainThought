import os
import argparse
import json
import torch
import numpy as np
import evaluate
from datasets import Dataset
from transformers import Trainer, TrainingArguments
from utils import set_seed, load_config, get_tokenizer_and_model
import wandb
from sklearn.model_selection import ParameterGrid
import copy
import random

def load_jsonl(path: str):
    with open(path) as f:
        return [json.loads(line) for line in f]

def compute_metrics(eval_pred):
    # Load metrics
    accuracy = evaluate.load("accuracy")
    precision = evaluate.load("precision")
    recall = evaluate.load("recall")
    f1 = evaluate.load("f1")
    
    logits, labels = eval_pred
    predictions = np.argmax(logits, axis=-1)
    
    # Filter out padding tokens
    mask = labels != -100
    filtered_predictions = predictions[mask]
    filtered_labels = labels[mask]
    
    # Compute all metrics
    results = {}
    results.update(accuracy.compute(predictions=filtered_predictions, references=filtered_labels))
    
    # For precision, recall, and F1, we need binary classification
    # For token-level tasks, we can check if the prediction is correct
    binary_preds = (filtered_predictions == filtered_labels).astype(int)
    binary_labels = np.ones_like(binary_preds)  # The ideal case is all correct predictions
    
    results.update(precision.compute(predictions=binary_preds, references=binary_labels, average='macro'))
    results.update(recall.compute(predictions=binary_preds, references=binary_labels, average='macro'))
    results.update(f1.compute(predictions=binary_preds, references=binary_labels, average='macro'))
    
    return results

def sample_dataset(dataset, sample_size, seed=42):
    """Sample a subset of the dataset using stratified sampling if possible"""
    if len(dataset) <= sample_size:
        return dataset
    
    random.seed(seed)
    return random.sample(dataset, sample_size)

def main(args):
    # Get device info
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    n_gpus = torch.cuda.device_count()
    print(f"Using device: {device}, Found {n_gpus} GPUs")
    
    # Load config
    cfg = load_config(args.config)
    print(f"Loaded configuration from: {args.config}")
    
    # Define base output directory to avoid overwriting existing checkpoints
    base_output_dir = "outputs/hyperparamTest"
    if not os.path.exists(base_output_dir):
        os.makedirs(base_output_dir, exist_ok=True)
    print(f"Saving all outputs to: {base_output_dir}")
    
    # Define a comprehensive hyperparameter grid
    hyperparameter_grid = {
        'lora_r': [8, 16, 32],
        'lora_alpha': [16, 32, 64],
        'learning_rate': [1e-4, 2e-4, 3e-4],
        'batch_size': [1, 2],
        'dropout': [0.0, 0.1, 0.2],
        'weight_decay': [0.0, 0.01],
        'adam_beta1': [0.9],
        'adam_beta2': [0.999],
        'max_length': [512, 768]  # Adding max_length as a hyperparameter to reduce repetition
    }
    
    # If small_grid is requested, use a smaller subset of hyperparameters
    if args.small_grid:
        # Extract just one value from each hyperparameter for testing
        small_grid = {}
        for param, values in hyperparameter_grid.items():
            small_grid[param] = [values[0]]  # Just take the first value from each parameter
        hyperparameter_grid = small_grid
        print("Using small hyperparameter grid for quick testing")
    
    # Create the parameter grid
    param_grid = list(ParameterGrid(hyperparameter_grid))
    print(f"Testing {len(param_grid)} hyperparameter combinations")
    
    # Load data
    train_list = load_jsonl('data/formatted/train.jsonl')
    val_list = load_jsonl('data/formatted/validation.jsonl')
    
    # Sample dataset based on args
    sample_size = args.sample_size if args.sample_size > 0 else (100 if args.small else len(train_list))
    val_sample_size = max(20, int(sample_size * 0.2))  # 20% of train size or minimum 20
    
    if sample_size < len(train_list):
        print(f"SAMPLED DATASET: Using {sample_size}/{len(train_list)} train examples")
        train_list = sample_dataset(train_list, sample_size, seed=cfg['training']['seed'])
    else:
        print(f"FULL DATASET: Using all {len(train_list)} train examples")
    
    if val_sample_size < len(val_list):
        print(f"SAMPLED VALIDATION: Using {val_sample_size}/{len(val_list)} validation examples")
        val_list = sample_dataset(val_list, val_sample_size, seed=cfg['training']['seed'])
    else:
        print(f"FULL VALIDATION: Using all {len(val_list)} validation examples")
    
    train_ds = Dataset.from_list(train_list)
    val_ds = Dataset.from_list(val_list)
    
    # Track best model performance
    best_model_info = {
        'eval_loss': float('inf'),
        'eval_accuracy': 0.0,
        'hyperparams': {},
        'output_dir': ''
    }
    
    # Initialize a project-level W&B run for hyperparameter search
    project_name = f"{cfg['wandb']['project']}-hyperparameter-search"
    search_run = wandb.init(
        project=project_name,
        entity=cfg['wandb']['entity'],
        name="hyperparameter-search",
        job_type="hyperparameter-search"
    )
    
    # Run grid search
    for hp_idx, hyperparams in enumerate(param_grid):
        print(f"\n{'='*50}")
        print(f"Training configuration {hp_idx+1}/{len(param_grid)}")
        print(f"Hyperparameters: {hyperparams}")
        print(f"{'='*50}\n")
        
        # Set the max_length for this run
        max_length = hyperparams['max_length']
        print(f"Using max_length: {max_length}")
        
        # Update config with current hyperparameters
        current_cfg = copy.deepcopy(cfg)
        current_cfg['model']['lora']['r'] = hyperparams['lora_r']
        current_cfg['model']['lora']['alpha'] = hyperparams['lora_alpha']
        current_cfg['training']['learning_rate'] = hyperparams['learning_rate']
        current_cfg['training']['per_device_train_batch_size'] = hyperparams['batch_size']
        
        # Add or update dropout in the config
        if 'dropout' not in current_cfg['model']:
            current_cfg['model']['dropout'] = hyperparams['dropout']
        else:
            current_cfg['model']['dropout'] = hyperparams['dropout']
        
        # Set seed for reproducibility
        set_seed(current_cfg['training']['seed'])
        
        # Get tokenizer and model
        tokenizer, model = get_tokenizer_and_model(current_cfg)
        
        # Prepare output directory for this configuration
        hp_dir = f"r{hyperparams['lora_r']}_a{hyperparams['lora_alpha']}_lr{hyperparams['learning_rate']}_bs{hyperparams['batch_size']}_do{hyperparams['dropout']}_wd{hyperparams['weight_decay']}_ml{hyperparams['max_length']}"
        output_dir = os.path.join(base_output_dir, f"hp_search/{hp_dir}")
        os.makedirs(output_dir, exist_ok=True)
        
        # Initialize W&B for this specific run
        run_name = f"lora_r{hyperparams['lora_r']}_a{hyperparams['lora_alpha']}_lr{hyperparams['learning_rate']}_bs{hyperparams['batch_size']}_do{hyperparams['dropout']}_wd{hyperparams['weight_decay']}_ml{hyperparams['max_length']}"
        if args.small:
            run_name = f"small_{run_name}"
        
        run = wandb.init(
            project=cfg['wandb']['project'],
            entity=cfg['wandb']['entity'],
            config=current_cfg,
            name=run_name,
            group="hyperparameter-search"
        )
        
        # Move model to device
        if n_gpus > 1:
            print(f"Using DataParallel across {n_gpus} GPUs")
            model = torch.nn.DataParallel(model)
        model = model.to(device)
        
        # Create tokenization function
        def tokenize_fn(ex):
            full_text = ex['input'] + " " + ex['target']
            
            tokenized = tokenizer(
                full_text,
                padding='max_length',
                truncation=True,
                max_length=max_length,
                return_tensors="pt"
            )
            
            tokenized["labels"] = tokenized["input_ids"].clone()
            return {k: v.squeeze(0) for k, v in tokenized.items()}
        
        # Tokenize the data
        print("Tokenizing datasets...")
        train_tokens = train_ds.map(tokenize_fn, batched=False)
        val_tokens = val_ds.map(tokenize_fn, batched=False)
        
        # Training arguments
        training_args = TrainingArguments(
            output_dir=output_dir,
            per_device_train_batch_size=hyperparams['batch_size'],
            gradient_accumulation_steps=cfg['training']['gradient_accumulation_steps'],
            learning_rate=hyperparams['learning_rate'],
            num_train_epochs=cfg['training']['num_train_epochs'],
            logging_steps=25,  # More frequent logging for hyperparameter search
            eval_steps=100,
            save_steps=200,
            save_total_limit=1,  # Only keep best model to save space
            run_name=run_name,
            fp16=cfg['training'].get('fp16', False),
            remove_unused_columns=False,
            report_to='wandb',
            label_names=["labels"],
            load_best_model_at_end=False,
            metric_for_best_model="eval_loss",
            greater_is_better=False,
            # Add Adam optimizer parameters
            weight_decay=hyperparams['weight_decay'],
            adam_beta1=hyperparams['adam_beta1'],
            adam_beta2=hyperparams['adam_beta2'],
            optim="adamw_torch",
        )
        
        trainer = Trainer(
            model=model,
            args=training_args,
            train_dataset=train_tokens,
            eval_dataset=val_tokens,
            compute_metrics=compute_metrics
        )
        
        print(f"Starting training for hyperparameter configuration {hp_idx+1}/{len(param_grid)}...")
        trainer.train()
        
        # Final evaluation
        print("Running final evaluation...")
        eval_results = trainer.evaluate()
        print(f"Evaluation results: {eval_results}")
        
        # Save configuration details with results
        results_path = os.path.join(output_dir, "results.json")
        with open(results_path, "w") as f:
            json.dump({
                "hyperparameters": hyperparams,
                "results": eval_results
            }, f, indent=2)
        
        # Log to W&B including the hyperparameters
        wandb.log({
            **eval_results,
            **{f"hp_{k}": v for k, v in hyperparams.items()}
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
        
        # Finish this W&B run
        run.finish()
    
    # Log best model information to the project-level run
    wandb.log({
        "best_eval_loss": best_model_info["eval_loss"],
        "best_eval_accuracy": best_model_info["eval_accuracy"],
        **{f"best_hp_{k}": v for k, v in best_model_info["hyperparams"].items()}
    })
    
    # Save the best hyperparameter configuration
    best_config_path = os.path.join(base_output_dir, "best_hyperparams.json")
    with open(best_config_path, "w") as f:
        json.dump(best_model_info, f, indent=2)
    
    print("\n" + "="*60)
    print("Hyperparameter search completed!")
    print(f"Best configuration: {best_model_info['hyperparams']}")
    print(f"Best validation loss: {best_model_info['eval_loss']}")
    print(f"Best validation accuracy: {best_model_info['eval_accuracy']}")
    print(f"Best model saved at: {best_model_info['output_dir']}")
    print("="*60 + "\n")
    
    search_run.finish()

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, default='default.yaml', help='Path to config file')
    parser.add_argument('--small', action='store_true', help='Use small dataset for testing')
    parser.add_argument('--small_grid', action='store_true', help='Use small hyperparameter grid for testing')
    parser.add_argument('--sample_size', type=int, default=-1, help='Number of examples to sample from the dataset. Default is to use 100 if --small, full dataset otherwise')
    args = parser.parse_args()
    main(args) 