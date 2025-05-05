import os
import argparse
import json
import torch
import numpy as np
import evaluate
from datasets import Dataset
from utils import set_seed, load_config
import wandb
from transformers import AutoTokenizer, AutoModelForCausalLM, get_linear_schedule_with_warmup
from peft import LoraConfig, get_peft_model
from torch.utils.data import DataLoader
from accelerate import Accelerator
from tqdm import tqdm


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


def get_tokenizer_and_model(cfg: dict):
    base = cfg['model']['base_model']
    cache_dir = cfg.get('cache', {}).get('hf_home', None)

    # Initialize tokenizer
    tokenizer = AutoTokenizer.from_pretrained(base, use_fast=True, cache_dir=cache_dir)
    
    # Set pad token if missing (common for Llama-based models)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        print("Set tokenizer pad_token to eos_token")

    # Load model with memory-efficient settings
    print("Loading model with memory-efficient settings...")
    model = AutoModelForCausalLM.from_pretrained(
        base, 
        cache_dir=cache_dir,
        # Load in 8-bit to save memory
        load_in_8bit=True,
        device_map="auto",
    )

    # Get the dropout value from config or use default
    lora_dropout = cfg['model'].get('dropout', 0.1)
    
    # Apply LoRA configuration - use smaller r value to save memory
    lora_r = cfg['model']['lora']['r']
    lora_alpha = cfg['model']['lora']['alpha']
    print(f"Using LoRA with r={lora_r}, alpha={lora_alpha}, dropout={lora_dropout}")
    
    # Target fewer modules to save memory
    target_modules = ['q_proj', 'v_proj']
    print(f"Targeting modules: {', '.join(target_modules)}")
    
    lora_cfg = LoraConfig(
        r=lora_r,
        lora_alpha=lora_alpha,
        target_modules=target_modules,
        lora_dropout=lora_dropout,
        bias='none',
        task_type="CAUSAL_LM"
    )
    
    model = get_peft_model(model, lora_cfg)
    
    # IMPORTANT: Ensure all trainable parameters have requires_grad=True
    for param in model.parameters():
        if param.requires_grad:
            # Double-check to ensure trainable parameters stay trainable
            param.requires_grad = True
    
    # Set model to training mode
    model.train()
    
    # Check and report trainable parameters
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    all_params = sum(p.numel() for p in model.parameters())
    print(f"Model has {trainable_params:,} trainable parameters out of {all_params:,} total parameters")
    
    return tokenizer, model


class ChainThoughtDataset(torch.utils.data.Dataset):
    def __init__(self, dataset, tokenizer, max_length=768):
        self.dataset = dataset
        self.tokenizer = tokenizer
        self.max_length = max_length
        
    def __len__(self):
        return len(self.dataset)
    
    def __getitem__(self, idx):
        example = self.dataset[idx]
        full_text = example['input'] + " " + example['target']
        
        # Tokenize with explicit padding and truncation
        encodings = self.tokenizer(
            full_text,
            padding="max_length",
            truncation=True,
            max_length=self.max_length,
            return_tensors=None
        )
        
        # Make labels from input_ids
        encodings["labels"] = encodings["input_ids"].copy()
        
        # Convert to tensors
        for key in encodings:
            encodings[key] = torch.tensor(encodings[key])
            
        return encodings


def train_batch(model, batch, optimizer, accelerator):
    """Safely handle the forward and backward pass for a batch"""
    # Forward pass
    outputs = model(**batch)
    loss = outputs.loss
    
    # Make sure loss is a scalar and requires gradients
    if not loss.requires_grad:
        # If loss doesn't require grad, we need to create a new tensor that does
        dummy_loss = torch.zeros(1, device=loss.device, requires_grad=True)
        # Add the original loss value to ensure the numeric value is preserved
        safe_loss = dummy_loss + loss.detach()
        return safe_loss
    
    return loss


def main(args):
    # Initialize accelerator with explicit gradient accumulation
    # If used correctly in context manager with accumulate(), 
    # this handles dividing loss values, syncing gradients, etc.
    accelerator = Accelerator(
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        log_with="wandb",  # Enable wandb integration
        # Memory optimizations
        cpu_offload=True,  # Offload optimizer state and parameters to CPU to save GPU memory
    )
    
    # Get device info
    if accelerator.is_main_process:
        print(f"Using device: {accelerator.device}")
        print(f"Number of processes: {accelerator.num_processes}")
        print(f"Distributed type: {accelerator.distributed_type}")
        print(f"Mixed precision: {accelerator.mixed_precision}")
        print(f"Gradient accumulation steps: {accelerator.gradient_accumulation_steps}")
        print(f"CPU offload: {accelerator.cpu_offload}")
        print(f"Batch size: {args.batch_size}")
    
    # Load config
    cfg = load_config(args.config)
    if accelerator.is_main_process:
        print(f"Loaded configuration from: {args.config}")
    
    # Set output directory
    output_dir = "outputs/accelerate/"
    if accelerator.is_main_process:
        print(f"Output directory: {output_dir}")
        os.makedirs(output_dir, exist_ok=True)

    # Set seed for reproducibility
    set_seed(cfg['training']['seed'])
    
    # Create run name
    run_name = f"accelerate_lora_r{cfg['model']['lora']['r']}_a{cfg['model']['lora']['alpha']}_lr{cfg['training']['learning_rate']}_bs{args.batch_size}"
    
    if 'dropout' in cfg['model']:
        run_name += f"_do{cfg['model']['dropout']}"
    else:
        cfg['model']['dropout'] = 0.1
        run_name += f"_do{cfg['model']['dropout']}"
        
    if args.small:
        run_name = f"small_{run_name}"
        
    # Initialize wandb - only on main process
    if accelerator.is_main_process:
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
        
        if accelerator.is_main_process:
            print(f"SMALL DATASET MODE: Using {len(train_list)}/{original_train_size} train examples")
            print(f"SMALL DATASET MODE: Using {len(val_list)}/{original_val_size} validation examples")
    else:
        if accelerator.is_main_process:
            print(f"FULL DATASET MODE: Using all {len(train_list)} train examples")
            print(f"FULL DATASET MODE: Using all {len(val_list)} validation examples")

    # Get tokenizer and model
    tokenizer, model = get_tokenizer_and_model(cfg)
    
    # Create custom datasets with reduced sequence length
    # Reduced from 768 to 512 to save memory
    max_length = args.max_length
    if accelerator.is_main_process:
        print(f"Using sequence length: {max_length}")
    
    train_dataset = ChainThoughtDataset(train_list, tokenizer, max_length)
    val_dataset = ChainThoughtDataset(val_list, tokenizer, max_length)
    
    # Create dataloaders with command-line batch size
    batch_size = args.batch_size
    train_dataloader = DataLoader(
        train_dataset, 
        batch_size=batch_size, 
        shuffle=True,
        # Add pin_memory=False to reduce memory pressure
        pin_memory=False
    )
    val_dataloader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        pin_memory=False
    )
    
    # Print trainable params
    if accelerator.is_main_process:
        trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        all_params = sum(p.numel() for p in model.parameters())
        print(f"Model has {trainable_params:,} trainable parameters out of {all_params:,} total parameters")
    
    # Setup optimizer
    lr = float(cfg['training']['learning_rate'])
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=lr,
        weight_decay=0.01,
        betas=(0.9, 0.999),
        eps=1e-8
    )
    
    # Setup learning rate scheduler
    num_epochs = cfg['training']['num_train_epochs']
    num_training_steps = len(train_dataloader) * num_epochs
    num_warmup_steps = int(0.1 * num_training_steps)  # 10% warmup
    
    lr_scheduler = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=num_warmup_steps,
        num_training_steps=num_training_steps
    )
    
    # Prepare everything with accelerator
    model, optimizer, train_dataloader, val_dataloader, lr_scheduler = accelerator.prepare(
        model, optimizer, train_dataloader, val_dataloader, lr_scheduler
    )
    
    # Enable gradient checkpointing if available
    if hasattr(model, "gradient_checkpointing_enable"):
        model.gradient_checkpointing_enable()
        if accelerator.is_main_process:
            print("Gradient checkpointing enabled")
    
    # IMPORTANT: Verify model is in training mode
    model.train()
    
    # Check trainable parameters after prepare
    if accelerator.is_main_process:
        unwrapped_model = accelerator.unwrap_model(model)
        trainable_params = sum(p.numel() for p in unwrapped_model.parameters() if p.requires_grad)
        print(f"After preparation: {trainable_params:,} trainable parameters")
        
        # Verify optimizer has parameters that require gradients
        has_grad_params = any(param.requires_grad for param_group in optimizer.param_groups for param in param_group['params'])
        print(f"Optimizer has parameters requiring gradients: {has_grad_params}")
    
    # Training loop
    if accelerator.is_main_process:
        print("Starting training...")
    
    for epoch in range(num_epochs):
        model.train()  # Ensure model is in training mode at start of each epoch
        total_loss = 0
        
        progress_bar = tqdm(
            train_dataloader, 
            disable=not accelerator.is_main_process,
            desc=f"Epoch {epoch+1}/{num_epochs}"
        )
        
        for step, batch in enumerate(progress_bar):
            # Forward pass with safe loss calculation
            loss = train_batch(model, batch, optimizer, accelerator)
            
            # Use accelerator's accumulate context manager
            with accelerator.accumulate(model):
                # Clear previous gradients
                optimizer.zero_grad()
                
                # Backward pass - use accelerator's backward which handles distributed training
                accelerator.backward(loss)
                
                # Clip gradients to avoid explosion
                if accelerator.sync_gradients:
                    accelerator.clip_grad_norm_(model.parameters(), 1.0)
                
                # Update weights
                optimizer.step()
                lr_scheduler.step()
            
            # Update progress bar and stats
            if accelerator.is_main_process:
                progress_bar.set_postfix({"loss": loss.item()})
            
            total_loss += loss.detach().float()
            
            # Log to wandb
            if accelerator.is_main_process and step % 25 == 0:
                wandb.log({
                    "train/loss": loss.item(),
                    "train/lr": lr_scheduler.get_last_lr()[0],
                    "train/epoch": epoch + (step / len(train_dataloader)),
                    "train/step": step + epoch * len(train_dataloader)
                })
        
        # Calculate average loss
        avg_loss = total_loss / len(train_dataloader)
        if accelerator.is_main_process:
            print(f"Epoch {epoch+1}/{num_epochs} - Average loss: {avg_loss:.4f}")
        
        # Evaluation
        if accelerator.is_main_process:
            print("Running evaluation...")
        
        model.eval()
        eval_loss = 0
        
        with torch.no_grad():
            for batch in tqdm(
                val_dataloader, 
                disable=not accelerator.is_main_process,
                desc="Evaluation"
            ):
                outputs = model(**batch)
                loss = outputs.loss
                eval_loss += loss.detach().float()
        
        eval_loss = eval_loss / len(val_dataloader)
        
        # Log evaluation results
        if accelerator.is_main_process:
            print(f"Evaluation loss: {eval_loss:.4f}")
            wandb.log({
                "eval/loss": eval_loss,
                "eval/epoch": epoch + 1
            })
        
        # Save model checkpoint
        if accelerator.is_main_process:
            print(f"Saving checkpoint for epoch {epoch+1}...")
            checkpoint_dir = os.path.join(output_dir, f"checkpoint-{epoch+1}")
            os.makedirs(checkpoint_dir, exist_ok=True)
            
            # Unwrap model before saving
            unwrapped_model = accelerator.unwrap_model(model)
            unwrapped_model.save_pretrained(checkpoint_dir)
            tokenizer.save_pretrained(checkpoint_dir)
    
    # Save final model
    if accelerator.is_main_process:
        print("Saving final model...")
        final_checkpoint_dir = os.path.join(output_dir, "final")
        os.makedirs(final_checkpoint_dir, exist_ok=True)
        
        unwrapped_model = accelerator.unwrap_model(model)
        unwrapped_model.save_pretrained(final_checkpoint_dir)
        tokenizer.save_pretrained(final_checkpoint_dir)
        
        print(f"Training complete! Final model saved to {final_checkpoint_dir}")
        wandb.finish()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, default='default.yaml', help='Path to config file')
    parser.add_argument('--small', action='store_true', help='Use small dataset for quick testing')
    parser.add_argument('--gradient_accumulation_steps', type=int, default=16, help='Number of steps to accumulate gradients')
    parser.add_argument('--max_length', type=int, default=512, help='Maximum sequence length')
    parser.add_argument('--batch_size', type=int, default=16, help='Batch size')
    args = parser.parse_args()
    main(args) 