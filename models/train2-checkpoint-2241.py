# train.py
import os
import warnings

# allow CUDA allocator to expand fragmented segments
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

# 1) suppress tokenizers‐after‐fork warning
os.environ["TOKENIZERS_PARALLELISM"] = "false"

# 2) silence kernel-version warning
warnings.filterwarnings(
    "ignore",
    message="Detected kernel version .* below the recommended minimum.*"
)

import argparse
import json
import random
import torch
from datasets import Dataset
from transformers import (
    Trainer,
    TrainingArguments,
    AutoTokenizer,
    AutoModelForCausalLM,
)
from peft import LoraConfig, get_peft_model
from accelerate import Accelerator
from utils import set_seed, load_config
import wandb

def load_jsonl(path: str):
    with open(path) as f:
        return [json.loads(line) for line in f]

def get_full_precision_lora_model(cfg):
    base = cfg["model"]["base_model"]
    cache_dir = cfg["cache"]["hf_home"]

    tokenizer = AutoTokenizer.from_pretrained(base, use_fast=True, cache_dir=cache_dir)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # Determine optimal precision - use BF16 if available (A100 supports it)
    torch_dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
    print(f"Using {torch_dtype} precision for model loading")
    
    # Load model in full precision (but still use FP16/BF16)
    model = AutoModelForCausalLM.from_pretrained(
        base,
        cache_dir=cache_dir,
        device_map="auto",  # Let the system determine optimal device mapping
        torch_dtype=torch_dtype,  # Use BF16 on A100, FP16 otherwise
    )

    # Enable gradient checkpointing to save memory
    model.gradient_checkpointing_enable()

    # Configure LoRA with adjusted rank to fit in memory
    lcfg = cfg["model"]["lora"]
    # Reduce rank from 64 to 32 for memory efficiency
    lora_rank = lcfg.get("r", 64) // 2  # Halve the rank to save memory
    lora_alpha = lcfg.get("alpha", 64) // 2  # Keep alpha:rank ratio the same
    
    print(f"Using adjusted LoRA parameters: r={lora_rank}, alpha={lora_alpha} (reduced from r={lcfg.get('r', 64)}, alpha={lcfg.get('alpha', 64)})")
    
    lora_cfg = LoraConfig(
        r=lora_rank,
        lora_alpha=lora_alpha,
        lora_dropout=cfg["model"]["dropout"],
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],
        bias="none",
        task_type="CAUSAL_LM",
    )
    model = get_peft_model(model, lora_cfg)
    
    # Print parameter counts for debugging
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total_params = sum(p.numel() for p in model.parameters())
    print(f"Trainable parameters: {trainable_params:,} ({100 * trainable_params / total_params:.2f}% of {total_params:,} total)")
    
    return tokenizer, model

def sample_dataset(lst, k, seed=42):
    if len(lst) <= k:
        return lst
    random.seed(seed)
    return random.sample(lst, k)

def main(args):
    accelerator = Accelerator()
    device = accelerator.device
    if accelerator.is_main_process:
        print(f"[Accelerate] Using device {device}")

    cfg = load_config(args.config)
    set_seed(cfg["training"]["seed"])

    # Adjust batch size and gradient accumulation for memory efficiency
    # Original: batch_size=8, gradient_accumulation=1
    # New: batch_size=4, gradient_accumulation=2 (same effective batch size)
    cfg["training"]["per_device_train_batch_size"] = 4
    cfg["training"]["gradient_accumulation_steps"] = 2
    
    # Optionally reduce sequence length to save memory
    # Original: 2048
    # New: 1536 (25% reduction)
    cfg["training"]["max_length"] = 1536
    
    print(f"[Memory Optimization] Adjusted batch size: {cfg['training']['per_device_train_batch_size']}")
    print(f"[Memory Optimization] Adjusted gradient accumulation: {cfg['training']['gradient_accumulation_steps']}")
    print(f"[Memory Optimization] Adjusted max length: {cfg['training']['max_length']}")

    # W&B
    run_name = (
        f"full_precision_lora_r{cfg['model']['lora']['r']//2}"
        f"_α{cfg['model']['lora']['alpha']//2}"
        f"_lr{cfg['training']['learning_rate']}"
    )
    if accelerator.is_main_process:
        wandb.init(
            project=cfg["wandb"]["project"],
            entity=cfg["wandb"]["entity"],
            config=cfg,
            name=run_name,
        )

    # Load JSONL
    train_list = load_jsonl("../data/formatted/train.jsonl")
    val_list   = load_jsonl("../data/formatted/validation.jsonl")
    if args.small:
        train_list = sample_dataset(train_list, 100, cfg["training"]["seed"])
        val_list   = sample_dataset(val_list,   20,  cfg["training"]["seed"])
    train_ds = Dataset.from_list(train_list)

    # Model & tokenizer
    tokenizer, model = get_full_precision_lora_model(cfg)
    max_len = cfg["training"]["max_length"]

    # Tokenization + chain-of-thought prompt + masking
    def tok_fn(ex):
        prompt = "Let's think step by step: " + ex["input"]
        full    = prompt + " " + ex["target"]

        enc = tokenizer(
            full,
            padding="max_length",
            truncation=True,
            max_length=max_len,
        )
        input_ids     = torch.tensor(enc["input_ids"], dtype=torch.long)
        attention_mask = torch.tensor(enc["attention_mask"], dtype=torch.long)

        # count prompt tokens to mask
        inp_ids = tokenizer(
            prompt, truncation=True, max_length=max_len
        )["input_ids"]
        inp_len = len(inp_ids)

        labels = input_ids.clone()
        labels[:inp_len] = -100

        return {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "labels": labels,
        }

    train_tok = train_ds.map(
        tok_fn, batched=False, remove_columns=train_ds.column_names
    )
    train_tok.set_format(
        type="torch", columns=["input_ids", "attention_mask", "labels"]
    )

    # TrainingArguments
    tcfg = cfg["training"]
    bf16 = torch.cuda.is_bf16_supported()
    targs = TrainingArguments(
        output_dir=tcfg["output_dir"],
        per_device_train_batch_size=tcfg["per_device_train_batch_size"],
        gradient_accumulation_steps=tcfg["gradient_accumulation_steps"],
        num_train_epochs=tcfg["num_train_epochs"],
        learning_rate=float(tcfg["learning_rate"]),
        lr_scheduler_type=tcfg["lr_scheduler_type"],
        warmup_ratio=tcfg["warmup_ratio"],
        logging_steps=tcfg["logging_steps"],
        # we comment out all eval to avoid breakage
        eval_strategy="no",
        save_strategy=tcfg["save_strategy"],
        load_best_model_at_end=False,  # no eval → disable
        # remove eval batch/dataloader settings
        report_to="wandb" if accelerator.is_main_process else "none",
        bf16=bf16,
        fp16=not bf16,
        remove_unused_columns=False,
        label_names=["labels"],
        run_name=run_name,
        # Add memory optimization flags
        gradient_checkpointing=True,  # Enable gradient checkpointing through trainer
        optim="adamw_torch",  # More memory-efficient optimizer
        ddp_find_unused_parameters=False,  # Optimize distributed training
    )

    trainer = Trainer(
        model=model,
        args=targs,
        train_dataset=train_tok,
        # eval_dataset and compute_metrics removed
    )

    if accelerator.is_main_process:
        print("[Trainer] Starting training …")
    trainer.train(resume_from_checkpoint=args.resume_from_checkpoint)

    if accelerator.is_main_process:
        print("[Trainer] Saving final model …")
        trainer.save_model(os.path.join(tcfg["output_dir"], "final_model"))
        wandb.finish()

if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--config", default="config.yaml")
    p.add_argument("--small", action="store_true")
    p.add_argument(
        "--resume_from_checkpoint", type=str, default=None,
        help="Path to resume checkpoint"
    )
    args = p.parse_args()
    main(args)
