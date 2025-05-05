import os
import argparse
import json
import random
import torch
import numpy as np
import evaluate
from datasets import Dataset
from transformers import (
    Trainer,
    TrainingArguments,
    AutoTokenizer,
    AutoModelForCausalLM,
    BitsAndBytesConfig,
)
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
from accelerate import Accelerator
from utils import set_seed, load_config
import wandb

############################################################
# Utility helpers                                          #
############################################################
def load_jsonl(path: str):
    """Read a .jsonl file into a list of dicts"""
    with open(path) as f:
        return [json.loads(line) for line in f]

############################################################
# Model + tokenizer                                        #
############################################################

def get_quantised_lora_model(cfg: dict):
    """Load 4-bit QLoRA model & tokenizer, prepare for k-bit training"""
    base = cfg["model"]["base_model"]
    cache_dir = cfg.get("cache", {}).get("hf_home")

    # Tokenizer
    tokenizer = AutoTokenizer.from_pretrained(base, use_fast=True, cache_dir=cache_dir)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        print("[Tokenizer] pad_token set to eos_token")

    # 4-bit quantisation
    bnb_cfg = BitsAndBytesConfig(
        load_in_4bit=True,
        offload_to_cpu=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16,
    )

    model = AutoModelForCausalLM.from_pretrained(
        base,
        cache_dir=cache_dir,
        device_map="auto",
        quantization_config=bnb_cfg,
    )

    # Prepare and enable gradients
    model = prepare_model_for_kbit_training(model)
    model.enable_input_require_grads()
    model.gradient_checkpointing_enable()

    # LoRA
    lcfg = cfg["model"]["lora"]
    lora_dropout = cfg["model"].get("dropout", 0.1)
    lora_cfg = LoraConfig(
        r=lcfg["r"],
        lora_alpha=lcfg["alpha"],
        lora_dropout=lora_dropout,
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],
        bias="none",
        task_type="CAUSAL_LM",
    )
    model = get_peft_model(model, lora_cfg)
    return tokenizer, model

############################################################
# Main                                                     #
############################################################

def sample_dataset(lst, k, seed=42):
    if len(lst) <= k:
        return lst
    random.seed(seed)
    return random.sample(lst, k)


def main(args):
    accelerator = Accelerator()
    device = accelerator.device
    if accelerator.is_main_process:
        print(f"[Accelerate] device={device} | processes={accelerator.num_processes}")

    cfg = load_config(args.config)
    set_seed(cfg["training"]["seed"])

    # Output directory & W&B
    cfg["training"]["output_dir"] = "../outputs/train3/"
    run_name = (
        f"lora_r{cfg['model']['lora']['r']}_a{cfg['model']['lora']['alpha']}"
        f"_lr{cfg['training']['learning_rate']}_bs{cfg['training']['per_device_train_batch_size']}"
    )
    if args.small:
        run_name = "small_" + run_name
    if accelerator.is_main_process:
        wandb_run = wandb.init(
            project=cfg["wandb"]["project"],
            entity=cfg["wandb"]["entity"],
            config=cfg,
            name=run_name,
        )
        print(f"[W&B] Run: {run_name}")
    else:
        wandb_run = None

    # Load data
    train_list = load_jsonl("../data/formatted/train.jsonl")
    val_list = load_jsonl("../data/formatted/validation.jsonl")
    if args.small:
        train_list = sample_dataset(train_list, 100, cfg["training"]["seed"])
        val_list = sample_dataset(val_list, 20, cfg["training"]["seed"])
        if accelerator.is_main_process:
            print("[Data] SMALL mode: 100 train / 20 val")
    train_ds, val_ds = Dataset.from_list(train_list), Dataset.from_list(val_list)

    # Model & tokenizer
    tokenizer, model = get_quantised_lora_model(cfg)

    # Tokenisation function
    max_len = 2048
    def tok_fn(ex):
        txt = ex["input"] + " " + ex["target"]
        toks = tokenizer(txt, padding="max_length", truncation=True, max_length=max_len, return_tensors="pt")
        toks["labels"] = toks["input_ids"].clone()
        return {k: v.squeeze(0) for k, v in toks.items()}
    train_tok = train_ds.map(tok_fn, batched=False, remove_columns=train_ds.column_names)
    train_tok.set_format(type="torch", columns=["input_ids", "attention_mask", "labels"])
    val_tok = val_ds.map(tok_fn, batched=False, remove_columns=val_ds.column_names)
    val_tok.set_format(type="torch", columns=["input_ids", "attention_mask", "labels"])

    # TrainingArguments
    bf16_flag = torch.cuda.is_bf16_supported()
    targs = TrainingArguments(
        output_dir=cfg["training"]["output_dir"],
        per_device_train_batch_size=cfg["training"]["per_device_train_batch_size"],
        gradient_accumulation_steps=cfg["training"]["gradient_accumulation_steps"],
        num_train_epochs=cfg["training"]["num_train_epochs"],
        learning_rate=float(cfg["training"]["learning_rate"]),
        lr_scheduler_type="constant_with_warmup",
        warmup_ratio=0.03,
        optim="adamw_torch",
        adam_beta1=0.9,
        adam_beta2=0.95,
        weight_decay=0.1,
        logging_steps=25,
        save_steps=100,
        save_total_limit=3,
        per_device_eval_batch_size=1,
        #eval_accumulation_steps=1,
        #eval_strategy="epoch",
        save_strategy="steps",
        #load_best_model_at_end=True,
        #metric_for_best_model="eval_accuracy",
        greater_is_better=True,
        run_name=run_name,
        bf16=bf16_flag,
        fp16=not bf16_flag,
        remove_unused_columns=False,
        report_to="wandb" if accelerator.is_main_process else "none",
        label_names=["labels"],
    )

    # Trainer
    trainer = Trainer(
        model=model,
        args=targs,
        train_dataset=train_tok,
        eval_dataset=val_tok,
        processing_class=tokenizer,
    )

    # Training (with optional resume)
    if accelerator.is_main_process:
        print("[Trainer] Starting training …")
    train_kwargs = {}
    if args.resume_from_checkpoint:
        train_kwargs["resume_from_checkpoint"] = args.resume_from_checkpoint
        if accelerator.is_main_process:
            print(f"[Trainer] Resuming from {args.resume_from_checkpoint}")
    trainer.train(**train_kwargs)

    if accelerator.is_main_process:
        print("Clearing CUDA cache before evaluation…")
        torch.cuda.empty_cache()

    # Evaluation & save
    if accelerator.is_main_process:
        final_path = os.path.join(cfg["training"]["output_dir"], "best_model")
        trainer.save_model(final_path)
        print(f"[Save] Best model -> {final_path}")
        with open(os.path.join(cfg["training"]["output_dir"], "train3_config.yaml"), "w") as f:
            import yaml; yaml.dump(cfg, f)
        wandb_run.finish()


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--config", default="default.yaml")
    p.add_argument("--small", action="store_true")
    p.add_argument("--resume_from_checkpoint", type=str, default=None,
                   help="Path to a checkpoint to resume training from")
    args = p.parse_args()
    main(args)
