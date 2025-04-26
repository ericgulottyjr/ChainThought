import os
import pathlib
import warnings

# Silence kernel-version warnings
warnings.filterwarnings(
    "ignore",
    message="Detected kernel version .* below the recommended minimum.*"
)
# Disable tokenizer parallelism deadlock warning
os.environ["TOKENIZERS_PARALLELISM"] = "false"

# Define HF cache directory
cwd = pathlib.Path.cwd()
cache_dir = cwd / ".hf_cache"
cache_dir.mkdir(parents=True, exist_ok=True)
abs_cache_path = str(cache_dir.resolve())
os.environ['HF_HOME'] = abs_cache_path
print(f"HF_HOME set to: {os.environ['HF_HOME']}")

# Import libraries after env setup
import torch
# Enable cuDNN auto-tuner for optimal kernels
torch.backends.cudnn.enabled = True
torch.backends.cudnn.benchmark = True

import shutil
import wandb
from datasets import load_dataset, load_from_disk
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    pipeline,
    Trainer,
    TrainingArguments,
    BitsAndBytesConfig,
    default_data_collator,
)
from peft import LoraConfig, get_peft_model, TaskType

torch.cuda.empty_cache()

# Constants
device_map = {"": 0}
MAX_LEN = 256
DATA_CACHE = "gsm8k_cached"
MODEL_NAME = "deepseek-ai/deepseek-llm-7b-base"
OUTPUT_DIR = "deepseek7b-gsm8k-fixed"

# Data loading & preprocessing with improved tokenization
def load_data(split="train"):
    cache_path = f"{DATA_CACHE}_{split}"
    if os.path.isdir(cache_path):
        ds = load_from_disk(cache_path)
    else:
        ds = load_dataset("gsm8k", "main")[split]
        def preprocess(example):
            prompt = f"Question: {example['question']} Answer:"
            prompt_ids = tokenizer(prompt, add_special_tokens=False)["input_ids"]
            answer_ids = tokenizer(example['answer'].strip(), add_special_tokens=False)["input_ids"]
            # truncate to fit
            if len(prompt_ids) + len(answer_ids) > MAX_LEN:
                prompt_ids = prompt_ids[-(MAX_LEN - len(answer_ids)):]
            if len(prompt_ids) + len(answer_ids) > MAX_LEN:
                answer_ids = answer_ids[:(MAX_LEN - len(prompt_ids))]
            input_ids = prompt_ids + answer_ids
            attention_mask = [1] * len(input_ids)
            pad_len = MAX_LEN - len(input_ids)
            if tokenizer.padding_side == 'left':
                input_ids = [tokenizer.pad_token_id] * pad_len + input_ids
                attention_mask = [0] * pad_len + attention_mask
            else:
                input_ids = input_ids + [tokenizer.pad_token_id] * pad_len
                attention_mask = attention_mask + [0] * pad_len
            labels = [-100] * len(prompt_ids) + answer_ids
            if tokenizer.padding_side == 'left':
                labels = [-100] * pad_len + labels
            else:
                labels = labels + [-100] * pad_len
            if sum(1 for l in labels if l != -100) == 0:
                raise ValueError("All labels are masked (-100). Check preprocessing logic.")
            return {"input_ids": input_ids, "attention_mask": attention_mask, "labels": labels}
        ds = ds.map(preprocess, remove_columns=ds.column_names)
        ds.save_to_disk(cache_path)
    return ds

# Prepare tokenizer
tokenizer = AutoTokenizer.from_pretrained(
    MODEL_NAME,
    padding_side="left",
    trust_remote_code=True
)

# Model initializer with LoRA and quantization
def model_init():
    quant_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_compute_dtype=torch.float16,
        llm_int8_enable_fp32_cpu_offload=True
    )
    base = AutoModelForCausalLM.from_pretrained(
        MODEL_NAME,
        quantization_config=quant_config,
        device_map=device_map,
        torch_dtype=torch.bfloat16,
    )
    lora_cfg = LoraConfig(
        task_type=TaskType.CAUSAL_LM,
        inference_mode=False,
        r=8,
        lora_alpha=32,
        lora_dropout=0.05,
    )
    model = get_peft_model(base, lora_cfg)
    model.config.use_cache = False
    model.gradient_checkpointing_enable()
    model.enable_input_require_grads()
    return model

if __name__ == "__main__":
    fixed_params = {
        "learning_rate": 5e-6,
        "per_device_train_batch_size": 2,
        "gradient_accumulation_steps": 4,
        "weight_decay": 0.01,
        "num_train_epochs": 3,
    }

    # Load data
    train_ds = load_data("train")
    eval_ds = load_data("test")

    # Initialize W&B
    wandb.init(
        project="deepseek-gsm8k-final",
        name="fixed_run",
        config=fixed_params
    )

    # Build model
    model = model_init()

    # Training arguments
    args = TrainingArguments(
        output_dir=OUTPUT_DIR,
        per_device_train_batch_size=fixed_params["per_device_train_batch_size"],
        gradient_accumulation_steps=fixed_params["gradient_accumulation_steps"],
        learning_rate=fixed_params["learning_rate"],
        weight_decay=fixed_params["weight_decay"],
        num_train_epochs=fixed_params["num_train_epochs"],
        fp16=True,
        max_grad_norm=1.0,
        logging_steps=100,
        eval_strategy="steps",
        eval_steps=500,
        save_steps=500,
        save_total_limit=2,
        load_best_model_at_end=True,
        metric_for_best_model="eval_loss",
        label_names=["labels"],
        report_to=["wandb"],
        run_name="fixed_run",
        dataloader_num_workers=4,
        dataloader_pin_memory=True,
        dataloader_prefetch_factor=2,
    )

    # Trainer setup
    trainer = Trainer(
        model=model,
        args=args,
        train_dataset=train_ds,
        eval_dataset=eval_ds,
        data_collator=default_data_collator,
        processing_class=tokenizer,
    )

    # Train & save
    trainer.train()
    trainer.save_model(f"{OUTPUT_DIR}-best")

    # Side-by-side generative samples
    base_gen = pipeline("text-generation", model=MODEL_NAME, tokenizer=tokenizer)
    ft_gen = pipeline("text-generation", model=f"{OUTPUT_DIR}-best", tokenizer=tokenizer)
    samples = eval_ds.select(range(10))
    for ex in samples:
        q = f"Question: {ex['question']} Answer:"
        print("→ GOLD:        ", ex['answer'])
        print("→ BASE OUTPUT: ", base_gen(q, max_new_tokens=128)[0]['generated_text'])
        print("→ FT OUTPUT:   ", ft_gen(q, max_new_tokens=128)[0]['generated_text'])
        print("-" * 60)

    # Future: implement chain-of-thought prompting
    # e.g., prompt = q + " Let's think step by step:"
    # and compare reasoning traces between base_gen and ft_gen

    wandb.finish()
