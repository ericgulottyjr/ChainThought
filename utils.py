import os
import random
import numpy as np
import torch
import yaml
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import LoraConfig, get_peft_model


def load_config(path: str) -> dict:
    # Load YAML and set cache envs if provided
    with open(path) as f:
        cfg = yaml.safe_load(f)
    cache_cfg = cfg.get('cache', {})
    if cache_cfg:
        os.environ['HF_HOME'] = cache_cfg['hf_home']
        os.environ['HF_DATASETS_CACHE'] = cache_cfg['datasets_cache']
    return cfg


def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def get_tokenizer_and_model(cfg: dict):
    base = cfg['model']['base_model']
    cache_dir = cfg.get('cache', {}).get('hf_home', None)

    tokenizer = AutoTokenizer.from_pretrained(base, use_fast=True, cache_dir=cache_dir)
    model = AutoModelForCausalLM.from_pretrained(base, cache_dir=cache_dir)

    # Apply LoRA
    lora_cfg = LoraConfig(
        r=cfg['model']['lora']['r'],
        lora_alpha=cfg['model']['lora']['alpha'],
        target_modules=['q_proj', 'v_proj'],
        lora_dropout=0.05,
        bias='none'
    )
    model = get_peft_model(model, lora_cfg)
    return tokenizer, model