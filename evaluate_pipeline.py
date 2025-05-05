#!/usr/bin/env python3
import os
import argparse
import json
import re
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import PeftModel
import torch
from tqdm import tqdm


def slice_until_hashes(text: str) -> str:
    """Keep everything up to and including the first line that starts with '####'."""
    match = re.search(r'^####.*$', text, flags=re.MULTILINE)
    if match:
        return text[: match.end()].strip()
    return text.strip()


def extract_answer(text: str) -> str:
    """Extract a numeric answer using regex patterns, or the last number fallback."""
    patterns = [
        r"(?:answer|result)(?:\s+is)?(?:\s*:)?\s*(-?\d+(?:\.\d+)?)",
        r"(?:final answer|final result)(?:\s+is)?(?:\s*:)?\s*(-?\d+(?:\.\d+)?)",
        r"(?:the answer|the result)(?:\s+is)?(?:\s*:)?\s*(-?\d+(?:\.\d+)?)",
    ]
    for pat in patterns:
        matches = re.findall(pat, text, flags=re.IGNORECASE)
        if matches:
            return matches[-1].strip()
    nums = re.findall(r"(-?\d+(?:\.\d+)?)", text)
    return nums[-1].strip() if nums else ""


def evaluate_model(model, tokenizer, data, max_new_tokens, device, label):
    """
    Generate responses and compute multiple evaluation metrics:
      - exact-match accuracy (on numeric answer)
      - per-token accuracy (positional match)
      - token-level precision & recall (micro-average)
      - token-level F1 (micro-average)
    Displays running metrics and returns a dict.
    """
    exact_count = 0
    token_match = 0
    total_target_tokens = 0
    sum_pred_tokens = 0
    sum_target_tokens = 0
    sum_intersection = 0
    total_examples = len(data)

    pbar = tqdm(data, desc=f"Examples for {label}", unit="ex")
    for i, ex in enumerate(pbar, start=1):
        inputs = tokenizer(ex['input'], return_tensors='pt').to(device)
        with torch.no_grad():
            out = model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                pad_token_id=tokenizer.pad_token_id,
                eos_token_id=tokenizer.eos_token_id
            )
        gen = tokenizer.decode(
            out[0][inputs.input_ids.shape[1]:],
            skip_special_tokens=True
        )
        pred_slice = slice_until_hashes(gen)
        tgt_slice = slice_until_hashes(ex['target'])

        # Exact-match on answer
        ap = extract_answer(pred_slice)
        at = extract_answer(tgt_slice)
        if ap and at and ap == at:
            exact_count += 1

        # Per-token positional accuracy
        pred_tokens = pred_slice.split()
        tgt_tokens = tgt_slice.split()
        match_count = sum(1 for idx in range(min(len(pred_tokens), len(tgt_tokens)))
                          if pred_tokens[idx] == tgt_tokens[idx])
        token_match += match_count
        total_target_tokens += len(tgt_tokens)

        # Token-set precision/recall
        p_set = set(pred_tokens)
        t_set = set(tgt_tokens)
        inter = len(p_set & t_set)
        sum_intersection += inter
        sum_pred_tokens += len(p_set)
        sum_target_tokens += len(t_set)

        # Compute running micro-metrics
        acc_run = exact_count / i
        per_tok_run = token_match / total_target_tokens if total_target_tokens else 0
        prec_run = sum_intersection / sum_pred_tokens if sum_pred_tokens else 0
        rec_run = sum_intersection / sum_target_tokens if sum_target_tokens else 0
        f1_run = (2 * prec_run * rec_run / (prec_run + rec_run)) if (prec_run + rec_run) > 0 else 0

        pbar.set_postfix({
            'acc': f"{acc_run:.4f}",
            'pt_acc': f"{per_tok_run:.4f}",
            'prec': f"{prec_run:.4f}",
            'rec': f"{rec_run:.4f}",
            'f1': f"{f1_run:.4f}"
        })

    # Final micro-averaged metrics
    accuracy = exact_count / total_examples if total_examples else 0.0
    per_token_accuracy = token_match / total_target_tokens if total_target_tokens else 0.0
    precision = sum_intersection / sum_pred_tokens if sum_pred_tokens else 0.0
    recall = sum_intersection / sum_target_tokens if sum_target_tokens else 0.0
    f1 = (2 * precision * recall / (precision + recall)) if (precision + recall) > 0 else 0.0

    print(f"{label} -> accuracy: {accuracy:.4f}, per_tok_acc: {per_token_accuracy:.4f},"
          f" precision: {precision:.4f}, recall: {recall:.4f}, f1: {f1:.4f}")

    return {
        "accuracy": accuracy,
        "per_token_accuracy": per_token_accuracy,
        "precision": precision,
        "recall": recall,
        "f1": f1
    }


def load_jsonl(path: str):
    with open(path, 'r') as f:
        return [json.loads(line) for line in f]


def main():
    parser = argparse.ArgumentParser(
        description="Evaluate multiple checkpoints on GSM8K with detailed metrics"
    )
    parser.add_argument(
        '--base_model_id', type=str, default="deepseek-ai/deepseek-llm-7b-base",
        help="Base model ID from Hugging Face Hub."
    )
    parser.add_argument(
        "--best_checkpoints_dir", type=str, default="outputs/best_checkpoints",
        help="Directory containing adapter/model checkpoint subfolders"
    )
    parser.add_argument(
        "--test_file", type=str,
        default='data/formatted/test.jsonl', help="Path to the input test file (.jsonl format)."
    )
    parser.add_argument(
        "--max_new_tokens", type=int, default=256,
        help="Max tokens to generate"
    )
    parser.add_argument(
        "--limit", type=int, default=100,
        help="Limit number of examples (0 = no limit)"
    )
    parser.add_argument(
        "--cache_dir", type=str, default='data/.hf_home',
        help="Optional HF cache directory"
    )
    parser.add_argument(
        "--output_summary", type=str, default="outputs/eval_summary.json",
        help="Path to write aggregated metrics JSON"
    )
    parser.add_argument(
        "--run_baseline", action='store_true',
        help="Also evaluate the base model without any adapters"
    )
    args = parser.parse_args()

    # Setup HF cache
    if args.cache_dir:
        os.environ['HF_HOME'] = args.cache_dir
        os.environ['HF_DATASETS_CACHE'] = os.path.join(args.cache_dir, 'datasets')

    device = "cuda" if torch.cuda.is_available() else "cpu"

    # Load and optionally limit test data
    data = load_jsonl(args.test_file)
    if args.limit > 0:
        data = data[: args.limit]

    os.makedirs(os.path.dirname(args.output_summary), exist_ok=True)
    summary = {}

    # Baseline evaluation
    if args.run_baseline:
        print("\n=== Evaluating baseline (no adapter) ===")
        tok = AutoTokenizer.from_pretrained(
            args.base_model_id,
            cache_dir=os.environ.get('HF_HOME')
        )
        if tok.pad_token is None:
            tok.pad_token = tok.eos_token
        base_model = AutoModelForCausalLM.from_pretrained(
            args.base_model_id,
            cache_dir=os.environ.get('HF_HOME')
        ).eval().to(device)
        summary['baseline'] = evaluate_model(
            base_model, tok, data,
            args.max_new_tokens, device, 'baseline'
        )

    # Adapter checkpoints
    for ckpt in sorted(os.listdir(args.best_checkpoints_dir)):
        ckpt_path = os.path.join(args.best_checkpoints_dir, ckpt)
        if not os.path.isdir(ckpt_path):
            continue
        print(f"\n=== Evaluating {ckpt} ===")

        tok = AutoTokenizer.from_pretrained(
            args.base_model_id,
            cache_dir=os.environ.get('HF_HOME')
        )
        if tok.pad_token is None:
            tok.pad_token = tok.eos_token
        base_mod = AutoModelForCausalLM.from_pretrained(
            args.base_model_id,
            cache_dir=os.environ.get('HF_HOME')
        )
        adapter_mod = PeftModel.from_pretrained(
            base_mod, ckpt_path,
            cache_dir=os.environ.get('HF_HOME')
        ).eval().to(device)
        summary[ckpt] = evaluate_model(
            adapter_mod, tok, data,
            args.max_new_tokens, device, ckpt
        )
        del adapter_mod, base_mod
        torch.cuda.empty_cache()

    # Write summary JSON
    with open(args.output_summary, 'w') as outf:
        json.dump(summary, outf, indent=2)
    print(f"\nAll results saved to {args.output_summary}")


if __name__ == "__main__":
    main()
