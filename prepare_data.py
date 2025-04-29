import os
# Ensure all HF caching stays under project directory
os.environ['HF_HOME'] = 'data/.hf_home'
os.environ['HF_DATASETS_CACHE'] = 'data/.hf_home/datasets'

import json
from datasets import load_dataset

RAW_CACHE_DIR = 'data/raw'
FORMATTED_DIR = 'data/formatted'


def format_example(example):
    question = example['question']
    answer = example['answer']
    return {
        'input': f"Problem: {question}\nSteps:",
        'target': f"Answer: {answer}"
    }


def main(raw_dir: str = RAW_CACHE_DIR, out_dir: str = FORMATTED_DIR, seed: int = 42):
    os.makedirs(raw_dir, exist_ok=True)
    os.makedirs(out_dir, exist_ok=True)

    # Load and cache GSM8K locally
    ds = load_dataset('gsm8k', 'main', cache_dir=raw_dir)

    # Format examples
    formatted = ds['train'].map(format_example, remove_columns=ds['train'].column_names)

    # Split into train/validation/test
    train_val = formatted.train_test_split(test_size=0.1, seed=seed)
    test = train_val['test']
    train_val = train_val['train'].train_test_split(test_size=1/9, seed=seed)
    train, val = train_val['train'], train_val['test']

    # Write to JSONL
    for split_name, dataset in [('train', train), ('validation', val), ('test', test)]:
        path = os.path.join(out_dir, f"{split_name}.jsonl")
        with open(path, 'w') as f:
            for ex in dataset:
                f.write(json.dumps(ex) + '\n')
    print(f"Datasets saved in {out_dir}")


if __name__ == '__main__':
    import argparse
    p = argparse.ArgumentParser(description="Prepare and cache GSM8K dataset.")
    p.add_argument('--raw_dir', type=str, default=RAW_CACHE_DIR)
    p.add_argument('--out_dir', type=str, default=FORMATTED_DIR)
    p.add_argument('--seed', type=int, default=42)
    args = p.parse_args()
    main(args.raw_dir, args.out_dir, args.seed)