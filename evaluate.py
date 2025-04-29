import os
import argparse
import json
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import PeftModel
import torch

# Ensure HF cache is set (redundant if set globally, but safe)
# Consider loading from a central config like in train.py if needed
if 'HF_HOME' not in os.environ:
    print("Setting default HF_HOME cache...")
    os.environ['HF_HOME'] = 'data/.hf_home'
if 'HF_DATASETS_CACHE' not in os.environ:
     os.environ['HF_DATASETS_CACHE'] = 'data/.hf_home/datasets'

# --- Model Loading Functions ---

def load_model_and_tokenizer(model_id_or_path: str, model_type: str, adapter_path: str = None):
    print(f"Loading model: {model_id_or_path}")
    device = "cuda" if torch.cuda.is_available() else "cpu"
    cache_dir = os.environ.get('HF_HOME')

    tokenizer = AutoTokenizer.from_pretrained(model_id_or_path, cache_dir=cache_dir)
    # Set pad token if missing (common for Llama-based models)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        print("Set tokenizer pad_token to eos_token")

    if model_type == "base":
        print("Loading base model...")
        model = AutoModelForCausalLM.from_pretrained(
            model_id_or_path,
            cache_dir=cache_dir,
            # torch_dtype=torch.float16, # Uncomment if needed for memory
            # load_in_8bit=True,       # Uncomment if needed for memory
        )
    elif model_type == "tuned":
        if not adapter_path:
            raise ValueError("adapter_path must be provided for model_type='tuned'")
        print(f"Loading base model {model_id_or_path} for adapter...")
        base_model = AutoModelForCausalLM.from_pretrained(
            model_id_or_path,
            cache_dir=cache_dir,
            # torch_dtype=torch.float16,
            # load_in_8bit=True,
        )
        print(f"Loading adapter from {adapter_path}...")
        model = PeftModel.from_pretrained(base_model, adapter_path, cache_dir=cache_dir)
        print("Adapter loaded.")
    else:
        raise ValueError(f"Unknown model_type: {model_type}. Choose 'base' or 'tuned'.")

    model.eval() # Set to evaluation mode
    model = model.to(device)
    print(f"Model loaded successfully on {device}.")
    return tokenizer, model

# --- Data Loading ---

def load_jsonl(path: str):
    print(f"Loading data from: {path}")
    try:
        with open(path) as f:
            data = [json.loads(line) for line in f]
            print(f"Loaded {len(data)} examples.")
            return data
    except FileNotFoundError:
        print(f"ERROR: Data file not found at {path}")
        return []

# --- Main Evaluation Loop ---

def main(args):
    # Load the specified model and tokenizer
    tokenizer, model = load_model_and_tokenizer(
        args.base_model_id,
        args.model_type,
        args.adapter_path
    )

    # Load the evaluation data
    data = load_jsonl(args.test_file)
    if not data:
        return # Exit if data loading failed
        
    # Limit examples if needed for quick testing
    if args.limit > 0:
        print(f"Limiting evaluation to first {args.limit} examples.")
        data = data[:args.limit]

    results = []
    print("\n--- Starting Generation ---")
    for i, ex in enumerate(data):
        prompt = ex['input'] # Assuming the input field contains the full prompt
        target = ex.get('target', 'N/A') # Get target if available
        
        inputs = tokenizer(prompt, return_tensors='pt').to(model.device)
        
        print(f"\n--- Example {i+1} ---")
        print(f"Prompt:\n{prompt}")
        print(f"\nTarget:\n{target}")
        
        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_new_tokens=args.max_new_tokens,
                pad_token_id=tokenizer.pad_token_id, # Use pad token for stopping
                eos_token_id=tokenizer.eos_token_id,
                # Add generation config params if desired
                # do_sample=True,
                # temperature=0.7,
                # top_p=0.9,
            )
        
        # Decode generated sequence, skipping special tokens and the prompt part
        # Add 1 to input length to avoid potential space issues if prompt ends with space
        generated_ids = outputs[0][inputs.input_ids.shape[1]:] 
        prediction = tokenizer.decode(generated_ids, skip_special_tokens=True)
        
        print(f"\nGenerated Prediction:\n{prediction}")
        results.append({
            "index": i,
            "prompt": prompt,
            "target": target,
            "prediction": prediction
        })

    # Save results to a file
    output_filename = f"outputs/eval_{args.model_type}.jsonl"
    if args.adapter_path:
         # Add adapter name to filename if provided
         adapter_name = os.path.basename(os.path.normpath(args.adapter_path))
         output_filename = f"outputs/eval_{adapter_name}.jsonl"
         
    os.makedirs("outputs", exist_ok=True)
    print(f"\n--- Saving results to {output_filename} ---")
    with open(output_filename, 'w') as f:
        for result in results:
            f.write(json.dumps(result) + '\n')
    print("Evaluation complete.")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Evaluate base vs tuned LLMs on a dataset.")
    parser.add_argument('--base_model_id', type=str, default="deepseek-ai/deepseek-llm-7b-base", help="Base model ID from Hugging Face Hub.")
    parser.add_argument('--model_type', type=str, required=True, choices=['base', 'tuned'], help="Specify 'base' or 'tuned'.")
    parser.add_argument('--adapter_path', type=str, default=None, help="Path to the trained LoRA adapter directory (required if model_type='tuned').")
    parser.add_argument('--test_file', type=str, default='data/formatted/test.jsonl', help="Path to the input test file (.jsonl format).")
    parser.add_argument('--max_new_tokens', type=int, default=256, help="Max new tokens to generate.")
    parser.add_argument('--limit', type=int, default=100, help="Limit the number of examples to evaluate (default: 100).")
    # Removed --output_path, filename is now generated automatically
    
    args = parser.parse_args()
    
    if args.model_type == 'tuned' and not args.adapter_path:
        parser.error("--adapter_path is required when --model_type is 'tuned'")
        
    main(args)