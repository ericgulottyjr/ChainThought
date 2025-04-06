#!/usr/bin/env python
# -*- coding: utf-8 -*-

import os
import shutil
import argparse
from pathlib import Path

def parse_args():
    parser = argparse.ArgumentParser(description="Check or clean the Hugging Face cache")
    parser.add_argument(
        "--clean",
        action="store_true",
        help="Clean the Hugging Face cache",
    )
    parser.add_argument(
        "--cache_dir",
        type=str,
        default="./.hf_cache",
        help="Path to the Hugging Face cache directory",
    )
    return parser.parse_args()

def check_cache_size(cache_dir):
    """Check the size of the cache directory."""
    if not os.path.exists(cache_dir):
        print(f"Cache directory {cache_dir} does not exist")
        return 0
    
    total_size = 0
    for dirpath, dirnames, filenames in os.walk(cache_dir):
        for f in filenames:
            fp = os.path.join(dirpath, f)
            total_size += os.path.getsize(fp)
    
    # Convert to GB
    total_size_gb = total_size / (1024 ** 3)
    return total_size_gb

def clean_cache(cache_dir):
    """Clean the Hugging Face cache directory."""
    if not os.path.exists(cache_dir):
        print(f"Cache directory {cache_dir} does not exist")
        return
    
    # Check if there are any directories inside the cache dir
    cache_path = Path(cache_dir)
    
    # Clean transformers cache
    transformers_cache = cache_path / "transformers"
    if transformers_cache.exists():
        print(f"Cleaning transformers cache at {transformers_cache}")
        shutil.rmtree(transformers_cache)
        os.makedirs(transformers_cache, exist_ok=True)
    
    # Clean datasets cache
    datasets_cache = cache_path / "datasets"
    if datasets_cache.exists():
        print(f"Cleaning datasets cache at {datasets_cache}")
        shutil.rmtree(datasets_cache)
        os.makedirs(datasets_cache, exist_ok=True)
    
    print(f"Cache cleaned successfully")

def main():
    args = parse_args()
    
    # Set cache dir to absolute path
    cache_dir = os.path.abspath(args.cache_dir)
    
    # Create cache directory if it doesn't exist
    os.makedirs(cache_dir, exist_ok=True)
    
    # Set environment variables to use the specified cache directory
    os.environ["HF_HOME"] = cache_dir
    os.environ["TRANSFORMERS_CACHE"] = os.path.join(cache_dir, "transformers")
    os.environ["HF_DATASETS_CACHE"] = os.path.join(cache_dir, "datasets")
    
    # Create subdirectories
    os.makedirs(os.environ["TRANSFORMERS_CACHE"], exist_ok=True)
    os.makedirs(os.environ["HF_DATASETS_CACHE"], exist_ok=True)
    
    # Check cache size
    cache_size = check_cache_size(cache_dir)
    print(f"Hugging Face cache directory: {cache_dir}")
    print(f"Current cache size: {cache_size:.2f} GB")
    
    # Clean cache if requested
    if args.clean:
        clean_cache(cache_dir)
        
        # Check cache size after cleaning
        cache_size = check_cache_size(cache_dir)
        print(f"Cache size after cleaning: {cache_size:.2f} GB")
    
    print(f"\nTo use this cache directory in your code, add:")
    print("os.environ['HF_HOME'] = " + repr(cache_dir))
    print("os.environ['TRANSFORMERS_CACHE'] = " + repr(os.path.join(cache_dir, "transformers")))
    print("os.environ['HF_DATASETS_CACHE'] = " + repr(os.path.join(cache_dir, "datasets")))

if __name__ == "__main__":
    main() 