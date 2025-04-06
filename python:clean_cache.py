import os

# Set environment variables to use the specified cache directory
os.environ["HF_HOME"] = cache_dir
os.environ["HF_DATASETS_CACHE"] = os.path.join(cache_dir, "datasets")

# Create subdirectories
os.makedirs(os.path.join(cache_dir, "transformers"), exist_ok=True)
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
print("os.environ['HF_DATASETS_CACHE'] = " + repr(os.path.join(cache_dir, "datasets"))) 