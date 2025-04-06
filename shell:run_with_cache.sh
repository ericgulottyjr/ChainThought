# Setup working directory
WORKDIR=$(pwd)
echo "Working directory: $WORKDIR"

# Set Hugging Face cache to be in the project directory
export HF_HOME="$WORKDIR/.hf_cache"
# Remove the deprecated TRANSFORMERS_CACHE variable
export HF_DATASETS_CACHE="$WORKDIR/.hf_cache/datasets"

# Create cache directories
mkdir -p $HF_HOME
mkdir -p $HF_HOME/datasets
echo "Hugging Face cache set to: $HF_HOME"

# Create offload directory
mkdir -p $WORKDIR/offload_folder

# Clean previous cache if needed (uncomment if needed)
# python clean_cache.py --clean --cache_dir $HF_HOME

# Setup environment
python -c "from cluster_config import setup_environment; setup_environment()"

# Run the simplified evaluation
python baseline_evaluate.py \
  --num_samples 10 \
  --max_new_tokens 100 \
  --output_dir $WORKDIR/baseline_results

echo "Job completed at $(date)" 