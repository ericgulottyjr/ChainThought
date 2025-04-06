#!/bin/bash
# Quick script to run baseline evaluation with correct cache directory
# Use this for interactive sessions on the cluster

# Setup working directory
WORKDIR=$(pwd)
echo "Working directory: $WORKDIR"

# Set Hugging Face cache to be in the project directory
export HF_HOME="$WORKDIR/.hf_cache"
export TRANSFORMERS_CACHE="$WORKDIR/.hf_cache/transformers"
export HF_DATASETS_CACHE="$WORKDIR/.hf_cache/datasets"

# Create cache directories
mkdir -p $HF_HOME/transformers
mkdir -p $HF_HOME/datasets
echo "Hugging Face cache set to: $HF_HOME"

# Create offload directory
mkdir -p $WORKDIR/offload_folder

# Clean previous cache if needed (uncomment if needed)
# python clean_cache.py --clean --cache_dir $HF_HOME

# Setup environment
python -c "from cluster_config import setup_environment; setup_environment()"

# Run the evaluation or other command
# Replace with the specific command you're trying to run
python baseline_evaluate.py --num_samples 100 --offload_folder $WORKDIR/offload_folder

echo "Job completed at $(date)" 