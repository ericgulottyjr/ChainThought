#!/bin/bash

# Script to run train_basic.py on GPU 1 only
# Usage: bash run_basic.sh [--small]

# Tell PyTorch to only use GPU 1
export CUDA_VISIBLE_DEVICES=1

# Show which GPU is being used
echo "Running on GPU 1 (NVIDIA A100 80GB PCIe)"

# Memory optimizations
export PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:128,expandable_segments:True

# Set environment variables for better performance
export TOKENIZERS_PARALLELISM=false  # Avoid warnings
export TRANSFORMERS_OFFLINE=1  # Use cached models only

# Show GPU status before starting
echo "GPU status before training:"
nvidia-smi

# Run train_basic.py with memory-optimized settings
echo "Starting training with train_basic.py on GPU 1..."
python train_basic.py --config default.yaml "$@"

# Check exit status
STATUS=$?
if [ $STATUS -eq 0 ]; then
  echo "Training finished successfully."
else
  echo "Training failed with exit code $STATUS."
fi

# Show GPU status after completion
echo "GPU status after training:"
nvidia-smi

echo "Done!" 