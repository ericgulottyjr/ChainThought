#!/bin/bash

# Script to run training on a single specific GPU with memory-optimized settings
# Usage: bash run_single_gpu.sh [GPU_ID] [--small]
# Example: bash run_single_gpu.sh 1 --small  (runs on GPU 1 with small dataset)

# Parse GPU ID argument (default to 1 if not specified)
GPU_ID=${1:-1}
shift 2>/dev/null || true  # Shift to next arg if possible

# Validate GPU ID
if ! [[ "$GPU_ID" =~ ^[0-9]+$ ]]; then
    echo "ERROR: GPU ID must be a number"
    echo "Usage: bash run_single_gpu.sh [GPU_ID] [--small]"
    exit 1
fi

# Check if the specified GPU exists
NUM_GPUS=$(nvidia-smi --list-gpus | wc -l)
if [[ $GPU_ID -ge $NUM_GPUS ]]; then
    echo "ERROR: GPU $GPU_ID doesn't exist. System has $NUM_GPUS GPUs (0 to $((NUM_GPUS-1)))"
    exit 1
fi

echo "Training on GPU $GPU_ID"

# Set the CUDA_VISIBLE_DEVICES to only use the specified GPU
export CUDA_VISIBLE_DEVICES=$GPU_ID

# CUDA memory management - more aggressive settings
export PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:64,expandable_segments:True

# PyTorch memory management
export PYTORCH_NO_CUDA_MEMORY_CACHING=1

# Enable specific optimizations
export ACCELERATE_USE_CPU_OFFLOAD=True
export ACCELERATE_MIXED_PRECISION=fp16
export CUDA_LAUNCH_BLOCKING=1

# Cleanup function
cleanup() {
  echo "Cleaning up processes..."
  pkill -P $$ python 2>/dev/null || true
}
trap cleanup EXIT

# Show GPU status before training
echo "GPU status before training:"
nvidia-smi

# Run with very memory-efficient settings:
# - Small batch size (1)
# - More gradient accumulation steps (32)
# - Reduced max token length (512 instead of 768)
echo "Starting memory-optimized training on GPU $GPU_ID..."

python train_accelerate.py \
  --config default.yaml \
  --batch_size 1 \
  --gradient_accumulation_steps 32 \
  --max_length 512 \
  "$@"

# Check exit status
STATUS=$?
if [ $STATUS -eq 0 ]; then
  echo "Training finished successfully."
else
  echo "Training failed with exit code $STATUS."
fi

# Print GPU status after training
echo "GPU status after training:"
nvidia-smi

echo "Done!" 