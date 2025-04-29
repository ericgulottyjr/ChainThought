#!/bin/bash
#$ -P dl4ds
#$ -l gpus=1             # Request 1 GPU
#$ -l gpu_type=A100      # Specify A100
#$ -l h_rt=24:00:00     # Request 24 hours (adjust as needed for full dataset)
#$ -l mem_total=128G    # Request substantial RAM
#$ -o outputs/train_full_job_$JOB_ID.log
#$ -e outputs/train_full_job_$JOB_ID.err
#$ -j y
#$ -m beas
#$ -M bwong@bu.edu      # Replace with your email

# Create output directories if they don't exist
mkdir -p outputs

# Print environment information
echo "Job started at $(date)"
echo "Running on host: $(hostname)"
echo "Current directory: $(pwd)"

# Set up environment
cd /projectnb/dl4ds/students/bwong/ChainThought
module load miniconda
source activate chainstep

# Show GPU info
echo "GPU information:"
nvidia-smi

# Run training on the FULL dataset
# Make sure default.yaml has the desired hyperparameters
python train.py --config default.yaml

# Note: Do NOT use the --small flag for the final training run

echo "Job finished at $(date)" 