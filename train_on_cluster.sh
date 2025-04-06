#!/bin/bash
# Advanced job script for fine-tuning DeepSeek on GSM8K dataset
# Based on the cluster cheat sheet

# Job definition for qsub
#$ -N deepseek_train
#$ -pe omp 16
#$ -l mem_per_core=6G
#$ -l h_rt=48:00:00
#$ -t 1-1
#$ -j y
#$ -v

# Request GPU resources
#$ -l gpu=1
#$ -l gpu_c=3.5

# Email notifications
#$ -m e
#$ -M your.email@example.com

# Directory and path settings
echo "Current working directory: $(pwd)"
echo "Job ID: $JOB_ID"
echo "Task ID: $SGE_TASK_ID"

# Load required modules 
module avail
module load cuda/12.1
module load python/3.11
module list

# Job dependency management
# If you want to make this job dependent on another job completing
# Uncomment the following line and replace [jobid] with the ID of the job to wait for
# -hold_jid [jobid]

# Set environment variables
export OMP_NUM_THREADS=16
export MKL_NUM_THREADS=16

# Setup working directory
WORKDIR=$(pwd)
echo "Working directory: $WORKDIR"

# Create offload directory
mkdir -p $WORKDIR/offload_folder

# Activate virtual environment
source .venv/bin/activate

# Install dependencies
pip install -r requirements.txt

# Clean previous outputs
python clean_offload.py --offload_folder $WORKDIR/offload_folder

# Check environment before starting
python check_environment.py

# Running with parallel MPI workload configuration (based on cluster cheat sheet)
# -pe mpi 28_tasks_per_node
# Define training-specific parameters
EPOCHS=3
BATCH_SIZE=2
GRAD_ACCUM=16
MAX_SEQ_LEN=2048

# Run the training
echo "Starting training with epochs=$EPOCHS, batch_size=$BATCH_SIZE, grad_accum=$GRAD_ACCUM"
python train.py \
    --model_name_or_path deepseek-ai/deepseek-llm-7b-base \
    --output_dir $WORKDIR/outputs \
    --num_train_epochs $EPOCHS \
    --per_device_train_batch_size $BATCH_SIZE \
    --gradient_accumulation_steps $GRAD_ACCUM \
    --learning_rate 2e-5 \
    --max_seq_length $MAX_SEQ_LEN \
    --lora_r 16 \
    --lora_alpha 32 \
    --lora_dropout 0.05 \
    --offload_folder $WORKDIR/offload_folder \
    --seed 42

# Check the queue status
qstat -j $JOB_ID

# Print job completion info
echo "Job completed at $(date)"
echo "Training output saved to $WORKDIR/outputs" 