#!/bin/bash
# Advanced job script for evaluation on the compute cluster
# Based on the cluster cheat sheet

# Job definition for qsub
#$ -N deepseek_eval
#$ -pe omp 8
#$ -l mem_per_core=4G
#$ -l h_rt=12:00:00
#$ -t 1-1
#$ -j y
#$ -v

# Request GPU resources (evaluation needs less resources than training)
#$ -l gpu=1
#$ -l gpu_c=3.5

# Email notifications
#$ -m e
#$ -M your.email@example.com

# Use cluster-specific queue for jobs requiring GPUs
#$ -q gpu.q

# Set up environment
echo "Current working directory: $(pwd)"
echo "Job ID: $JOB_ID"
echo "Task ID: $SGE_TASK_ID"

# Load required modules
module avail
module load cuda/12.1
module load python/3.11
module list

# Set environment variables
export OMP_NUM_THREADS=8
export MKL_NUM_THREADS=8

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

# Determine which evaluation to run
EVAL_TYPE=${1:-"baseline"}  # Default to baseline if no argument provided

if [ "$EVAL_TYPE" = "baseline" ]; then
    # Run baseline evaluation
    echo "Starting baseline evaluation"
    OUTPUT_DIR="$WORKDIR/baseline_results"
    mkdir -p $OUTPUT_DIR
    
    python baseline_evaluate.py \
        --model_name_or_path deepseek-ai/deepseek-llm-7b-base \
        --num_samples 100 \
        --max_new_tokens 512 \
        --output_dir $OUTPUT_DIR \
        --offload_folder $WORKDIR/offload_folder \
        --seed 42
        
elif [ "$EVAL_TYPE" = "finetuned" ]; then
    # Run fine-tuned model evaluation
    echo "Starting fine-tuned model evaluation"
    OUTPUT_DIR="$WORKDIR/finetuned_results"
    mkdir -p $OUTPUT_DIR
    
    python evaluate.py \
        --model_name_or_path deepseek-ai/deepseek-llm-7b-base \
        --peft_model_path $WORKDIR/outputs \
        --num_samples 100 \
        --max_new_tokens 512 \
        --output_dir $OUTPUT_DIR \
        --offload_folder $WORKDIR/offload_folder \
        --seed 42
        
elif [ "$EVAL_TYPE" = "compare" ]; then
    # Run comparison between baseline and fine-tuned
    echo "Starting comparison analysis"
    OUTPUT_DIR="$WORKDIR/comparison_results"
    mkdir -p $OUTPUT_DIR
    
    python compare_results.py \
        --baseline_dir $WORKDIR/baseline_results \
        --finetuned_dir $WORKDIR/finetuned_results \
        --output_dir $OUTPUT_DIR
        
else
    echo "Unknown evaluation type: $EVAL_TYPE"
    echo "Usage: qsub evaluate_on_cluster.sh [baseline|finetuned|compare]"
    exit 1
fi

# Check the queue status
qstat -j $JOB_ID

# Print job completion info
echo "Job completed at $(date)"
echo "Evaluation results saved to $OUTPUT_DIR" 