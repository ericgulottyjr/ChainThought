#!/bin/bash
#
# Batch job script for running on the shared computing cluster
#

#SBATCH --job-name=deepseek_gsm8k
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=48:00:00
#SBATCH -p cmp  # Use the compute partition
#SBATCH --gres=gpu:1    # Request 1 GPU
#SBATCH --output=job_output_%j.out
#SBATCH --error=job_output_%j.err
#SBATCH -e  # Send an email when the job ends
#SBATCH --mail-user=your.email@example.com

# Display job info using qstat commands
echo "=== Job Information ==="
qstat -u $USER
qstat -j $SLURM_JOB_ID
qstat -s r  # Show running jobs

# Load necessary modules
echo "=== Loading Modules ==="
module avail  # Show available modules
module spider python  # Look for python module
module load python/3.11  # Load Python
module load cuda/12.1  # Load CUDA
module list  # Show loaded modules

# Setup environment
echo "=== Setting up environment ==="
export OMP_NUM_THREADS=8  # Set number of threads
export PYTHONPATH="$(pwd):$PYTHONPATH"

# Create and activate a virtual environment if needed
if [ ! -d ".venv" ]; then
    echo "Creating new virtual environment..."
    python -m venv .venv
fi

source .venv/bin/activate

# Install requirements
echo "=== Installing requirements ==="
pip install -r requirements.txt

# Clean offload folder
echo "=== Cleaning offload folder ==="
python clean_offload.py

# Check environment
echo "=== Checking environment ==="
python check_environment.py

# Determine which operation to run based on argument
OPERATION=${1:-"baseline"}  # Default to baseline if no arg provided

if [ "$OPERATION" = "baseline" ]; then
    echo "=== Running baseline evaluation ==="
    # Run with 16 CPUs (using -pe omp 16 as per the cheat sheet)
    python baseline_evaluate.py --num_samples 100 --offload_folder ./offload_folder
elif [ "$OPERATION" = "train" ]; then
    echo "=== Running fine-tuning ==="
    # Run with 16 CPUs and 28 tasks per node
    python train.py --output_dir ./outputs --num_train_epochs 3 --offload_folder ./offload_folder --per_device_train_batch_size 2 --gradient_accumulation_steps 16
elif [ "$OPERATION" = "evaluate" ]; then
    echo "=== Running fine-tuned evaluation ==="
    python evaluate.py --peft_model_path ./outputs --num_samples 100 --output_dir ./finetuned_results --offload_folder ./offload_folder
elif [ "$OPERATION" = "compare" ]; then
    echo "=== Running comparison ==="
    python compare_results.py --baseline_dir ./baseline_results --finetuned_dir ./finetuned_results --output_dir ./comparison_results
else
    echo "Unknown operation: $OPERATION"
    echo "Usage: sbatch run_on_cluster.sh [baseline|train|evaluate|compare]"
    exit 1
fi

# Check queue status after job starts (for debugging)
qstat -g c  # Display the list of queues and load information
qstat -q  # Display jobs running in a particular queue

echo "Job completed at $(date)" 