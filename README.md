# ChainThought
DS542 Final Project - Fine-tuning Deepseek-7B on GSM8K

## Dataset

[GSM8K](https://github.com/openai/grade-school-math) - Grade School Math 8K is a dataset of 8.5K high-quality linguistically diverse grade school math word problems created by human problem writers.

## Base Model

[Deepseek 7B](https://github.com/deepseek-ai/deepseek-LLM) - A foundation model designed for code and natural language tasks with strong performance on mathematical reasoning.

## Setup

1. Clone the repository:
```bash
git clone https://github.com/ericgulottyjr/ChainThought.git
cd ChainThought
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

## Fine-tuning

To fine-tune the Deepseek-7B model on GSM8K, run:

```bash
python train.py --output_dir ./outputs --num_train_epochs 3
```

Additional arguments:
- `--model_name_or_path`: Path to the base model (default: "deepseek-ai/deepseek-llm-7b-base")
- `--per_device_train_batch_size`: Batch size per device (default: 4)
- `--learning_rate`: Learning rate (default: 2e-5)
- `--gradient_accumulation_steps`: Gradient accumulation steps (default: 8)
- `--max_seq_length`: Maximum sequence length (default: 2048)
- `--lora_r`: LoRA attention dimension (default: 16)
- `--lora_alpha`: LoRA alpha parameter (default: 32)
- `--lora_dropout`: LoRA dropout parameter (default: 0.05)

## Evaluation

### Baseline Evaluation

To evaluate the baseline performance of Deepseek-7B on GSM8K without fine-tuning, run:

```bash
python baseline_evaluate.py --num_samples 100
```

This will save the evaluation results to the `./baseline_results` directory.

### Fine-tuned Model Evaluation

To evaluate the fine-tuned model on the GSM8K test set, run:

```bash
python evaluate.py --peft_model_path ./outputs --num_samples 100 --output_dir ./finetuned_results
``` 

Additional arguments:
- `--model_name_or_path`: Path to the base model (default: "deepseek-ai/deepseek-llm-7b-base")
- `--max_new_tokens`: Maximum number of new tokens to generate (default: 512)

### Comparing Results

To compare the performance of the baseline model versus the fine-tuned model, run:

```bash
python compare_results.py --baseline_dir ./baseline_results --finetuned_dir ./finetuned_results --output_dir ./comparison_results
```

This will generate a detailed comparison report with statistics and visualizations in the `./comparison_results` directory.

## Generation

To generate responses from the fine-tuned model on custom math problems, run:

```bash
python generate.py --peft_model_path ./outputs
```

You will be prompted to enter a math problem. Alternatively, you can provide a problem directly:

```bash
python generate.py --peft_model_path ./outputs --question "Janet's ducks lay 16 eggs per day. She eats three for breakfast every morning and bakes muffins for her friends every day with four eggs. She sells the remainder at the farmers' market daily for $2 per fresh duck egg. How much in dollars does she make every day at the farmers' market?"
```

Additional arguments:
- `--model_name_or_path`: Path to the base model (default: "deepseek-ai/deepseek-llm-7b-base")
- `--max_new_tokens`: Maximum number of new tokens to generate (default: 512)
- `--temperature`: Sampling temperature (default: 0.7)

## Hardware Requirements

- RAM: At least 16GB
- GPU: At least 16GB VRAM (preferred NVIDIA GPU with CUDA support)
- Storage: At least 20GB free space

## Project Structure

- `train.py`: Script for fine-tuning the model
- `evaluate.py`: Script for evaluating the fine-tuned model on the GSM8K test set
- `baseline_evaluate.py`: Script for establishing baseline performance of Deepseek-7B
- `compare_results.py`: Script for comparing baseline and fine-tuned model results
- `generate.py`: Script for generating responses from the model
- `requirements.txt`: Required Python packages

## References

- GSM8K: https://github.com/openai/grade-school-math
- Deepseek-LLM: https://github.com/deepseek.ai/deepseek-LLM
- PEFT: https://github.com/huggingface/peft

## Running on Compute Cluster

This project includes scripts for running on the shared computing cluster. The following instructions will help you run the code efficiently on the cluster.

### Checking the Environment

Before running any jobs, check the cluster environment:

```bash
python check_environment.py
```

This will provide information about the available hardware, software, and configurations.

### Submitting Jobs

We've provided several job submission scripts that use the cluster's job scheduling system:

#### General Job Submission

For a general job that can run any operation:

```bash
# Submit a job to run baseline evaluation
qsub run_on_cluster.sh baseline

# Submit a job to run training
qsub run_on_cluster.sh train

# Submit a job to evaluate the fine-tuned model
qsub run_on_cluster.sh evaluate

# Submit a job to compare results
qsub run_on_cluster.sh compare
```

#### Specialized Training Job

For training with optimized settings:

```bash
qsub train_on_cluster.sh
```

This script is optimized for the training task with appropriate resource requests (16 CPUs, 1 GPU).

#### Specialized Evaluation Job

For evaluation with optimized settings:

```bash
# Run baseline evaluation
qsub evaluate_on_cluster.sh baseline

# Run fine-tuned model evaluation
qsub evaluate_on_cluster.sh finetuned

# Run comparison
qsub evaluate_on_cluster.sh compare
```

### Monitoring Jobs

To monitor your jobs on the cluster:

```bash
# List all your current jobs
qstat -u $USER

# List of all running jobs
qstat -s r

# Display the resources requested by a job
qstat -j JOB_ID

# Display the list of queues and load information
qstat -g c

# Show pending jobs
qstat -s p
```

### Job Requirements

- Training job: 48 hours wall time, 1 GPU, 16 CPUs, 96GB memory
- Evaluation job: 12 hours wall time, 1 GPU, 8 CPUs, 32GB memory
- Comparison job: 1 hour wall time, 0 GPU, 4 CPUs, 16GB memory

### Additional Cluster Commands

The job scripts leverage the following cluster commands:

- `module`: For loading software modules (Python, CUDA)
- `qstat`: For checking job status
- `-pe omp N`: Request N CPUs per task
- `-l mem_per_core=XG`: Request X GB of memory per CPU core
- `-l h_rt=HH:MM:SS`: Set hard runtime limit for the job
- `-l gpu=N`: Request N GPUs

See the shared computing cluster cheat sheet for more details on available commands.

### Managing Hugging Face Cache

When running on the compute cluster, the Hugging Face cache directory is set to a local folder in your project directory (`.hf_cache`). This prevents issues with home directory storage limits.

To check or clean the cache:

```bash
# Check the current cache size
python clean_cache.py

# Clean the cache if it gets too large
python clean_cache.py --clean
```

If you encounter any model loading errors related to cache or storage issues, you can try:

1. Clean the cache using the script above
2. Explicitly set the cache directory in your environment:
   ```bash
   export HF_HOME="$(pwd)/.hf_cache"
   export TRANSFORMERS_CACHE="$(pwd)/.hf_cache/transformers" 
   export HF_DATASETS_CACHE="$(pwd)/.hf_cache/datasets"
   ```
3. Run your command again

The job submission scripts already include these environment settings.
