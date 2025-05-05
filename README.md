# ChainThought

## Repository Structure

```
ChainThought/
├── models/                      # Model implementations
│   ├── BaseLoRAFineTuner.py     # Base LoRA fine-tuning implementation
│   ├── EnhancedLoRAFinetuner.py # Enhanced LoRA fine-tuning with added capabilities
│   ├── GSM8kSolver.py           # Model for solving GSM8K math problems
│   └── QLoRATrainer.py          # Quantized LoRA training implementation
│
├── config/                      # Configuration files for models
│   ├── LoRAFineTuner.yaml       # Configuration for base LoRA fine-tuner
│   ├── GSM8KSolver.yaml         # Configuration for GSM8K solver
│   └── QLoRATrainer.yaml        # Configuration for QLoRA training
│
├── data/                        # Dataset storage
│   ├── raw/                     # Raw, unprocessed datasets
│   ├── formatted/               # Processed datasets ready for training
│   └── .hf_home/                # Hugging Face cache
│
├── evaluation/                  # Evaluation scripts and utilities
│   ├── evaluate.py              # Core evaluation script
│   ├── evaluate_metrics.py      # Metrics calculation for evaluation
│   └── evaluate_pipeline.py     # End-to-end evaluation pipeline
│
├── wandb_screenshots/           # Weights & Biases monitoring screenshots
│
├── outputs/                     # Output directory for model checkpoints and results
│
├── hyperparamTrain2.py          # Training script with hyperparameter support
├── hyperparamTest.py            # Testing script with hyperparameter support
├── prepare_data.py              # Data preparation and processing script
├── utils.py                     # Utility functions used across the project
├── requirements.txt             # Python dependencies
├── sweep_config.yaml            # Configuration for hyperparameter sweeps
│
└── report.pdf                   # Project report (final version)
```

This repository contains code for the ChainThought project, which focuses on fine-tuning language models using LoRA (Low-Rank Adaptation) and QLoRA (Quantized LoRA) techniques. The codebase is organized to support training, evaluation, and inference with various model configurations. 