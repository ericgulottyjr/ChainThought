# Comparing Approaches

This document compares the key features and design choices across the different training approaches in our repository.

## Feature Comparison

| Feature | BaseLoRAFineTuner.py | EnhancedLoRAFinetuner.py | QLoRATrainer.py | GSM8kSolver.py |
|---------|----------------------|--------------------------|----------------|----------------|
| Precision | FP16/BF16 | FP16/BF16 | 4-bit Quantization | 4-bit Quantization |
| LoRA Integration | Basic | Enhanced | QLoRA | QLoRA |
| Sequence Length | 1024 | 768 | 2048 | 256 |
| Prompting Style | Basic | Basic | Basic | Direct Q&A |
| Memory Optimization | Minimal | Moderate | Advanced | Advanced |
| Training Epochs | 3 | 3 | 2 | 3 |
| Parameter Counting | No | Yes | Yes | No |
| Data Processing | Simple | Enhanced | Optimized | Cached |
| Inference Pipeline | No | No | No | Yes |
| Evaluation Metrics | External | External | External | Built-in |

## Evolution of Approaches

### 1. Basic Approach: BaseLoRAFineTuner.py

**Core Features:**
- Basic LoRA implementation
- Standard sequence length (1024 tokens)
- Simple data processing pipeline
- Focus on model architecture exploration

**Limitations:**
- Limited memory optimization
- Basic data processing capabilities
- No built-in inference or evaluation

### 2. Improved Base: EnhancedLoRAFinetuner.py

**Evolution from BaseLoRAFineTuner.py:**
- Reduced sequence length (768 tokens) for efficiency
- Added parameter counting for debugging
- Enhanced data processing pipeline
- Improved documentation and configuration options

**Advantages:**
- Better memory efficiency
- More robust data handling
- Better code organization and readability
- Dataset caching for faster training

### 3. Quantized Approach: QLoRATrainer.py

**Evolution from EnhancedLoRAFinetuner.py:**
- 4-bit quantization for memory efficiency
- Longer sequences (2048 tokens) for better reasoning
- Advanced memory optimization techniques
- Focus on efficient training with limited epochs
- Offloading to CPU to save GPU memory

### 5. Independent Approach: GSM8kSolver.py (formerly checkpoint-1500.py)

**Unique Characteristics:**
- Significantly shorter sequences (256 tokens)
- Direct question-answer format prompting
- Dataset caching for improved training efficiency
- Complete training + inference pipeline in a single script
- Built-in evaluation with side-by-side comparisons

**Special Features:**
- Full inference pipeline included with the training code
- Direct comparison of base vs. fine-tuned outputs
- Explicit exact-match and F1 evaluation metrics
- More efficient data processing with caching
- Focus on inference speed and deployment

## Key Tradeoffs

### Memory vs. Precision
- **BaseLoRAFineTuner.py/EnhancedLoRAFinetuner.py**: Prioritize precision with full-parameter training
- **QLoRATrainer.py/GSM8kSolver.py**: Prioritize memory efficiency with 4-bit quantization

### Sequence Length
- **GSM8kSolver.py**: Shortest (256) for efficiency 
- **EnhancedLoRAFinetuner.py**: Moderate (768) for balanced approach
- **BaseLoRAFineTuner.py**: Standard (1024) for adequate context
- **QLoRATrainer.py**: Longest (2048) for detailed reasoning

### Reasoning Approach
- **GSM8kSolver.py**: Direct question-answer format
- **Others**: Simple input-target concatenation

### Training Resources
- **QLoRATrainer.py**: Efficient compute (2 epochs)
- **BaseLoRAFineTuner.py/EnhancedLoRAFinetuner.py/GSM8kSolver.py**: Moderate (3 epochs)

## Conclusion

These four approaches represent a progression of fine-tuning techniques for mathematical reasoning, from basic implementations to sophisticated approaches with specialized prompting and memory optimizations. The QLoRATrainer.py model demonstrates the benefits of quantization for efficient training, while GSM8kSolver.py offers a deployment-focused approach with an integrated inference pipeline. 