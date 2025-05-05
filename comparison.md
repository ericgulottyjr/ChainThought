# Fine-Tuning Models for Deepseek-7B on GSM8K: A Comprehensive Comparison

This document compares five different approaches to fine-tuning the Deepseek-7B model on the GSM8K mathematical reasoning dataset, showing the evolution of techniques and optimizations.

## Quick Reference Table

| Feature | BaseLoRAFineTuner.py | EnhancedLoRAFinetuner.py | QLoRATrainer.py | MemoryOptimizedLoRA.py | GSM8kSolver.py |
|---------|----------|-----------|-------------------------|---------------------------|-------------------|
| **Base Model** | deepseek-7b | deepseek-7b | deepseek-7b | deepseek-7b | deepseek-7b |
| **Quantization** | None | None | 4-bit (NF4) | None (Full precision BF16/FP16) | 4-bit |
| **LoRA Rank** | 16 | 16 | 16 | 32 (reduced from 64) | 8/16 |
| **LoRA Alpha** | 32 | 32 | 32 | 32 (reduced from 64) | 32 |
| **Target Modules** | Not explicit | q/k/v/o_proj | q/k/v/o_proj | q/k/v/o_proj | Not explicitly shown |
| **Batch Size** | 1 | 1 | 1 | 4 (reduced from 8) | 2 |
| **Grad Accum** | 16 | 16 | 16 | 2 (increased from 1) | 4 |
| **Learning Rate** | 3e-4 | Variable | 1e-4 | 5e-5 | 5e-4 |
| **Epochs** | 3 | Variable | 2 | 3 | 3 |
| **Sequence Length** | 1024 | 768 | 2048 | 1536 (reduced from 2048) | 256 |
| **Prompting** | Simple concat | Simple concat | Simple concat | Chain-of-thought | Q&A format |
| **Special Features** | Basic implementation | Enhanced metrics, BF16 support | QLoRA, custom metrics | Full precision, CoT prompting, memory optimization | Dataset caching, inference pipeline |

## Evolution Timeline

```
BaseLoRAFineTuner.py (Basic) --> EnhancedLoRAFinetuner.py (Enhanced) 
                      ↓
                 QLoRATrainer.py (QLoRA Approach)
                      ↓
                 MemoryOptimizedLoRA.py (Full Precision CoT)
                      
GSM8kSolver.py (Independent Approach)
```

## Detailed Comparison

### 1. Base Implementation: BaseLoRAFineTuner.py (formerly train.py)

**Key Features:**
- Standard LoRA fine-tuning approach without quantization
- Simple concatenation of input and target texts
- Basic training loop with minimal optimization
- Moderate sequence length (1024 tokens)
- Minimal memory optimization techniques

**Limitations:**
- Limited memory efficiency
- Basic tokenization without special prompting
- No optimization for mathematical reasoning tasks
- Limited evaluation metrics

### 2. Enhanced Implementation: EnhancedLoRAFinetuner.py (formerly train2.py)

**Evolution from BaseLoRAFineTuner.py:**
- Reduced sequence length (768) for better memory efficiency
- Added multiple evaluation metrics
- Improved gradient handling and data processing
- More sophisticated learning rate scheduling
- Better error handling and training stability

**Key Improvements:**
- Enhanced monitoring via more detailed metrics
- Memory optimization through reduced sequence length
- BF16 precision support for better numerical stability
- Improved tokenization and data processing
- More robust training pipeline with error recovery

### 3. QLoRA Approach: QLoRATrainer.py (formerly train1-checkpoint-746.py)

**Novel Approach:**
- Implemented 4-bit quantization (NF4 format) with QLoRA
- Doubled sequence length (2048) for more context
- Two-epoch training with constant-with-warmup LR schedule
- Accelerator integration for distributed training
- Higher dropout (0.10) for better generalization

**Key Features:**
- Memory-efficient 4-bit quantization
- Longer sequences for mathematical reasoning
- Precise optimizer tuning (Adam betas 0.9/0.95)
- Focus on efficient training with limited epochs
- Offloading to CPU to save GPU memory

### 4. Full Precision Chain-of-Thought: MemoryOptimizedLoRA.py (formerly train2-checkpoint-2241.py)

**Evolution from QLoRATrainer.py:**
- Removed quantization for higher precision (BF16/FP16)
- Chain-of-thought prompting ("Let's think step by step:")
- Reduced LoRA rank from 64 to 32 for memory efficiency
- Token masking to focus training on reasoning part
- Memory optimization with reduced sequence length (1536)

**Key Innovations:**
- Explicit instruction for step-by-step reasoning
- Memory optimizations to enable full precision training
- Masking prompt tokens in labels to focus training
- Balanced batch size (4) and gradient accumulation (2)
- Parameter efficient training with explicitly counted parameters

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
- **MemoryOptimizedLoRA.py**: Balanced approach with full precision but memory optimizations

### Sequence Length
- **GSM8kSolver.py**: Shortest (256) for efficiency 
- **EnhancedLoRAFinetuner.py**: Moderate (768) for balanced approach
- **BaseLoRAFineTuner.py**: Standard (1024) for adequate context
- **MemoryOptimizedLoRA.py**: Extended (1536) for better reasoning
- **QLoRATrainer.py**: Longest (2048) for detailed reasoning

### Reasoning Approach
- **MemoryOptimizedLoRA.py**: Explicit chain-of-thought prompting
- **GSM8kSolver.py**: Direct question-answer format
- **Others**: Simple input-target concatenation

### Training Resources
- **QLoRATrainer.py**: Efficient compute (2 epochs)
- **BaseLoRAFineTuner.py/EnhancedLoRAFinetuner.py/GSM8kSolver.py/MemoryOptimizedLoRA.py**: Moderate (3 epochs)

## Conclusion

These five approaches represent a progression of fine-tuning techniques for mathematical reasoning, from basic implementations to sophisticated approaches with specialized prompting and memory optimizations. The MemoryOptimizedLoRA.py model represents a culmination of lessons learned, combining full precision training with chain-of-thought prompting and careful memory optimization. It builds upon the QLoRA efficiency techniques from QLoRATrainer.py while removing quantization for higher precision training. Meanwhile, GSM8kSolver.py continues to offer an independent approach focusing on deployment efficiency. 