# ChainThought Project

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

---

## AI Usage Statement

Throughout the development of this final project we utilized Artificial Intelligence (AI) to streamline our programming process. The following summarizes how these AI systems were employed:

| Phase | Purpose of AI Assistance | Typical AI Tools / Models Invoked |
|-------|--------------------------|-----------------------------------|
| **Code generation** | Drafted boilerplate modules, helper functions, and template scripts to accelerate initial implementation. | Large‑language‑model (LLM) chat assistants (e.g., OpenAI ChatGPT) and IDE code‑completion plugins. |
| **Debugging** | Explained stack traces, suggested fixes for dependency conflicts, memory/OOM errors, and performance bottlenecks. | Conversational LLMs (natural‑language queries paired with error logs). |
| **Evaluation & analysis** | Proposed metric definitions, drafted small evaluation scripts, and interpreted preliminary results. | LLMs for pseudocode → Python translations; automatic docstring generation. | 
| **Design suggestions & iterative improvements** | Recommended hyper‑parameter tweaks, data‑pipeline optimizations, and experiment‑tracking conventions. | Chat assistants and model fine‑tuning guides. |

**Summary:** AI models served as accelerators for repetitive or exploratory tasks. All substantive code, analyses, and written discussion reflect our team's work.


---

# Guide: Running This Project with Local DeepSeek LLMs

Follow the steps below to clone the repo, build the three fine-tuned DeepSeek-7B models (V1 / V2 / V3) entirely on your own machine, serve them with **LM Studio**, and run the chat app locally.

---

## 1 - Clone & bootstrap the Chatbot repo

```bash
git clone https://github.com/dlaboy25/chain-thought-chat
cd chain-thought-chat
npm install
```

---

## 2 - Create your `.env.local`

```bash
cp w.env.example .env.local
```

* Fill in the keys required in the comments (`AUTH_SECRET`, `DATABASE_URL`, …).
* Add the LM Studio endpoint you’ll start later:

```dotenv
DEEPSEEK_BASE_URL=http://localhost:1234/v1
```

---

## 3 - Download model assets from this repo to use in the Chatbot

| What                        | Where it ends up                                  |
|-----------------------------|---------------------------------------------------|
| DeepSeek-7B **base** model  | `~/models/deepseek-base/` (≈ 14 GB from HF)       |
| Adapter **V1**              | `ChainThought/adapters/v1/*` → `~/models/deepseek-v1/adapter/` |
| Adapter **V2**              | `ChainThought/adapters/v2/*` → `~/models/deepseek-v2/adapter/` |
| Adapter **V3**              | `ChainThought/adapters/v3/*` → `~/models/deepseek-v3/adapter/` |

Download the base once:

```bash
huggingface-cli download deepseek-ai/deepseek-llm-7b-base \
  --local-dir ~/models/deepseek-base --local-dir-use-symlinks False
```

Copy adapters into place:

```bash
for ver in v1 v2 v3; do
  mkdir -p ~/models/deepseek-$ver/adapter
  cp ./adapters/$ver/adapter_* ~/models/deepseek-$ver/adapter/
done
```

---

## 4 - Merge / Convert to GGUF

Set up a Python env (only once):

```bash
python3 -m venv ~/venvs/ChainThought
source ~/venvs/ChainThought/bin/activate
pip install torch transformers peft accelerate safetensors \
            llama-cpp-python[convert] \
            git+https://github.com/ggerganov/llama.cpp.git@master#subdirectory=gguf-py
```

### 4a - Convert the untouched **DeepSeek-Base**

> The base model has no LoRA adapter, so you can convert it directly.

```bash
python -m llama_cpp.convert \
  ~/models/deepseek-base            \
  ~/models/deepseek-base-f16.gguf   \
  --outtype f16
```

### 4b - Merge each LoRA variant and convert

Run **once for each** variant (`v1`, `v2`, `v3`):

```bash
# ---------- replace vX with v1 / v2 / v3 ----------
cd ~/models/deepseek-vX                      # base & adapter sub-dirs exist

python - <<'PY'
from pathlib import Path
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel
import torch

BASE  = Path('../deepseek-base').resolve()
ADAPT = Path('adapter').resolve()
OUT   = Path('merged').resolve(); OUT.mkdir(exist_ok=True)

print("Merging", ADAPT, "→", OUT)
m = AutoModelForCausalLM.from_pretrained(BASE, torch_dtype=torch.float16)
m = PeftModel.from_pretrained(m, ADAPT)
m.merge_and_unload()
m.save_pretrained(OUT, safe_serialization=True)
AutoTokenizer.from_pretrained(BASE).save_pretrained(OUT)
PY

# convert HF → GGUF FP16
python -m llama_cpp.convert \
  merged  ./deepseek-vX-f16.gguf  --outtype f16
```

After these steps you will have

`~/models/deepseek-base-f16.gguf`,
`~/models/deepseek-v1/deepseek-v1-f16.gguf`,  
`~/models/deepseek-v2/deepseek-v2-f16.gguf`,  
`~/models/deepseek-v3/deepseek-v3-f16.gguf`.

---

## 5 - Load models into LM Studio

1. Install LM Studio ≥ 0.3.x from <https://lmstudio.ai>.  
2. **Models → Local Models → Add Local Model** → import each `*.gguf`.  
3. Open the **Developer (`</>`) tab** and load in the different models. *Depending on your RAM/VRAM you may only be able to keep one or two 7-B models loaded at the same time.*
4. Start the server

---


## 6 - Run the Next.js app

```bash
pnpm run dev
```

Visit <http://localhost:3000>, choose **deepseek base**, **deepseek v1**, **deepseek v2**, or **deepseek v3** from the model selector, and chat – all LLM traffic is processed locally by your LM Studio server. Remember that you can only chat with the LLMs that you are running locally.
