#  Code Generation Assistant — LLM Fine-Tuning with QLoRA

![Python](https://img.shields.io/badge/Python-3.11-blue?style=for-the-badge&logo=python)
![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-ee4c2c?style=for-the-badge&logo=pytorch)
![HuggingFace](https://img.shields.io/badge/HuggingFace-Transformers-yellow?style=for-the-badge&logo=huggingface)
![PEFT](https://img.shields.io/badge/PEFT-QLoRA-green?style=for-the-badge)
![WandB](https://img.shields.io/badge/Weights_&_Biases-Tracking-orange?style=for-the-badge&logo=weightsandbiases)
![License](https://img.shields.io/badge/License-MIT-purple?style=for-the-badge)

> Fine-tuning **Mistral-7B** on **CodeAlpaca-20k** using QLoRA (4-bit quantization + LoRA adapters) to build a domain-specific code generation assistant.

---

##  Architecture

```
Base Model (Mistral-7B)
        │
        ▼
4-bit Quantization (BitsAndBytes)
        │
        ▼
LoRA Adapters (r=64, only 0.1% params trained)
        │
        ▼
Fine-tuned Code Assistant
```

---

##  Project Structure

```
code-finetune/
├── .env                          # API keys (never commit)
├── .env.example                  # Safe template to commit
├── .gitignore
├── requirements.txt
├── run.py                        #  Run entire pipeline at once
├── README.md
├── configs/
│   └── qlora_config.yaml         # All hyperparameters
├── data/
│   ├── processed/
│   │   ├── train.json            # 19k samples
│   │   └── val.json              # 1k samples
│   └── eval_results.json         # Evaluation output
├── src/
│   ├── prepare_data.py           # Data formatting
│   ├── train.py                  # QLoRA training
│   ├── inference.py              # Interactive CLI chat
│   └── evaluate.py               # BLEU + CodeBLEU + Pass@1
├── outputs/
│   └── mistral-qlora/
│       └── final-adapter/        # Saved LoRA weights
└── notebooks/
    └── explore.ipynb             # Data exploration
```

---

##  Setup

### Requirements
- Python 3.11
- CUDA-compatible GPU (6GB+ VRAM recommended)
- Windows / Linux / Mac

### Installation

```bash
# 1. Clone & enter
git clone https://github.com/yourusername/code-finetune
cd code-finetune

# 2. Create virtual env (Python 3.11 required)
py -3.11 -m venv venv
venv\Scripts\activate        # Windows
source venv/bin/activate     # Mac/Linux

# 3. Install dependencies
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
pip install -r requirements.txt

# 4. Add your keys
cp .env.example .env
# Fill in WANDB_API_KEY in .env
```

---

## Run Entire Pipeline at Once

```bash
python run.py
```

This single command runs all steps in order:
1. ✅ Prepares data
2. ✅ Trains the model
3. ✅ Evaluates results
4. ✅ Launches inference chat

Or run steps individually:

```bash
python run.py --step data        # Only prepare data
python run.py --step train       # Only train
python run.py --step evaluate    # Only evaluate
python run.py --step inference   # Only chat
```

---

## Usage (Manual)

### Step 1 — Prepare Data
```bash
python src/prepare_data.py
```

### Step 2 — Train
```bash
python src/train.py
```

### Step 3 — Chat with your model
```bash
python src/inference.py
```

### Step 4 — Evaluate
```bash
python src/evaluate.py
```

---

##  Results

| Metric | Score |
|--------|-------|
| BLEU | 38.4% |
| CodeBLEU | 44.7% |
| Pass@1 | 61.0% |

---

## Key Concepts

| Concept | What it does |
|---------|-------------|
| **QLoRA** | 4-bit quantization cuts VRAM from 28GB → 6GB |
| **LoRA** | Trains only 0.1% of params via low-rank adapters |
| **SFTTrainer** | Supervised fine-tuning with instruction format |
| **CodeBLEU** | Code-aware metric (syntax + data flow aware) |
| **Pass@1** | Checks if generated code actually executes |

---

## Tech Stack

| Tool | Purpose |
|------|---------|
| `transformers` | Model loading & training |
| `peft` | LoRA / QLoRA adapters |
| `bitsandbytes` | 4-bit quantization |
| `datasets` | Data loading (HuggingFace) |
| `trl` | SFTTrainer for instruction tuning |
| `wandb` | Experiment tracking |
| `evaluate` | BLEU metric |
| `codebleu` | Code-specific metric |
| `python-dotenv` | Secure API key management |

---

##  Configuration

All hyperparameters live in `configs/qlora_config.yaml`:

```yaml
model_name: "mistralai/Mistral-7B-v0.1"

lora_config:
  r: 64                  # LoRA rank
  lora_alpha: 16
  lora_dropout: 0.1

training:
  num_train_epochs: 3
  per_device_train_batch_size: 4
  learning_rate: 2e-4
  max_seq_length: 512
```

---

##  Ways to Exten
- Swap dataset → fine-tune on your **own codebase**
- Try **DPO** (Direct Preference Optimization) for better output quality
- Build a **Gradio UI** on top of `inference.py`
- Deploy adapter to **HuggingFace Spaces**
- Add **multi-language support** (JS, Java, C++)

---

##  License

MIT