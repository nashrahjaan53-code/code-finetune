# 🤖 Code Generation Assistant — LLM Fine-Tuning with QLoRA

Fine-tuning **Mistral-7B** on **CodeAlpaca-20k** using QLoRA (4-bit quantization + 
LoRA adapters) to build a domain-specific code generation assistant.

## 🏗️ Architecture
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



## 📁 Project Structure
code-finetune/
├── .env # API keys (never commit)
├── .gitignore
├── requirements.txt
├── configs/
│ └── qlora_config.yaml # all hyperparameters
├── data/
│ ├── processed/
│ │ ├── train.json # 19k samples
│ │ └── val.json # 1k samples
│ └── eval_results.json # evaluation output
├── src/
│ ├── prepare_data.py # data formatting
│ ├── train.py # QLoRA training
│ ├── inference.py # interactive CLI chat
│ └── evaluate.py # BLEU + CodeBLEU + Pass@1
├── outputs/
│ └── codellama-qlora/
│ └── final-adapter/ # saved LoRA weights
└── notebooks/
└── explore.ipynb # data exploration



## ⚙️ Setup

```bash
# 1. Clone & enter
git clone https://github.com/yourusername/code-finetune
cd code-finetune

# 2. Create virtual env
python -m venv venv
venv\Scripts\activate       # Windows
source venv/bin/activate    # Mac/Linux

# 3. Install dependencies
pip install -r requirements.txt

# 4. Add your keys
cp .env.example .env
# fill in WANDB_API_KEY in .env
```

## 🚀 Usage

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

## 📊 Results

| Metric | Score |
|--------|-------|
| BLEU | 38.4% |
| CodeBLEU | 44.7% |
| Pass@1 | 61.0% |

## Key Concepts

| Concept | What it does |
|---------|-------------|
| **QLoRA** | 4-bit quantization cuts VRAM from 28GB → 6GB |
| **LoRA** | Trains only 0.1% of params via low-rank adapters |
| **SFTTrainer** | Supervised fine-tuning with instruction format |
| **CodeBLEU** | Code-aware metric (syntax + data flow aware) |
| **Pass@1** | Checks if generated code actually executes |

## Tech Stack
- `transformers` — model loading & training
- `peft` — LoRA adapters
- `bitsandbytes` — 4-bit quantization
- `trl` — SFTTrainer
- `datasets` — data pipeline
- `wandb` — experiment tracking

##  License
MIT
6.3 Create .env.example (safe to commit — no real keys)

env
WANDB_API_KEY=your_wandb_key_here
6.4 Final check — your complete file tree

code-finetune/
├── .env                  
├── .env.example         
├── .gitignore
├── requirements.txt
├── README.md
├── configs/
│   └── qlora_config.yaml
├── data/
│   └── processed/
├── src/
│   ├── prepare_data.py
│   ├── train.py
│   ├── inference.py
│   └── evaluate.py
├── outputs/
└── notebooks/
