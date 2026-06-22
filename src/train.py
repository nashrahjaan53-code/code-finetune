import os
from dotenv import load_dotenv
import yaml
import torch
import wandb
from datasets import load_dataset
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
    TrainingArguments,
)
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
from trl import SFTTrainer


def load_config(path="configs/qlora_config.yaml"):
    with open(path) as f:
        return yaml.safe_load(f)

def load_tokenizer(model_name):
    print(f" Loading tokenizer: {model_name}")
    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True, use_fast=False)
    tokenizer.pad_token = tokenizer.eos_token   # CodeLlama has no pad token
    tokenizer.padding_side = "right"
    return tokenizer

def load_model(model_name, bnb_cfg, use_quantization=True):
    if use_quantization:
        print(f" Loading model in 4-bit: {model_name}")
        bnb_config = BitsAndBytesConfig(
            load_in_4bit=bnb_cfg["load_in_4bit"],
            bnb_4bit_quant_type=bnb_cfg["bnb_4bit_quant_type"],
            bnb_4bit_compute_dtype=torch.float16,
            bnb_4bit_use_double_quant=bnb_cfg["bnb_4bit_use_double_quant"],
        )
        model = AutoModelForCausalLM.from_pretrained(
            model_name,
            quantization_config=bnb_config,
            device_map="auto",               # auto places layers across GPU/CPU
            trust_remote_code=True,
        )
    else:
        print(f" Loading model in standard precision (no quantization): {model_name}")
        model = AutoModelForCausalLM.from_pretrained(
            model_name,
            device_map="auto",               # auto places layers across GPU/CPU
            trust_remote_code=True,
        )
    model.config.use_cache = False       # disable KV cache during training
    model.config.pretraining_tp = 1
    return model


def apply_lora(model, lora_cfg, use_quantization=True):
    print("Applying LoRA adapters...")
    
    if use_quantization:
        model = prepare_model_for_kbit_training(model)  # prepares 4-bit model for training
    
    lora_config = LoraConfig(
        r=lora_cfg["r"],
        lora_alpha=lora_cfg["lora_alpha"],
        lora_dropout=lora_cfg["lora_dropout"],
        bias=lora_cfg["bias"],
        task_type=lora_cfg["task_type"],
        target_modules=lora_cfg["target_modules"],
    )
    
    model = get_peft_model(model, lora_config)
    model.print_trainable_parameters()  # shows how few params we're actually training!
    return model

def load_data():
    print(" Loading processed dataset...")
    train = load_dataset("json", data_files="data/processed/train.jsonl", split="train")
    val   = load_dataset("json", data_files="data/processed/val.jsonl",   split="train")
    print(f"Train: {len(train)} | Val: {len(val)}")
    return train, val

def train():
    cfg = load_config()
    
    # W&B init
    wandb.init(project="code-finetune", name="codellama-qlora-run1")
    
    model_name  = cfg["model_name"]
    tokenizer   = load_tokenizer(model_name)
    
    has_cuda = torch.cuda.is_available()
    use_quant = cfg["bnb_config"]["load_in_4bit"] and has_cuda
    
    if not has_cuda:
        print("\n[WARNING] NVIDIA GPU/CUDA not detected. Falling back to CPU mode:")
        print("  - Disabling 4-bit bitsandbytes quantization")
        print("  - Disabling float16 precision (using float32)\n")

    model       = load_model(model_name, cfg["bnb_config"], use_quantization=use_quant)
    model       = apply_lora(model, cfg["lora_config"], use_quantization=use_quant)
    train_ds, val_ds = load_data()
    
    t = cfg["training"]
    
    fp16_val = t["fp16"] if has_cuda else False
    
    training_args = TrainingArguments(
        output_dir=t["output_dir"],
        num_train_epochs=t["num_train_epochs"],
        per_device_train_batch_size=t["per_device_train_batch_size"],
        gradient_accumulation_steps=t["gradient_accumulation_steps"],
        learning_rate=t["learning_rate"],
        warmup_ratio=t["warmup_ratio"],
        lr_scheduler_type=t["lr_scheduler_type"],
        fp16=fp16_val,
        use_cpu=not has_cuda,
        logging_steps=t["logging_steps"],
        save_steps=t["save_steps"],
        eval_steps=t["eval_steps"],
        evaluation_strategy="steps",
        save_total_limit=2,              # keep only last 2 checkpoints
        load_best_model_at_end=True,
        report_to=t["report_to"],
    )
    
    trainer = SFTTrainer(
        model=model,
        train_dataset=train_ds,
        eval_dataset=val_ds,
        tokenizer=tokenizer,
        args=training_args,
        dataset_text_field="text",       # column name from our prepare_data.py
        max_seq_length=t["max_seq_length"],
        packing=False,
    )
    
    print("\n Starting training...\n")
    trainer.train()
    
    
    save_path = f"{t['output_dir']}/final-adapter"
    trainer.model.save_pretrained(save_path)
    tokenizer.save_pretrained(save_path)
    print(f"\n Adapter saved to {save_path}")
    
    wandb.finish()

if __name__ == "__main__":
    train()