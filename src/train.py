import os
import yaml
import torch
import wandb
from dotenv import load_dotenv
from datasets import load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer, TrainingArguments
from peft import LoraConfig, get_peft_model
from trl import SFTTrainer

load_dotenv()

def load_config(path="configs/qlora_config.yaml"):
    with open(path) as f:
        return yaml.safe_load(f)

def load_tokenizer(model_name):
    print(f"Loading tokenizer: {model_name}")
    tokenizer = AutoTokenizer.from_pretrained(
        model_name,
        trust_remote_code=True,
        use_fast=False
    )
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"
    return tokenizer

def load_model(model_name):
    print(f"Loading model: {model_name}")
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=torch.float32,
        device_map=None,
        trust_remote_code=True,
        low_cpu_mem_usage=True,
    )
    model.config.use_cache = False
    return model

def apply_lora(model, lora_cfg):
    print("Applying LoRA adapters...")
    lora_config = LoraConfig(
        r=lora_cfg["r"],
        lora_alpha=lora_cfg["lora_alpha"],
        lora_dropout=lora_cfg["lora_dropout"],
        bias=lora_cfg["bias"],
        task_type=lora_cfg["task_type"],
        target_modules=lora_cfg["target_modules"],
    )
    model = get_peft_model(model, lora_config)
    model.print_trainable_parameters()
    return model

def load_data():
    print("Loading processed dataset...")
    train = load_dataset("json", data_files="data/processed/train.jsonl", split="train")
    val   = load_dataset("json", data_files="data/processed/val.jsonl",   split="train")
    print(f"Train: {len(train)} | Val: {len(val)}")
    return train, val

def train():
    cfg = load_config()
    wandb.init(project="code-finetune", name="qwen-lora-cpu-run1")

    model_name = cfg["model_name"]
    tokenizer  = load_tokenizer(model_name)
    model      = load_model(model_name)
    model      = apply_lora(model, cfg["lora_config"])
    train_ds, val_ds = load_data()

    t = cfg["training"]

    training_args = TrainingArguments(
        output_dir=t["output_dir"],
        num_train_epochs=t["num_train_epochs"],
        per_device_train_batch_size=t["per_device_train_batch_size"],
        gradient_accumulation_steps=t["gradient_accumulation_steps"],
        learning_rate=t["learning_rate"],
        warmup_ratio=t["warmup_ratio"],
        lr_scheduler_type=t["lr_scheduler_type"],
        fp16=False,
        logging_steps=t["logging_steps"],
        save_steps=t["save_steps"],
        eval_steps=t["eval_steps"],
        eval_strategy="steps",
        save_total_limit=2,
        load_best_model_at_end=True,
        report_to="wandb",
        use_cpu=True,
    )

    trainer = SFTTrainer(
        model=model,
        train_dataset=train_ds,
        eval_dataset=val_ds,
        tokenizer=tokenizer,
        args=training_args,
        dataset_text_field="text",
        max_seq_length=t["max_seq_length"],
    )

    print("\nStarting training...\n")
    trainer.train()

    save_path = f"{t['output_dir']}/final-adapter"
    trainer.model.save_pretrained(save_path)
    tokenizer.save_pretrained(save_path)
    print(f"\nAdapter saved to {save_path}")
    wandb.finish()

if __name__ == "__main__":
    train()