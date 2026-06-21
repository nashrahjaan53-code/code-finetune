from datasets import load_dataset
import json
import os


def load_raw_data():
    print(" Loading CodeAlpaca-20k dataset...")
    dataset = load_dataset("sahil2801/CodeAlpaca-20k", split="train")
    print(f" Loaded {len(dataset)} samples")
    return dataset


def format_prompt(sample):
  
    if sample["input"]:
        prompt = f"""### Instruction:
{sample["instruction"]}

### Input:
{sample["input"]}

### Response:
{sample["output"]}"""
    else:
        prompt = f"""### Instruction:
{sample["instruction"]}

### Response:
{sample["output"]}"""
    
    return {"text": prompt}

def prepare_and_save(output_dir="data/processed"):
    os.makedirs(output_dir, exist_ok=True)
    
    dataset = load_raw_data()
    
  
    print(" Formatting prompts...")
    formatted = dataset.map(format_prompt, remove_columns=dataset.column_names)
    
    # Train / Val split (95/5)
    split = formatted.train_test_split(test_size=0.05, seed=42)
    train_ds = split["train"]
    val_ds   = split["test"]
    
    print(f"📊 Train: {len(train_ds)} | Val: {len(val_ds)}")
    
    # Save as JSON
    train_ds.to_json(f"{output_dir}/train.json")
    val_ds.to_json(f"{output_dir}/val.json")
    print(f"💾 Saved to {output_dir}/")
    
    # Preview 1 sample
    print("\n── Sample Preview ──────────────────────────")
    print(train_ds[0]["text"])
    print("────────────────────────────────────────────")
    
    return train_ds, val_ds

if __name__ == "__main__":
    prepare_and_save()