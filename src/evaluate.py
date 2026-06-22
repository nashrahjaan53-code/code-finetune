import os
import json
import torch
from dotenv import load_dotenv
from datasets import load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel
from evaluate import load
from codebleu import calc_codebleu
import yaml
nltk.download("punkt", quiet=True)

load_dotenv()

def load_config(path="configs/qlora_config.yaml"):
    with open(path) as f:
        return yaml.safe_load(f)

cfg = load_config()
BASE_MODEL   = cfg["model_name"]
ADAPTER_PATH = "outputs/codellama-qlora/final-adapter"
RESULTS_PATH = "data/eval_results.json"
NUM_SAMPLES  = 100          # evaluate on 100 val samples (fast)
MAX_NEW_TOKENS = 256


def load_model():
    print(" Loading fine-tuned model...")
    tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL)
    tokenizer.pad_token = tokenizer.eos_token

    dtype = torch.float16 if torch.cuda.is_available() else torch.float32
    model = AutoModelForCausalLM.from_pretrained(
        BASE_MODEL,
        torch_dtype=dtype,
        device_map="auto",
    )
    model = PeftModel.from_pretrained(model, ADAPTER_PATH)
    model.eval()
    return model, tokenizer


def build_prompt(sample):
    if sample.get("input", ""):
        return f"""### Instruction:\n{sample['instruction']}\n\n### Input:\n{sample['input']}\n\n### Response:\n"""
    return f"""### Instruction:\n{sample['instruction']}\n\n### Response:\n"""

def generate_one(model, tokenizer, prompt):
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=MAX_NEW_TOKENS,
            temperature=0.2,
            top_p=0.95,
            do_sample=True,
            repetition_penalty=1.1,
            pad_token_id=tokenizer.eos_token_id,
        )
    response = tokenizer.decode(
        outputs[0][inputs["input_ids"].shape[1]:],
        skip_special_tokens=True
    )
    return response.strip()


def compute_bleu(predictions, references):
    bleu = load("bleu")
    result = bleu.compute(
        predictions=predictions,
        references=[[ref] for ref in references]
    )
    return round(result["bleu"] * 100, 2)


def compute_codebleu(predictions, references):
    result = calc_codebleu(
        references=[[ref] for ref in references],
        hypotheses=predictions,
        lang="python",
        weights=(0.25, 0.25, 0.25, 0.25),  # ngram, weighted, syntax, dataflow
    )
    return round(result["codebleu"] * 100, 2)


def compute_pass_at_1(predictions):
    passed = 0
    errors = []

    for i, code in enumerate(predictions):
        try:
            exec(compile(code, "<string>", "exec"), {})  # sandboxed exec
            passed += 1
        except Exception as e:
            errors.append({"sample": i, "error": str(e)})

    pass_at_1 = round((passed / len(predictions)) * 100, 2)
    return pass_at_1, errors

def evaluate():
    # load val set
    val_ds = load_dataset(
        "json",
        data_files="data/processed/val.json",
        split="train"
    ).select(range(NUM_SAMPLES))

    model, tokenizer = load_model()

    predictions = []
    references  = []

    print(f"\n Evaluating on {NUM_SAMPLES} samples...\n")

    for i, sample in enumerate(val_ds):
        # rebuild original sample structure from formatted text
        # val.json has "text" field — we need to split out the response
        text = sample["text"]
        ref  = text.split("### Response:")[-1].strip()

        prompt = "### Response:".join(text.split("### Response:")[:-1]) + "### Response:\n"

        pred = generate_one(model, tokenizer, prompt)

        predictions.append(pred)
        references.append(ref)

        if (i + 1) % 10 == 0:
            print(f"   {i+1}/{NUM_SAMPLES} done...")

    print("\n Computing metrics...")

    bleu      = compute_bleu(predictions, references)
    codebleu  = compute_codebleu(predictions, references)
    pass_at_1, errors = compute_pass_at_1(predictions)

    results = {
        "num_samples" : NUM_SAMPLES,
        "bleu"        : bleu,
        "codebleu"    : codebleu,
        "pass_at_1"   : pass_at_1,
        "num_errors"  : len(errors),
        "error_samples": errors[:5],   # save first 5 errors for inspection
    }


    print("\n" + "─" * 40)
    print("📈 EVALUATION RESULTS")
    print("─" * 40)
    print(f"  BLEU       : {bleu}%")
    print(f"  CodeBLEU   : {codebleu}%")
    print(f"  Pass@1     : {pass_at_1}%")
    print(f"  Errors     : {len(errors)}/{NUM_SAMPLES}")
    print("─" * 40)


    os.makedirs("data", exist_ok=True)
    with open(RESULTS_PATH, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\n Results saved to {RESULTS_PATH}")

if __name__ == "__main__":
    evaluate()