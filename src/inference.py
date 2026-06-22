import os
import torch
from dotenv import load_dotenv
from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline
from peft import PeftModel

load_dotenv()


BASE_MODEL    = "mistralai/Mistral-7B-v0.1"
ADAPTER_PATH  = "outputs/codellama-qlora/final-adapter"
MAX_NEW_TOKENS = 512

def load_finetuned_model():
    print(" Loading base model...")
    tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL)
    tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        BASE_MODEL,
        torch_dtype=torch.float16,
        device_map="auto",
    )

    print(" Loading LoRA adapter...")
    model = PeftModel.from_pretrained(model, ADAPTER_PATH)
    model.eval()   # inference mode

    print(" Model ready!\n")
    return model, tokenizer

# ── Format prompt (same format as training) ───────────────────────────────────
def build_prompt(instruction, input_text=""):
    if input_text:
        return f"""### Instruction:
{instruction}

### Input:
{input_text}

### Response:
"""
    return f"""### Instruction:
{instruction}

### Response:
"""

# ── Generate response ─────────────────────────────────────────────────────────
def generate(model, tokenizer, instruction, input_text=""):
    prompt = build_prompt(instruction, input_text)

    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)

    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=MAX_NEW_TOKENS,
            temperature=0.2,        # low = more deterministic (good for code)
            top_p=0.95,
            do_sample=True,
            repetition_penalty=1.1, # prevents repetitive output
            eos_token_id=tokenizer.eos_token_id,
            pad_token_id=tokenizer.eos_token_id,
        )

    # decode only the newly generated tokens (not the prompt)
    response = tokenizer.decode(
        outputs[0][inputs["input_ids"].shape[1]:],
        skip_special_tokens=True
    )
    return response.strip()

# ── Interactive CLI chat loop ─────────────────────────────────────────────────
def chat():
    model, tokenizer = load_finetuned_model()

    print(" Code Assistant Ready — type 'exit' to quit\n")
    print("─" * 50)

    while True:
        instruction = input("\n Instruction: ").strip()
        if instruction.lower() == "exit":
            print(" Bye!")
            break

        input_text = input(" Input (optional, press Enter to skip): ").strip()

        print("\n  Generating...\n")
        response = generate(model, tokenizer, instruction, input_text)

        print("─" * 50)
        print(f" Response:\n\n{response}")
        print("─" * 50)

if __name__ == "__main__":
    chat()