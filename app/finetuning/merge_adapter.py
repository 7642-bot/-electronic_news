import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel
from app.config.settings import BASE_MODEL, PATHS

def merge():
    adapter_path = PATHS["models"] / "electronics-lora" / "final"
    out_path = PATHS["models"] / "electronics-merged"

    tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL)
    base = AutoModelForCausalLM.from_pretrained(
        BASE_MODEL, torch_dtype=torch.bfloat16, device_map="auto"
    )
    merged = PeftModel.from_pretrained(base, str(adapter_path))
    merged = merged.merge_and_unload()
    merged.save_pretrained(str(out_path))
    tokenizer.save_pretrained(str(out_path))
    print(f"✓ Merged model saved to {out_path}")
