import os

import fire
import torch
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer


def merge(
    base_model_path: str,
    lora_adapter_path: str,
    output_path: str,
    sft_lora_path: str = "",
    torch_dtype: str = "bfloat16",
):
    dtype = getattr(torch, torch_dtype)

    model = AutoModelForCausalLM.from_pretrained(base_model_path, torch_dtype=dtype, device_map="auto")

    tokenizer_source = lora_adapter_path if os.path.exists(lora_adapter_path) else base_model_path
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_source, trust_remote_code=True)

    if model.get_input_embeddings().weight.shape[0] != len(tokenizer):
        model.resize_token_embeddings(len(tokenizer))

    if sft_lora_path:
        model = PeftModel.from_pretrained(model, sft_lora_path)
        model = model.merge_and_unload()

    model = PeftModel.from_pretrained(model, lora_adapter_path)
    model = model.merge_and_unload()

    os.makedirs(output_path, exist_ok=True)
    model.save_pretrained(output_path)
    tokenizer.save_pretrained(output_path)


if __name__ == "__main__":
    fire.Fire(merge)
