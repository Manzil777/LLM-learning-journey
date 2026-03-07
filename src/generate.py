
from transformers import AutoTokenizer, AutoModelForCausalLM
import torch
import argparse
import os
from datetime import datetime

def load_model(model_name="distilgpt2"):
    print(f"Loading model: {model_name}")
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_pretrained(model_name)
    model.eval()
    return tokenizer, model

def generate(prompt, tokenizer, model, temperature=0.7, top_k=50, top_p=0.9, max_new_tokens=100):
    inputs = tokenizer(prompt, return_tensors="pt")
    with torch.no_grad():
        outputs = model.generate(
            inputs["input_ids"],
            attention_mask=inputs["attention_mask"],
            pad_token_id=tokenizer.eos_token_id,
            max_new_tokens=max_new_tokens,
            do_sample=True,
            temperature=temperature,
            top_k=top_k,
            top_p=top_p
        )
    return tokenizer.decode(outputs[0], skip_special_tokens=True)

def save_output(prompt, output, params, filepath="outputs/generation_examples.txt"):
    os.makedirs("outputs", exist_ok=True)
    with open(filepath, "a") as f:
        f.write(f"\n{'='*50}\n")
        f.write(f"Timestamp : {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write(f"Prompt    : {prompt}\n")
        f.write(f"Params    : {params}\n")
        f.write(f"Output    : {output}\n")

def main():
    parser = argparse.ArgumentParser(description="LLM Text Generation Script")
    parser.add_argument("--prompt", type=str, required=True)
    parser.add_argument("--temperature", type=float, default=0.7)
    parser.add_argument("--top_k", type=int, default=50)
    parser.add_argument("--top_p", type=float, default=0.9)
    parser.add_argument("--max_new_tokens", type=int, default=100)
    parser.add_argument("--model", type=str, default="distilgpt2")
    args = parser.parse_args()

    tokenizer, model = load_model(args.model)

    params = {
        "temperature": args.temperature,
        "top_k": args.top_k,
        "top_p": args.top_p,
        "max_new_tokens": args.max_new_tokens
    }

    print(f"\nPrompt: {args.prompt}")
    output = generate(args.prompt, tokenizer, model, **params)
    print(f"\nGenerated:\n{output}")

    save_output(args.prompt, output, params)
    print(f"\nSaved to outputs/generation_examples.txt")

if __name__ == "__main__":
    main()
