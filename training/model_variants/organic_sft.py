"""Generate evaluation answers with the retained Human-SFT model variant.

This is the executable Python form of the original Colab artifact:
https://colab.research.google.com/drive/1_sYa6PfujNiEYdeRU2ILXj9DrvYG8km9
"""

from argparse import ArgumentParser
import os
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
DEFAULT_INPUT = ROOT / "data" / "qa_pairs" / "ask_science_test.csv"
DEFAULT_OUTPUT = ROOT / "scripts" / "generations" / "organic_sft.csv"
SYSTEM_MESSAGE = "You are a helpful assistant."


def parse_args(argv=None):
    parser = ArgumentParser(description=__doc__)
    parser.add_argument("--adapter", required=True, help="LoRA adapter path or Hub ID")
    parser.add_argument(
        "--base-model",
        default="meta-llama/Llama-3.1-8B-Instruct",
    )
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--max-new-tokens", type=int, default=1024)
    parser.add_argument("--checkpoint-every", type=int, default=5)
    return parser.parse_args(argv)


def generate_answers(args):
    import pandas as pd
    from peft import PeftModel
    from transformers import AutoModelForCausalLM, AutoTokenizer

    os.environ.setdefault("HF_HOME", str(ROOT / ".cache" / "huggingface"))
    tokenizer = AutoTokenizer.from_pretrained(args.base_model, use_fast=True)
    model = AutoModelForCausalLM.from_pretrained(
        args.base_model,
        device_map="auto",
        torch_dtype="auto",
    )
    model = PeftModel.from_pretrained(model, args.adapter)
    model.eval()

    source = pd.read_csv(args.input)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    if args.output.exists():
        partial = pd.read_csv(args.output)
        start_index = len(partial)
        data = {
            "Question": partial["Question"].tolist(),
            "Answer": partial["Answer"].tolist(),
        }
        print(f"Resuming from index {start_index}")
    else:
        start_index = 0
        data = {"Question": [], "Answer": []}

    for index in range(start_index, len(source)):
        question = source.iloc[index]["Question"]
        prompt = f"<|system|>{SYSTEM_MESSAGE}\n<|user|>{question}\n<|assistant|>"
        inputs = tokenizer(prompt, return_tensors="pt", truncation=True).to(model.device)
        generated = model.generate(
            **inputs,
            max_new_tokens=args.max_new_tokens,
            do_sample=False,
            pad_token_id=tokenizer.pad_token_id,
        )
        decoded = tokenizer.decode(generated[0], skip_special_tokens=True)
        answer = decoded[len(prompt) :].strip() if decoded.startswith(prompt) else decoded.strip()
        data["Question"].append(question)
        data["Answer"].append(answer)
        if args.checkpoint_every and (index + 1) % args.checkpoint_every == 0:
            pd.DataFrame(data).to_csv(args.output, index=False)

    pd.DataFrame(data).to_csv(args.output, index=False)


if __name__ == "__main__":
    generate_answers(parse_args())
