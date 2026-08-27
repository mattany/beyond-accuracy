from argparse import ArgumentParser
import json
from pathlib import Path

from training.data_generation.gen_batch import create_input_batch_files
from training.data_generation.upload_batch_file import run as get_results_from_gpt
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
GPT_BATCH_DIR = ROOT / "evaluation" / "model_generation" / "output_batches" / "gpt_4"
GPT_OUTPUT_DIR = ROOT / "evaluation" / "model_generation" / "gpt4_outputs"
MODEL = "gpt-4-turbo-2024-04-09"
MODEL_NAME = "gpt_4"

def add_gpt_column_to_eval_dataset(eval_dataset, gpt_outputs_jsonl, model_name="gpt_4"):
    answers = []
    with open(gpt_outputs_jsonl, mode='r', encoding='utf-8') as file:
        for line in file:
            data = json.loads(line)
            custom_id = data['custom_id']
            question_id = int(custom_id.split('-')[-1])
            answer = data['response']['body']['choices'][0]['message']['content']
            answers.append(answer)
    df = pd.read_csv(eval_dataset)
    df[f"{model_name}_outputs"] = pd.Series(answers)
    df.to_csv(eval_dataset, index=False)




def parse_args():
    parser = ArgumentParser()
    parser.add_argument(
        "--input",
        type=Path,
        default=ROOT / "evaluation" / "model_outputs" / "main" / "all_models_joined.csv",
    )
    parser.add_argument("--output-dir", type=Path, default=GPT_OUTPUT_DIR)
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    eval_dataset = args.input
    args.output_dir.mkdir(parents=True, exist_ok=True)
    questions = pd.read_csv(eval_dataset)["question"].tolist()
    create_input_batch_files(questions, output_dir=GPT_BATCH_DIR, prefix="eval_dataset_", system_prompt="", model=MODEL)
    get_results_from_gpt(gpt_input_batch_dir=GPT_BATCH_DIR, prefix="eval_dataset_", output_dir=args.output_dir)
    add_gpt_column_to_eval_dataset(eval_dataset, args.output_dir / "gpt_output_file_0.jsonl", model_name=MODEL_NAME)
