import json

from SFT.batch_file_gen.gen_batch import create_input_batch_files
from SFT.batch_file_gen.upload_batch_file import run as get_results_from_gpt
import pandas as pd
from config import PROJECT_DIR
# GPT_BATCH_DIR = f"{PROJECT_DIR}/Benchmarking/eval_dataset_generation/output_batches/gpt_4o/"
# GPT_OUTPUT_DIR = "/Users/mattan.yeroushalmi/studies/thesis/Benchmarking/eval_dataset_generation/gpt4o_outputs"
# MODEL = "gpt-4o-2024-08-06"
# MODEL_NAME = "gpt_4o"

GPT_BATCH_DIR = f"{PROJECT_DIR}/Benchmarking/eval_dataset_generation/output_batches/gpt_4/"
GPT_OUTPUT_DIR = "/Users/mattan.yeroushalmi/studies/thesis/Benchmarking/eval_dataset_generation/gpt4_outputs"
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




if __name__ == "__main__":
    eval_dataset = "/Users/mattan.yeroushalmi/studies/thesis/Benchmarking/deep_eval/data/evaluation_dataset.csv"
    questions = pd.read_csv(eval_dataset)["question"].tolist()
    create_input_batch_files(questions, output_dir=GPT_BATCH_DIR, prefix="eval_dataset_", system_prompt="", model=MODEL)
    get_results_from_gpt(gpt_input_batch_dir=GPT_BATCH_DIR, prefix="eval_dataset_", output_dir=GPT_OUTPUT_DIR)
    add_gpt_column_to_eval_dataset(eval_dataset, f"{GPT_OUTPUT_DIR}/gpt_output_file_0.jsonl", model_name=MODEL_NAME)
