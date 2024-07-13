import json
import os
import shutil

import pandas as pd

from SFT.batch_file_gen.config import PROJECT_DIR

# Parameters
mini_batch_size = 1
requests_per_file = 200
question_id = 0

INPUT_CSV = f"{PROJECT_DIR}/SFT/data/ask_science.csv"
OUTPUT_DIR = f"{PROJECT_DIR}/SFT/data/input_batches"
SFT_INPUT_BATCH_PREFIX = "sft_input_batch_file_"
# System prompt for generating high-quality scientific answers
SFT_SYSTEM_PROMPT = """You are tasked with writing high-quality scientific answers , given these criteria:
1. The explanation has a structured flow from simple to complex concepts.
2. Establish clear connections between various parts of the explanation.
3. Achieve a good balance between introduction, scientific content, examples, and conclusion.
4. Assume the reader has minimal prior knowledge.
5. Use examples.
6. Avoid jargon.
7. Ensure the language is unambiguous, concise, and with clearly defined terminology.
8. Use of paragraphs will be preferred over bullet points and lists. 

The answers should be short.
"""
EXPLANATION_QUALITY_SYSTEM_PROMPT = """
"""

def get_question_content(batch):
    """
    Concatenates the questions in the given batch into a single string.

    Args:
    batch (list): List of questions in the current batch.

    Returns:
    str: Concatenated string of questions.
    """
    global question_id
    retval = ""
    for question in batch:
        retval += f"{question}"
        question_id += 1
    return retval

def remove_quotes(line):
    """
    Removes quotes from the beginning and end of the line if present.

    Args:
    line (str): The input string from which quotes need to be removed.

    Returns:
    str: String without the leading and trailing quotes.
    """
    if line.startswith('"') and line.endswith('"'):
        return line[1:-1]
    return line


# Read the questions from the CSV file
input_questions = pd.read_csv(INPUT_CSV, sep="\t")["title"].tolist()


def delete_all_files_in_dir(directory):
    if os.path.exists(directory) and os.path.isdir(directory):
        shutil.rmtree(directory)
    os.mkdir(directory)


def create_input_batch_files(output_dir, prefix, system_prompt):
    delete_all_files_in_dir(output_dir)
    question_batch_id = 0
    for index in range(0, len(input_questions), mini_batch_size):
        batch_index = ((index // mini_batch_size) + 1) // requests_per_file
        content = get_question_content(input_questions[index:index + mini_batch_size])

        # Write to JSONL file
        with open(f"{output_dir}/{prefix}{batch_index}.jsonl", "a") as out_f:
            output_dict = {"custom_id": f"question-batch-{question_batch_id}", "method": "POST",
                           "url": "/v1/chat/completions",
                           "body": {"model": "gpt-3.5-turbo",
                                    "messages": [{"role": "system", "content": system_prompt},
                                                 {"role": "user", "content": content}],
                                    "max_tokens": 256}}
            out_f.write(json.dumps(output_dict) + "\n")
            question_batch_id += 1


if __name__ == "__main__":
    create_input_batch_files(output_dir=OUTPUT_DIR, prefix=SFT_INPUT_BATCH_PREFIX, system_prompt=SFT_SYSTEM_PROMPT)


