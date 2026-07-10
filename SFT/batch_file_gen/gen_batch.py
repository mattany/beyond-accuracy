import json
import os
import shutil

import pandas as pd

from SFT.batch_file_gen.constants import INPUT_CSV, GPT_INPUT_BATCH_DIR, GPT_INPUT_BATCH_PREFIX, SFT_SYSTEM_PROMPT
# Parameters
mini_batch_size = 1   # Only 1 is supported currently
requests_per_file = 200
question_id = 0

# EXPLANATION_QUALITY_SYSTEM_PROMPT = """
# """

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





def delete_all_files_in_dir(directory):
    if os.path.exists(directory) and os.path.isdir(directory):
        shutil.rmtree(directory)
    os.mkdir(directory)


def create_input_batch_files(input_questions, output_dir, prefix, system_prompt, model="gpt-5-2025-08-07"):
    delete_all_files_in_dir(output_dir)
    question_batch_id = 0
    for index in range(0, len(input_questions), mini_batch_size):
        batch_index = ((index // mini_batch_size) + 1) // requests_per_file
        content = get_question_content(input_questions[index:index + mini_batch_size])

        # Write to JSONL file
        with open(f"{output_dir}/{prefix}{batch_index}.jsonl", "a") as out_f:
            output_dict = {"custom_id": f"question-batch-{question_batch_id}", "method": "POST",
                           "url": "/v1/chat/completions",
                           "body": {"model": model,
                                    "messages": [{"role": "system", "content": system_prompt},
                                                 {"role": "user", "content": content}],
                                    "reasoning_effort": "medium",
                                    "max_completion_tokens": 2048}}
            out_f.write(json.dumps(output_dict) + "\n")
            question_batch_id += 1


if __name__ == "__main__":
    # Read the questions from the CSV file
    input_questions = pd.read_csv(INPUT_CSV, sep="\t")["Question"].tolist()
    create_input_batch_files(input_questions=input_questions, output_dir=GPT_INPUT_BATCH_DIR, prefix=GPT_INPUT_BATCH_PREFIX, system_prompt=SFT_SYSTEM_PROMPT)


