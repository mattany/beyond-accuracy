import json

# Parameters
mini_batch_size = 1
requests_per_file = 200
question_batch_id = 0
question_id = 0

# System prompt for generating high-quality scientific answers
system_prompt = """
You are tasked with writing high-quality scientific answers , given these criteria:
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
with open("../data/GPT_Questions.csv", "r", encoding="utf-8") as f:
    lines = [remove_quotes(line.strip()) for line in f.readlines()][1:]

    # Process lines in batches
    for index in range(0, len(lines), mini_batch_size):
        batch_index = ((index // mini_batch_size) + 1) // 400
        content = get_question_content(lines[index:index + mini_batch_size])

        # Write to JSONL file
        with open(f"batch_file_{batch_index}.jsonl", "a") as out_f:
            output_dict = {"custom_id": f"question-batch-{question_batch_id}", "method": "POST",
                           "url": "/v1/chat/completions",
                           "body": {"model": "gpt-3.5-turbo",
                                    "messages": [{"role": "system", "content": system_prompt},
                                                 {"role": "user","content": get_question_content(lines[index:index + mini_batch_size])}],
                                    "max_tokens": 256}}
            out_f.write(json.dumps(output_dict) + "\n")
            question_batch_id += 1


