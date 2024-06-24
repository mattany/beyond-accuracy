import json

mini_batch_size = 10
requests_per_file = 200
question_batch_id = 0
question_id = 0


system_prompt = """You are a science educator tasked with writing a high-quality scientific answers to the following questions, given these criteria:
1. The explanation has a structured flow from simple to complex concepts.
2. Establish clear connections between various parts of the explanation.
3. Achieve a good balance between introduction, scientific content, examples, and conclusion.
4. Assume the reader has minimal prior knowledge.
5. Use examples.
6. Avoid jargon.
7. Ensure the language is unambiguous, concise, and with clearly defined terminology.
8. Use of paragraphs will be preferred over bullet points and lists.

Your answers should be no more than 256 tokens long.
Your response should be a json with key value pairs in the form:
question_number:answer
"""


def get_question_content(batch):
    global question_id
    retval = ""
    for question in batch:
        retval += f"Question #{question_id}: {question}"
        question_id += 1
    return retval


with open("data/GPT_Questions.csv", "r") as f:
    lines = [_ for _ in f.readlines()]
    for index in range(0, len(lines), mini_batch_size):
        batch_index = ((index // mini_batch_size) + 1) // 200
        with open(f"batch_file_{batch_index}.jsonl", "a") as out_f:
            output_dict = {"custom_id": f"question-batch-{question_batch_id}", "method": "POST",
                           "url": "/v1/chat/completions",
                           "body": {"model": "gpt-4o",
                                    "response_format": {"type": "json_object"},
                                    "messages": [{"role": "system", "content": system_prompt
                                                  },
                                                 {"role": "user",
                                                  "content": get_question_content(lines[index:index + mini_batch_size])}],
                                    "max_tokens": 3000}}
            out_f.write(json.dumps(output_dict) + "\n")
            question_batch_id += 1
