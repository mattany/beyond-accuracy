import csv
import json
import os

from SFT.batch_file_gen.constants import INPUT_CSV, GPT_OUTPUT_DIR, GPT_OUTPUT_FILE_PREFIX, OUTPUT_CSV


def read_questions(csv_path):
    """
    Reads questions from a CSV file.

    Args:
    csv_path (str): Path to the CSV file containing questions.

    Returns:
    dict: Dictionary where keys are question IDs and values are questions.
    """
    questions = {}
    with open(csv_path, mode='r', encoding='utf-8') as file:
        reader = csv.DictReader(file, delimiter='\t')
        question_id = 0
        for row in reader:
            # Adjusting the index since question number in CSV = id + 1
            questions[question_id] = row['Question']
            question_id += 1
    return questions

def read_answers():
    """
    Reads answers from JSONL files.

    Args:
    n_file_paths (int): Number of JSONL files to read from.

    Returns:
    dict: Dictionary where keys are question IDs and values are answers.
    """
    gpt_answers = {}
    batch_amt = len([f for f in os.listdir(GPT_OUTPUT_DIR) if os.path.isfile(os.path.join(GPT_OUTPUT_DIR, f))])
    gpt_output_paths = [f"{GPT_OUTPUT_DIR}/{GPT_OUTPUT_FILE_PREFIX}{batch_index}.jsonl" for batch_index in range(batch_amt)]

    for path in gpt_output_paths:
        with open(path, mode='r', encoding='utf-8') as file:
            for line in file:
                data = json.loads(line)
                custom_id = data['custom_id']
                question_id = int(custom_id.split('-')[-1])
                answer = data['response']['body']['choices'][0]['message']['content']
                truncated = 1 if data['response']['body']['choices'][0]['finish_reason'] != 'stop' else 0
                gpt_answers[question_id] = answer, truncated
    return gpt_answers

def write_answers_to_csv(questions, answers):
    """
    Writes questions and their corresponding answers to a CSV file.

    Args:
    questions (dict): Dictionary of questions.
    answers (dict): Dictionary of answers.
    output_csv_path (str): Path to the output CSV file.
    """
    with open(OUTPUT_CSV, mode='w', newline='', encoding='utf-8') as file:
        writer = csv.writer(file)
        writer.writerow(['Index', 'Question', 'Answer', 'Truncated'])
        for i, (question_id, question) in enumerate(questions.items()):
            answer, truncated = answers.get(question_id, "No answer available")
            writer.writerow([i, question, answer, truncated])


if __name__ == "__main__":
    # Read questions and answers
    questions = read_questions(INPUT_CSV)
    answers = read_answers()

    # Write questions and answers to a new CSV file
    write_answers_to_csv(questions, answers)

    print(f"Answers written to {OUTPUT_CSV}")
