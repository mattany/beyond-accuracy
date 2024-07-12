import csv
import json

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
        reader = csv.DictReader(file)
        question_id = 0
        for row in reader:
            # Adjusting the index since question number in CSV = id + 1
            questions[question_id] = row['Question']
            question_id += 1
    return questions

def read_answers(n_file_paths):
    """
    Reads answers from JSONL files.

    Args:
    n_file_paths (int): Number of JSONL files to read from.

    Returns:
    dict: Dictionary where keys are question IDs and values are answers.
    """
    answers = {}
    for i in range(n_file_paths):
        path = f'outputs\output_batch_file_{i}.jsonl'
        with open(path, mode='r', encoding='utf-8') as file:
            for line in file:
                data = json.loads(line)
                custom_id = data['custom_id']
                question_id = int(custom_id.split('-')[-1])
                answer = data['response']['body']['choices'][0]['message']['content']
                answers[question_id] = answer
    return answers

def write_answers_to_csv(questions, answers, output_csv_path):
    """
    Writes questions and their corresponding answers to a CSV file.

    Args:
    questions (dict): Dictionary of questions.
    answers (dict): Dictionary of answers.
    output_csv_path (str): Path to the output CSV file.
    """
    with open(output_csv_path, mode='w', newline='', encoding='utf-8') as file:
        writer = csv.writer(file)
        writer.writerow(['Question', 'Answer'])
        for question_id, question in questions.items():
            answer = answers.get(question_id, "No answer available")
            writer.writerow([question, answer])


if __name__ == "__main__":
    # Paths to your files
    questions_csv_path = 'GPT_Questions.csv'
    output_csv_path = 'GPT_Answers.csv'

    # Read questions and answers
    questions = read_questions(questions_csv_path)
    answers = read_answers(17)

    # Write questions and answers to a new CSV file
    write_answers_to_csv(questions, answers, output_csv_path)

    print(f"Answers written to {output_csv_path}")
