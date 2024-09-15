
import pandas as pd
import logging
from Jargon.jargon_util import calculate_grade

logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger("main_logger")

# import random
# Generate a sample of 100 random integers from the range [0, 9312]
# random.seed(3)
# random_sample = random.sample(range(0, 9313), 100)
# print(random_sample)
# random_sample = [3898, 8916, 2136, 6061, 7766, 1073, 215, 7687, 4249, 9024, 3839, 3141, 7704, 8863, 9005, 7804, 6506, 2467, 3799, 2484, 8571, 6388, 248, 1049, 2611, 701, 4935, 508, 4414, 7745, 6350, 6994, 6471, 7284, 2197, 5988, 1596, 587, 2227, 8108, 3555, 4226, 7146, 4932, 6900, 8310, 6322, 5749, 8750, 6677, 3807, 5517, 469, 4582, 2672, 5347, 8876, 1705, 3459, 4375, 4668, 2038, 1039, 7897, 7921, 1450, 5637, 1091, 6725, 2470, 329, 4815, 6998, 6802, 1948, 724, 736, 6189, 5422, 9025, 4572, 8280, 3865, 590, 5073, 118, 1261, 1771, 8774, 514, 3233, 6683, 4777, 4315, 2559, 695, 5567, 5141, 5901, 2266]
random_sample = [3898, 8916, 2136, 6061, 7766]


def answer_generator_from_csv(csv_path, indexes):
    df = pd.read_csv(csv_path)
    df = df.where(df["index"].isin(indexes)).dropna()
    return df


def add_answer_grades(answers_df: pd.DataFrame):
    answers_df["dejargonizer_report"] = None
    results = [calculate_grade(row["answer"]) for index, row in answers_df.iterrows()]
    answers_df["dejargonizer_report"] = results
    return answers_df


def main():

    answers = answer_generator_from_csv("/Users/mattan.yeroushalmi/studies/thesis/Benchmarking/deep_eval/DPO_data/llama3_18B_ask_science_answers.csv", random_sample)
    answers = add_answer_grades(answers)

    answers.to_csv("/Users/mattan.yeroushalmi/studies/thesis/Benchmarking/deep_eval/Jargon/llama3_18B_ask_science_jargon_index.csv")

if __name__ == "__main__":
    main()