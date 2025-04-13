from datetime import datetime
import pandas as pd
import logging

from custom_metrics.classes.Jargon.config import JARGON_BASE_PATH
from custom_metrics.classes.Jargon.jargon_util import calculate_grade
from deepeval.metrics import BaseMetric
from deepeval.test_case import LLMTestCase
import asyncio


class JargonMetric(BaseMetric):
    def __init__(
        self,
        threshold: float = 0.5,
        # Optional
    ):
        self.threshold = threshold
        self.names = set(pd.read_csv(JARGON_BASE_PATH + "names.csv", header=None)[0])
        WORDS = "DataUKUS2018-2021.csv"
        self.words = (
            pd.read_csv(JARGON_BASE_PATH + WORDS, header=None).set_index(0)[1].to_dict()
        )

    def measure(self, test_case: LLMTestCase) -> float:
        # Although not required, we recommend catching errors
        # in a try block
        try:
            self.score = calculate_grade(test_case.actual_output, self.names, self.words)
            self.success = self.score >= self.threshold
            return self.score
        except Exception as e:
            # set metric error and re-raise it
            self.error = str(e)
            raise

    async def a_measure(self, test_case: LLMTestCase) -> float:
        return self.measure(test_case)
        # return await asyncio.to_thread(self.measure, test_case)

    def is_successful(self) -> bool:
        if self.error is not None:
            self.success = False
        else:
            return self.success

    @property
    def __name__(self):
        return "Jargon Metric"


logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger("main_logger")


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
    random_sample = [3898, 8916, 2136, 6061, 7766]
    answers = answer_generator_from_csv(
        "/Users/mattan.yeroushalmi/studies/thesis/Benchmarking/deep_eval/ollama-bad-answers/llama3_18B_ask_science_answers.csv",
        random_sample,
    )
    answers = add_answer_grades(answers)

    answers.to_csv(
        "/Users/mattan.yeroushalmi/studies/thesis/Benchmarking/deep_eval/Jargon/llama3_18B_ask_science_jargon_index.csv"
    )


if __name__ == "__main__":
    main()
