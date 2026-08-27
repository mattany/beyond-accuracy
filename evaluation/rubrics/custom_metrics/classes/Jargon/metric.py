import pandas as pd

from evaluation.rubrics.custom_metrics.classes.Jargon.config import JARGON_BASE_PATH
from evaluation.rubrics.custom_metrics.classes.Jargon.jargon_util import analyze_text
from deepeval.metrics import BaseMetric
from deepeval.test_case import LLMTestCase


class JargonMetric(BaseMetric):
    def __init__(
        self,
        threshold: float = 0.5,
        verbose_mode: bool = False,
    ):
        self.threshold = threshold
        self.verbose_mode = verbose_mode
        self.names = set(pd.read_csv(JARGON_BASE_PATH + "names.csv", header=None)[0])
        WORDS = "DataUKUS2018-2021.csv"
        self.words = (
            pd.read_csv(JARGON_BASE_PATH + WORDS, header=None).set_index(0)[1].to_dict()
        )

    def measure(self, test_case: LLMTestCase) -> float:
        # Although not required, we recommend catching errors
        # in a try block
        try:
            self.score = analyze_text(test_case.actual_output, self.words, self.names, verbose=self.verbose_mode)
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
