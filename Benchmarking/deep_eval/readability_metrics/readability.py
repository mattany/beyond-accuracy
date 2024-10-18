from deepeval.metrics import BaseMetric
from deepeval.test_case import LLMTestCase
from readability import Readability


class ReadabilityMetric(BaseMetric):
    def __init__(
            self,
            metric: str,
            name: str = "",
            threshold: float = 0.5,

    ):
        self.name = name or metric.capitalize()
        self.metric = metric
        self.threshold = threshold
        self.score = -1

    def measure(self, test_case: LLMTestCase) -> float:
        # Although not required, we recommend catching errors
        # in a try block
        try:
            r = Readability(test_case.actual_output)
            result = getattr(r, self.metric)()
            # print(f"Grade level {fk.grade_level}")
            if self.metric == 'flesch':
                self.score = result.score / 100
            else:
                self.score = result.score
            self.success = self.score >= self.threshold
            return self.score
        except Exception as e:
            # set metric error and re-raise it
            self.error = str(e)
            raise

    async def a_measure(self, test_case: LLMTestCase) -> float:
        return self.measure(test_case)

    def is_successful(self) -> bool:
        if self.error is not None:
            self.success = False
        else:
            return self.success

    @property
    def __name__(self):
        return self.name
