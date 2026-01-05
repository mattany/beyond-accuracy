from deepeval.metrics import BaseMetric
from deepeval.test_case import LLMTestCase
import scireadability


class ReadabilityMetric(BaseMetric):
    """
    Readability metric using scireadability library.
    
    scireadability is a fork of textstat optimized for scientific text with improved
    syllable counting for technical/scientific vocabulary.
    Supports short texts (no minimum word count requirement).
    """
    
    # Map metric names to scireadability functions
    METRIC_MAP = {
        'flesch_kincaid': scireadability.flesch_kincaid_grade,
        'flesch': scireadability.flesch_reading_ease,
        'dale_chall': scireadability.dale_chall_readability_score,
        'ari': scireadability.automated_readability_index,
    }
    
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
        # Use textstat library which handles short texts
        try:
            if self.metric not in self.METRIC_MAP:
                raise ValueError(f"Unknown metric: {self.metric}. Available: {list(self.METRIC_MAP.keys())}")
            
            func = self.METRIC_MAP[self.metric]
            result = func(test_case.actual_output)
            
            # Flesch Reading Ease is 0-100 scale, normalize to 0-1
            if self.metric == 'flesch':
                self.score = result / 100
            else:
                self.score = result
            
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
