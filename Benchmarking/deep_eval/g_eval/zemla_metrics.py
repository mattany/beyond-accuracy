from deepeval.metrics import GEval
from deepeval.test_case import LLMTestCaseParams

internal_coherence_metric = GEval(
    name="Internal Coherence",
    criteria="The parts of this explanation fit together coherently.",
    evaluation_params=[LLMTestCaseParams.ACTUAL_OUTPUT]
)

completeness_metric = GEval(
    name="Completeness",
    criteria="There are no gaps in this explanation (actual output) for the given input question.",
    evaluation_params=[LLMTestCaseParams.INPUT, LLMTestCaseParams.ACTUAL_OUTPUT]
)

completeness_metric_explicit = GEval(
    name="Completeness (Explicit)",
    evaluation_steps=[
        "Read and understand the following definition: if an explanation suggests that A causes B, but it is not immediately clear how A causes B, that is considered a gap.",
        'Given the input question and output model answer, determine if the answer has gaps or not.',
        "It doesn't matter if the answer or the explanations in it are correct or not. If the answer has any gaps, give it a score of 0. Otherwise give it a score of 10.",
        "Make sure that the reason you gave for the score doesn't take correctness or accuracy into account, focusing only on gaps in the explanation."
    ],
    evaluation_params=[LLMTestCaseParams.INPUT, LLMTestCaseParams.ACTUAL_OUTPUT]
)

alternatives_metric = GEval(
    name="Lack of Alternatives",
    criteria="There probably aren't many other reasonable alternative explanations to the actual output for this input question.",
    evaluation_params=[LLMTestCaseParams.INPUT, LLMTestCaseParams.ACTUAL_OUTPUT]
)

articulation_metric = GEval(
    name="Articulation",
    criteria="Regardless of accuracy, this explanation is well articulated.",
    evaluation_params=[LLMTestCaseParams.ACTUAL_OUTPUT]
)

perceived_truth_metric = GEval(
    name="Perceived Truth",
    criteria="You believe this explanation to be true.",
    evaluation_params=[LLMTestCaseParams.ACTUAL_OUTPUT]
)

