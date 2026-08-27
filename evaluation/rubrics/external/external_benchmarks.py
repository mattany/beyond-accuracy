from deepeval.benchmarks import TruthfulQA
from deepeval.benchmarks.tasks import TruthfulQATask
from deepeval.benchmarks.modes import TruthfulQAMode

# Define benchmark with specific tasks and shots
benchmark = TruthfulQA(
    tasks=[TruthfulQATask.SCIENCE],
    mode=TruthfulQAMode.MC2
)

# Replace 'mistral_7b' with your own custom model
benchmark.evaluate(model=mistral_7b)
print(benchmark.overall_score)