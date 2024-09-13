from deepeval.models import DeepEvalBaseLLM
from ollama_model import OllamaModel


def truthful_QA(model: DeepEvalBaseLLM):
    from deepeval.benchmarks import TruthfulQA
    from deepeval.benchmarks.tasks import TruthfulQATask
    from deepeval.benchmarks.modes import TruthfulQAMode

    # Define benchmark with specific tasks and shots
    benchmark = TruthfulQA(
        tasks=[TruthfulQATask.ADVERTISING, TruthfulQATask.FICTION],
        mode=TruthfulQAMode.MC2
    )

    # Replace 'mistral_7b' with your own custom model
    benchmark.evaluate(model=model)
    print(benchmark.overall_score)


custom_llm = OllamaModel(visual=False)
truthful_QA(custom_llm)