import asyncio

import pandas as pc
import pandas as pd
from deepeval import assert_test
from deepeval.test_case import LLMTestCase
from deepeval.metrics import AnswerRelevancyMetric
import textwrap
from mlx_model import MLXModel
from ollama_model import OllamaModel
from prompt_templates import generate_prompt, system_prompt

llama_3_1_8b_instruct = "mlx-community/Meta-Llama-3.1-8B-Instruct-4bit"
llama_3_8b_instruct = "mlx-community/Meta-Llama-3-8B-Instruct-4bit"

# custom_llm = MLXModel(llama_3_8b_instruct)

custom_llm = OllamaModel(system_prompt=system_prompt)


async def generate_answer(prompt):

    # print(f"MODEL NAME: {custom_llm.get_model_name()}")
    # prompt = "is it in theory possible to 3d print a joint?"
    # prompt = 'Why is the sky blue?'
    prompt_with_system = generate_prompt(prompt)
    print(f"Awaiting prompt {prompt}")
    generation = await custom_llm.a_generate(
        prompt_with_system
        # prompt
    )
    wrapper = textwrap.TextWrapper(width=100, replace_whitespace=False, drop_whitespace=False)

    wrapped_text = "\n".join([wrapper.fill(line) for line in generation.splitlines()])
    return wrapped_text
    # print(wrapped_text)


async def generate_answers_in_parallel(prompts):
    tasks = [generate_answer(prompt) for prompt in prompts]
    results = await asyncio.gather(*tasks)
    return results

async def main():
    prompts = pd.read_csv()
    prompts = [
        "What is the capital of France?",
        "Explain the theory of relativity.",
        "Describe quantum mechanics in simple terms.",
    ]

    results = await generate_answers_in_parallel(prompts)

    # Output the results
    for i, result in enumerate(results):
        # print(f"Prompt {i+1}: {prompts[i]}")
        # print(f"Response: {result}\n")
        print(result)
# def test_answer_relevancy():
#     answer_relevancy_metric = AnswerRelevancyMetric(threshold=0.5)
#     test_case = LLMTestCase(
#         input="What if these shoes don't fit?",
#         # Replace this with the actual output of your LLM application
#         actual_output="We offer a 30-day full refund at no extra cost."
#     )
#     assert_test(test_case, [answer_relevancy_metric])

if __name__ == "__main__":
    asyncio.run(main())