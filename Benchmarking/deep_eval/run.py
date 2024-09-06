import asyncio
import time
import pandas as pd
import tqdm
from tqdm.asyncio import tqdm_asyncio
from deepeval import assert_test
from deepeval.test_case import LLMTestCase
from deepeval.metrics import AnswerRelevancyMetric
import textwrap
from mlx_model import MLXModel
from ollama_model import OllamaModel
from prompt_templates import generate_prompt, system_prompt

BATCH_SIZE = 1024
llama_3_1_8b_instruct = "mlx-community/Meta-Llama-3.1-8B-Instruct-4bit"
llama_3_8b_instruct = "mlx-community/Meta-Llama-3-8B-Instruct-4bit"

# custom_llm = MLXModel(llama_3_8b_instruct)

# custom_llm = OllamaModel(system_prompt=system_prompt, visual=True)
custom_llm = OllamaModel(visual=False)
# custom_llm = OllamaModel(model_name="llama3.1:70b", system_prompt=system_prompt, visual=True)

async def generate_answer(prompt, index=0, visual=True):

    # print(f"MODEL NAME: {custom_llm.get_model_name()}")
    # prompt = "is it in theory possible to 3d print a joint?"
    # prompt = 'Why is the sky blue?'
    prompt_with_system = generate_prompt(prompt)
    # print(f"Awaiting prompt {prompt}")
    if visual:
        generation = custom_llm.generate(
            prompt_with_system
            # prompt
        )
        wrapper = textwrap.TextWrapper(width=100, replace_whitespace=False, drop_whitespace=False)

        wrapped_text = "\n".join([wrapper.fill(line) for line in generation.splitlines()])
        return index, wrapped_text
    generation = await custom_llm.a_generate(
        prompt_with_system
        # prompt
    )
    return index, prompt, generation
    # print(wrapped_text)


async def generate_answers_in_parallel(prompts):
    tasks = [generate_answer(prompt, index, visual=False) for index, prompt in prompts]
    results = await tqdm_asyncio.gather(*tasks)
    return results


def create_batches(csv_file, batch_size):
    # Read the CSV (assuming it's tab-delimited)
    df = pd.read_csv(csv_file, delimiter='\t')

    # Create list of tuples (index, question)
    questions = list(df.itertuples(index=False, name=None))

    # Group questions into batches of size N
    batches = [questions[i:i + batch_size] for i in range(0, len(questions), batch_size)]

    return batches


async def main():
    dataset_path = "/Users/mattan.yeroushalmi/studies/thesis/SFT/data/ask_science.csv"
    batches = create_batches(dataset_path, batch_size=BATCH_SIZE)
    for batch in tqdm.tqdm(batches):
        # start_time = time.time()
        results = await generate_answers_in_parallel(batch)
        # end_time = time.time()
        # print(f"Batch {i} took {end_time - start_time} seconds")
        # print(f"Batch {i} took {(end_time - start_time)/BATCH_SIZE} per answer. BATCH_SIZE: {BATCH_SIZE}")
        out_df = pd.DataFrame(results, columns=["index", "question", "answer"])
        out_df.to_csv(
            f"/Users/mattan.yeroushalmi/studies/thesis/Benchmarking/deep_eval/DPO_data/llama3_18B_ask_science_answers.csv",
            mode='a', header=True, index=False
        )
    # Output the results
    # for i, result in enumerate(results):
        # print(f"Prompt {i+1}: {prompts[i]}")
        # print(f"Response: {result}\n")
        # print(result)
# def test_answer_relevancy():
#     answer_relevancy_metric = AnswerRelevancyMetric(threshold=0.5)
#     test_case = LLMTestCase(
#         input="What if these shoes don't fit?",
#         # Replace this with the actual output of your LLM application
#         actual_output="We offer a 30-day full refund at no extra cost."
#     )
#     assert_test(test_case, [answer_relevancy_metric])

async def test():
    res = await generate_answer("What is the meaning of life?")
    print(res[1])

if __name__ == "__main__":
    asyncio.run(main())