from argparse import ArgumentParser
import asyncio
from pathlib import Path
import sys
import textwrap

if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import pandas as pd
import tqdm
from tqdm.asyncio import tqdm_asyncio

from evaluation.rubrics.ollama_model import OllamaModel
from evaluation.rubrics.prompt_templates import generate_prompt, system_prompt

BATCH_SIZE = 32
ROOT = Path(__file__).resolve().parents[2]
llama_3_1_8b_instruct = "mlx-community/Meta-Llama-3.1-8B-Instruct-4bit"
llama_3_8b_instruct = "mlx-community/Meta-Llama-3-8B-Instruct-4bit"

custom_llm = OllamaModel(system_prompt=system_prompt, visual=True)
# custom_llm = OllamaModel(visual=False)
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


async def main(args):
    batches = create_batches(args.input, batch_size=BATCH_SIZE)
    i = 0
    for batch in tqdm.tqdm(batches):
        i += 1
        if i <= 64:
            
            print(f"skipping batch {i}")
       
            continue
        # start_time = time.time()
        results = await generate_answers_in_parallel(batch)
        # end_time = time.time()
        # print(f"Batch {i} took {end_time - start_time} seconds")
        # print(f"Batch {i} took {(end_time - start_time)/BATCH_SIZE} per answer. BATCH_SIZE: {BATCH_SIZE}")
        out_df = pd.DataFrame(results, columns=["index", "question", "answer"])
        args.output.parent.mkdir(parents=True, exist_ok=True)
        out_df.to_csv(args.output, mode='a', header=False, index=False)

def parse_args(argv=None):
    parser = ArgumentParser()
    parser.add_argument(
        "--input",
        type=Path,
        default=ROOT / "data" / "qa_pairs" / "ask_science.csv",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=ROOT / "training" / "dpo" / "llama3_18B_ask_science_answers.csv",
    )
    return parser.parse_args(argv)


if __name__ == "__main__":
    asyncio.run(main(parse_args()))