"""Run one interactive teacher-generation request."""

from argparse import ArgumentParser
import os
import textwrap
import time


SYSTEM_PROMPT = """You are tasked with writing high-quality scientific answers, given these criteria:
1. The explanation should have a structured flow from simple to complex concepts.
2. Establish clear connections between various parts of the explanation.
3. Assume the reader has minimal prior knowledge.
4. Usage of didactic tools such as examples, metaphors, analogy, and humor is encouraged.
5. If possible, try to paint mental images that will stay with the reader.
6. Avoid domain specific jargon and unfamiliar concepts.
7. Ensure the language is unambiguous, concise, and with clearly defined terminology.
8. Use of paragraphs will be preferred over bullet points and lists.

The answers should be around two to three paragraphs long.
"""


def parse_args(argv=None):
    parser = ArgumentParser(description=__doc__)
    parser.add_argument(
        "question",
        nargs="?",
        default="When dams are being built, how do they build it with all the water still there?",
    )
    parser.add_argument("--model", default="gpt-3.5-turbo")
    parser.add_argument("--max-tokens", type=int, default=256)
    return parser.parse_args(argv)


def load_api_key():
    api_key = os.environ.get("OPENAI_API_KEY")
    if not api_key:
        raise RuntimeError("OPENAI_API_KEY is required to run the OpenAI playground")
    return api_key


def main(argv=None):
    args = parse_args(argv)
    api_key = load_api_key()

    from openai import OpenAI

    client = OpenAI(api_key=api_key)
    started = time.time()
    completion = client.chat.completions.create(
        model=args.model,
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": args.question},
        ],
        max_tokens=args.max_tokens,
    )
    print(f"Time: {time.time() - started} seconds")
    print(completion.usage)
    print(textwrap.fill(completion.choices[0].message.content, width=256))


if __name__ == "__main__":
    main()
