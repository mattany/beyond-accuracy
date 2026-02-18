"""
Generate responses from GPT-5, Kimi K2 Thinking, and Grok 4
and add them as new columns to corrected_evaluation_dataset.csv.

All three APIs support OpenAI-compatible endpoints, so we use ChatOpenAI
with custom base_url where needed. Models run in parallel via asyncio
with per-model RPM rate limiting. Each individual response is persisted
to a per-model recovery JSON so progress survives crashes.
"""

import asyncio
import json
import re
import sys
from pathlib import Path

import pandas as pd
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from tqdm.asyncio import tqdm as atqdm

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from config import PROJECT_DIR, OPENAI_API_KEY, XAI_API_KEY, MOONSHOT_API_KEY

RECOVERY_DIR = (
    Path(PROJECT_DIR)
    / "Benchmarking/deep_eval/data/test_data/.recovery"
)

MODELS = [
    {
        "column_name": "gpt-5",
        "model_id": "gpt-5",
        "api_key": OPENAI_API_KEY,
        "base_url": None,
        "rpm": 500,  # OpenAI Tier 3 ($100 paid)
    },
    # {
    #     "column_name": "gemini-3-pro",
    #     "model_id": "gemini-3-pro-preview",
    #     "api_key": GOOGLE_API_KEY,
    #     "base_url": "https://generativelanguage.googleapis.com/v1beta/openai/",
    #     "rpm": 25,  # Google AI free tier — disabled (503s)
    # },
    {
        "column_name": "kimi-k2-thinking",
        "model_id": "kimi-k2-thinking",
        "api_key": MOONSHOT_API_KEY,
        "base_url": "https://api.moonshot.ai/v1",
        "rpm": 60,  # Moonshot AI — adjust based on your account tier
    },
    {
        "column_name": "grok-4",
        "model_id": "grok-4",
        "api_key": XAI_API_KEY,
        "base_url": "https://api.x.ai/v1",
        "rpm": 480,  # xAI default for grok-4
    },
]

PROMPT = ChatPromptTemplate.from_template(
    "Answer the following question succinctly in three paragraphs or less. "
    "Keep your answer short.\nQuestion: {question}"
)


class RPMRateLimiter:
    """Sliding-window rate limiter: allows up to *rpm* requests per 60 s."""

    def __init__(self, rpm: int):
        self._sem = asyncio.Semaphore(rpm)

    async def _release_later(self):
        await asyncio.sleep(60.0)
        self._sem.release()

    async def acquire(self):
        await self._sem.acquire()
        asyncio.ensure_future(self._release_later())


def _recovery_path(model_name: str) -> Path:
    return RECOVERY_DIR / f"{model_name}.json"


def load_recovery(model_name: str) -> dict[int, str]:
    path = _recovery_path(model_name)
    if path.exists():
        data = json.loads(path.read_text())
        return {int(k): v for k, v in data.items()}
    return {}


def save_to_recovery(model_name: str, idx: int, response: str, lock: asyncio.Lock):
    """Append a single result to the recovery file (called under lock)."""
    path = _recovery_path(model_name)
    data = {}
    if path.exists():
        data = json.loads(path.read_text())
    data[str(idx)] = response
    path.write_text(json.dumps(data, ensure_ascii=False))


MAX_RETRIES = 5
RETRYABLE_STATUS_CODES = {"429", "503", "500", "502"}


def _parse_retry_delay(error_msg: str) -> float | None:
    """Extract retry delay from API error message (e.g. 'retry in 27.08s')."""
    match = re.search(r"retry in ([\d.]+)s", str(error_msg), re.IGNORECASE)
    return float(match.group(1)) if match else None


def _is_retryable(e: Exception) -> bool:
    msg = str(e)
    return any(code in msg for code in RETRYABLE_STATUS_CODES)


async def generate_single(
    chain,
    idx: int,
    question: str,
    model_name: str,
    limiter: RPMRateLimiter,
    file_lock: asyncio.Lock,
) -> tuple[int, str]:
    for attempt in range(MAX_RETRIES):
        await limiter.acquire()
        try:
            resp = await chain.ainvoke({"question": question})
            content = resp.content
            async with file_lock:
                save_to_recovery(model_name, idx, content, file_lock)
            return idx, content
        except Exception as e:
            if not _is_retryable(e):
                raise
            delay = _parse_retry_delay(str(e)) or (30 * (2 ** attempt))
            print(f"  {model_name} q{idx} (attempt {attempt + 1}/{MAX_RETRIES}): "
                  f"{e!s:.200}\n    -> retrying in {delay:.0f}s")
            await asyncio.sleep(delay)
    raise RuntimeError(f"{model_name} q{idx}: exhausted {MAX_RETRIES} retries")


async def generate_responses(questions: list[str], model_cfg: dict) -> list[str]:
    kwargs = {"model": model_cfg["model_id"], "api_key": model_cfg["api_key"]}
    if model_cfg["base_url"]:
        kwargs["base_url"] = model_cfg["base_url"]

    name = model_cfg["column_name"]
    recovered = load_recovery(name)
    if recovered:
        print(f"  {name}: recovered {len(recovered)}/{len(questions)} from previous run")

    llm = ChatOpenAI(**kwargs)
    chain = PROMPT | llm
    limiter = RPMRateLimiter(model_cfg["rpm"])
    file_lock = asyncio.Lock()

    tasks = []
    for i, q in enumerate(questions):
        if i not in recovered:
            tasks.append(generate_single(chain, i, q, name, limiter, file_lock))

    if tasks:
        results = await atqdm.gather(
            *tasks,
            desc=f"{name} ({model_cfg['rpm']} rpm)",
            total=len(tasks),
        )
        for idx, content in results:
            recovered[idx] = content

    return [recovered[i] for i in range(len(questions))]


async def run_model(
    cfg: dict,
    questions: list[str],
    results: dict,
    input_path: Path,
):
    name = cfg["column_name"]
    print(f"  Starting: {name}")
    try:
        responses = await generate_responses(questions, cfg)
        results[name] = responses
        print(f"  Done: {name} ({len(responses)} responses)")
    except Exception as e:
        print(f"  ERROR generating {name}: {e}")


async def main():
    input_path = (
        Path(PROJECT_DIR)
        / "Benchmarking/deep_eval/data/test_data/corrected_evaluation_dataset.csv"
    )
    df = pd.read_csv(input_path)

    if "question" not in df.columns:
        raise ValueError("CSV must contain a 'question' column.")

    questions = df["question"].tolist()
    print(f"Loaded {len(questions)} questions from {input_path.name}")

    RECOVERY_DIR.mkdir(parents=True, exist_ok=True)

    models_to_run = []
    for cfg in MODELS:
        if not cfg["api_key"]:
            print(f"  SKIPPING {cfg['column_name']} — API key not set in config.py")
            continue
        if cfg["column_name"] in df.columns:
            existing_recovery = _recovery_path(cfg["column_name"])
            if existing_recovery.exists():
                existing_recovery.unlink()
            print(f"  SKIPPING {cfg['column_name']} — column already exists in CSV")
            continue
        models_to_run.append(cfg)

    if not models_to_run:
        print("No models to run. Check API keys in config.py and existing columns.")
        return

    print(f"\nRunning {len(models_to_run)} models in parallel: "
          f"{[m['column_name'] for m in models_to_run]}\n")

    results: dict[str, list[str]] = {}
    await asyncio.gather(
        *(run_model(cfg, questions, results, input_path) for cfg in models_to_run)
    )

    for name, responses in results.items():
        df[name] = responses

    output_path = input_path.parent / "corrected_evaluation_dataset.csv"
    df.to_csv(output_path, index=False)
    print(f"\nUpdated dataset saved to: {output_path}")
    print(f"New columns added: {list(results.keys())}")
    print(f"Total columns now: {len(df.columns)}")

    for cfg in models_to_run:
        recovery = _recovery_path(cfg["column_name"])
        if recovery.exists():
            recovery.unlink()
    if RECOVERY_DIR.exists() and not any(RECOVERY_DIR.iterdir()):
        RECOVERY_DIR.rmdir()
    print("Recovery files cleaned up.")


if __name__ == "__main__":
    asyncio.run(main())
