import os

from SFT.batch_file_gen import config

PROJECT_DIR = config.PROJECT_DIR

# Which teacher to distill answers from. Everything else (client, endpoint,
# model, file layout, request body) is derived from this. Switch teachers
# without editing code via an env var, e.g.:
#   TEACHER_PROVIDER=kimi python -m SFT.batch_file_gen.gen_batch
PROVIDER = os.environ.get("TEACHER_PROVIDER", "openai").strip().lower()

# Per-provider settings. `api_key` names refer to attributes you set in
# SFT/batch_file_gen/config.py (which is gitignored). Both providers speak the
# OpenAI Batch wire format, they only differ in base_url / model / request body.
PROVIDERS = {
    "openai": {
        "api_key": getattr(config, "OPENAI_API_KEY", None),
        "base_url": None,  # default OpenAI endpoint
        "model": "gpt-5-2025-08-07",
        "output_token_limit": 2048,
        "input_dir": "gpt_5_input_batches",
        "output_dir": "GPT5_outputs",
        "answers_csv": "ask_science_gpt_5_answers.csv",
        "jobs_file": "batch_jobs.json",
        "recovery_file": "batch_status.txt",
    },
    "kimi": {
        # Moonshot's OpenAI-compatible endpoint. This is the international host;
        # use https://api.moonshot.cn/v1 if your key is a mainland-China account.
        "api_key": getattr(config, "MOONSHOT_API_KEY", None),
        "base_url": getattr(config, "MOONSHOT_BASE_URL", "https://api.moonshot.ai/v1"),
        "model": getattr(config, "MOONSHOT_MODEL", "kimi-k2.6"),
        "output_token_limit": getattr(config, "MOONSHOT_MAX_TOKENS", 768),
        "input_dir": "kimi_input_batches",
        "output_dir": "kimi_outputs",
        "answers_csv": "ask_science_kimi_answers.csv",
        "jobs_file": "batch_jobs_kimi.json",
        "recovery_file": "batch_status_kimi.txt",
    },
}

if PROVIDER not in PROVIDERS:
    raise ValueError(
        f"Unknown TEACHER_PROVIDER={PROVIDER!r}; choose from {list(PROVIDERS)}"
    )

ACTIVE = PROVIDERS[PROVIDER]
MODEL = ACTIVE["model"]
BASE_URL = ACTIVE["base_url"]
API_KEY = ACTIVE["api_key"]
OUTPUT_TOKEN_LIMIT = ACTIVE["output_token_limit"]

INPUT_CSV = f"{PROJECT_DIR}/SFT/data/ask_science.csv"
OUTPUT_CSV = f"{PROJECT_DIR}/SFT/data/{ACTIVE['answers_csv']}"
GPT_INPUT_BATCH_DIR = f"{PROJECT_DIR}/SFT/data/{ACTIVE['input_dir']}"
GPT_OUTPUT_DIR = f"{PROJECT_DIR}/SFT/data/{ACTIVE['output_dir']}"
GPT_OUTPUT_FILE_PREFIX = "gpt_output_file_"
GPT_INPUT_BATCH_PREFIX = "sft_input_batch_file_"

JOBS_PATH = f"{PROJECT_DIR}/SFT/{ACTIVE['jobs_file']}"
RECOVERY_PATH = f"{PROJECT_DIR}/SFT/{ACTIVE['recovery_file']}"
COMPLETION_WINDOW = "24h"


def get_client():
    """OpenAI SDK client pointed at the active provider's endpoint."""
    from openai import OpenAI

    if not API_KEY:
        key_name = "OPENAI_API_KEY" if PROVIDER == "openai" else "MOONSHOT_API_KEY"
        raise RuntimeError(
            f"No API key for provider '{PROVIDER}'. Add {key_name} = \"...\" to "
            f"SFT/batch_file_gen/config.py"
        )
    kwargs = {"api_key": API_KEY}
    if BASE_URL:
        kwargs["base_url"] = BASE_URL
    return OpenAI(**kwargs)


def build_request_body(model, system_prompt, content):
    """Provider-specific chat-completion body for one batch request line.

    OpenAI (gpt-5) uses reasoning_effort + max_completion_tokens. Moonshot/Kimi
    k2 models reject the sampling params (temperature/top_p/n/penalties) in batch
    mode and use max_tokens instead, so we send a minimal body there.
    """
    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": content},
    ]
    if PROVIDER == "openai":
        return {
            "model": model,
            "messages": messages,
            "reasoning_effort": "medium",
            "max_completion_tokens": OUTPUT_TOKEN_LIMIT,
        }
    # Kimi k2 models "think" by default, burning the token budget on hidden
    # reasoning before the answer (and slowing/inflating each request). For a
    # distillation teacher we want direct answers, so disable thinking.
    return {
        "model": model,
        "messages": messages,
        "max_tokens": OUTPUT_TOKEN_LIMIT,
        "thinking": {"type": "disabled"},
    }


# System prompt for generating high-quality scientific answers
SFT_SYSTEM_PROMPT = """You are tasked with writing high-quality scientific answers, given these criteria:
1. The explanation should have a structured flow from simple to complex concepts.
2. Establish clear connections between various parts of the explanation.
3. Assume the reader has minimal prior knowledge.
4. Usage of didactic tools such as examples, metaphors, analogy, and humor is encouraged.
5. If possible, try to paint mental images that will stay with the reader. e.g. "Consider each computer as a node and the Internet as a web."
6. Avoid domain specific jargon and unfamiliar concepts.
7. Ensure the language is unambiguous, concise, and with clearly defined terminology.
8. Use of paragraphs will be preferred over bullet points and lists. 

The answers should be around two to three paragraphs long.
"""
