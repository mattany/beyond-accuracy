"""On-demand, concurrent (asyncio) teacher-answer generation via the realtime
chat-completions API, as a drop-in alternative to the batch pipeline when the
provider's batch queue is throttled.

Reuses the same provider config as the batch scripts (model, endpoint, system
prompt, request body, output CSV) so the downstream split/SFT steps are
unchanged. Honors account rate limits and is resumable: results stream to a
`<output>.progress.jsonl` checkpoint, so re-running skips already-answered rows.

Usage (from repo root):
    TEACHER_PROVIDER=kimi python -m training.data_generation.realtime_gen
    TEACHER_PROVIDER=kimi python -m training.data_generation.realtime_gen --limit 10

Default limits match a Kimi account with Concurrency=100, RPM=500, TPM=3,000,000.
"""
import argparse
import asyncio
import csv
import json
import os
import random
import time

from tqdm import tqdm

from training.data_generation.constants import (
    API_KEY,
    BASE_URL,
    INPUT_CSV,
    MODEL,
    OUTPUT_CSV,
    OUTPUT_TOKEN_LIMIT,
    PROVIDER,
    SFT_SYSTEM_PROMPT,
)

PROGRESS_PATH = f"{OUTPUT_CSV}.progress.jsonl"
# Rough per-request input size (system prompt + question). Only used to reserve
# TPM budget up front; RPM is the binding limit in practice.
EST_INPUT_TOKENS = 320


class RateLimiter:
    """Async token bucket. `rate_per_min` tokens refill over 60s, capacity == rate."""

    def __init__(self, rate_per_min):
        self.capacity = float(rate_per_min)
        self.tokens = float(rate_per_min)
        self.rate = rate_per_min / 60.0
        self.updated = time.monotonic()
        self.lock = asyncio.Lock()

    async def acquire(self, amount=1):
        amount = min(amount, self.capacity)
        while True:
            async with self.lock:
                now = time.monotonic()
                self.tokens = min(self.capacity, self.tokens + (now - self.updated) * self.rate)
                self.updated = now
                if self.tokens >= amount:
                    self.tokens -= amount
                    return
                wait = (amount - self.tokens) / self.rate
            await asyncio.sleep(wait)


def make_client():
    from openai import AsyncOpenAI

    if not API_KEY:
        key_name = "OPENAI_API_KEY" if PROVIDER == "openai" else "MOONSHOT_API_KEY"
        raise RuntimeError(
            f"No API key for provider '{PROVIDER}'. Add {key_name} to "
            f"training/data_generation/config.py"
        )
    kwargs = {"api_key": API_KEY, "timeout": 180.0, "max_retries": 0}
    if BASE_URL:
        kwargs["base_url"] = BASE_URL
    return AsyncOpenAI(**kwargs)


def completion_params():
    """(native kwargs, extra_body) for chat.completions.create, per provider.

    Kimi passes `thinking` via extra_body (unknown to the SDK) and uses
    max_tokens; OpenAI uses reasoning_effort + max_completion_tokens.
    """
    if PROVIDER == "openai":
        return {"reasoning_effort": "medium", "max_completion_tokens": OUTPUT_TOKEN_LIMIT}, None
    return {"max_tokens": OUTPUT_TOKEN_LIMIT}, {"thinking": {"type": "disabled"}}


def load_questions():
    """Ordered [(index, question)], matching merge.py's enumerate-based Index."""
    questions = []
    with open(INPUT_CSV, mode="r", encoding="utf-8") as f:
        reader = csv.DictReader(f, delimiter="\t")
        for i, row in enumerate(reader):
            questions.append((i, row["Question"]))
    return questions


def load_done_indices():
    done = {}
    if not os.path.exists(PROGRESS_PATH):
        return done
    with open(PROGRESS_PATH, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            rec = json.loads(line)
            # Only treat a real answer as done; failures get retried on rerun.
            if rec.get("answer") and not str(rec["answer"]).startswith("ERROR:"):
                done[rec["index"]] = rec
    return done


async def answer_one(client, sem, rpm, tpm, native_kwargs, extra_body,
                     index, question, max_retries, writer_lock, progress_file):
    est_tokens = EST_INPUT_TOKENS + OUTPUT_TOKEN_LIMIT
    async with sem:
        last_err = "empty"
        for attempt in range(max_retries + 1):
            await rpm.acquire(1)
            await tpm.acquire(est_tokens)
            try:
                kwargs = dict(
                    model=MODEL,
                    messages=[
                        {"role": "system", "content": SFT_SYSTEM_PROMPT},
                        {"role": "user", "content": question},
                    ],
                    **native_kwargs,
                )
                if extra_body:
                    kwargs["extra_body"] = extra_body
                resp = await client.chat.completions.create(**kwargs)
                choice = resp.choices[0]
                answer = choice.message.content or ""
                if not answer.strip():
                    last_err = "empty_content"
                    raise ValueError("empty content")
                truncated = 0 if choice.finish_reason == "stop" else 1
                rec = {"index": index, "question": question, "answer": answer, "truncated": truncated}
                break
            except Exception as e:  # noqa: BLE001 - retry on any transient API error
                last_err = f"{type(e).__name__}: {str(e)[:120]}"
                if attempt == max_retries:
                    rec = {"index": index, "question": question,
                           "answer": f"ERROR: {last_err}", "truncated": 1}
                    break
                await asyncio.sleep(min(60.0, 2 ** attempt) + random.random())

    async with writer_lock:
        progress_file.write(json.dumps(rec, ensure_ascii=False) + "\n")
        progress_file.flush()
    return rec


def write_final_csv(done_records):
    rows = sorted(done_records.values(), key=lambda r: r["index"])
    with open(OUTPUT_CSV, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["Index", "Question", "Answer", "Truncated"])
        for r in rows:
            writer.writerow([r["index"], r["question"], r["answer"], r["truncated"]])


async def main_async(args):
    questions = load_questions()
    if args.limit:
        questions = questions[: args.limit]
    done = load_done_indices()
    todo = [(i, q) for (i, q) in questions if i not in done]

    print(f"provider={PROVIDER} model={MODEL} max_tokens={OUTPUT_TOKEN_LIMIT}")
    print(f"total={len(questions)} done={len(done)} todo={len(todo)}")
    print(f"limits: concurrency={args.concurrency} rpm={args.rpm} tpm={args.tpm}")
    if not todo:
        print("Nothing to do; writing final CSV from checkpoint.")
        write_final_csv(done)
        print(f"Wrote {OUTPUT_CSV}")
        return

    client = make_client()
    sem = asyncio.Semaphore(args.concurrency)
    rpm = RateLimiter(args.rpm)
    tpm = RateLimiter(args.tpm)
    native_kwargs, extra_body = completion_params()
    writer_lock = asyncio.Lock()

    errors = 0
    truncated = 0
    with open(PROGRESS_PATH, "a", encoding="utf-8") as progress_file:
        tasks = [
            asyncio.create_task(
                answer_one(client, sem, rpm, tpm, native_kwargs, extra_body,
                           i, q, args.max_retries, writer_lock, progress_file)
            )
            for (i, q) in todo
        ]
        for fut in tqdm(asyncio.as_completed(tasks), total=len(tasks), desc="answers", unit="q"):
            rec = await fut
            done[rec["index"]] = rec
            if str(rec["answer"]).startswith("ERROR:"):
                errors += 1
            elif rec["truncated"]:
                truncated += 1

    await client.close()

    write_final_csv(done)
    print(f"\nDone. wrote {OUTPUT_CSV}")
    print(f"answers={len(done)} errors={errors} truncated={truncated}")
    if errors:
        print("Re-run the same command to retry only the errored rows (resume via checkpoint).")


def parse_args():
    p = argparse.ArgumentParser(description="Concurrent realtime teacher-answer generation.")
    p.add_argument("--concurrency", type=int, default=100)
    p.add_argument("--rpm", type=int, default=500, help="max requests per minute")
    p.add_argument("--tpm", type=int, default=3_000_000, help="max tokens per minute (reserved by estimate)")
    p.add_argument("--max-retries", type=int, default=5)
    p.add_argument("--limit", type=int, default=0, help="only process first N questions (smoke test)")
    return p.parse_args()


if __name__ == "__main__":
    asyncio.run(main_async(parse_args()))
