"""One-shot script: wait for the manually-cancelled batches to finalize, download
their partial outputs, build a single 'makeup' batch containing just the missing
(straggler) requests from those cancelled batches, submit it, wait for it to
complete, download it, then run the merge step.

Does NOT touch batch 29's single max_tokens failure (question-batch-5834) -
that one was explicitly accepted as a loss and should stay excluded.
"""
import glob
import json
import os
import time

from training.data_generation.constants import (
    GPT_INPUT_BATCH_DIR,
    GPT_INPUT_BATCH_PREFIX,
    GPT_OUTPUT_DIR,
    GPT_OUTPUT_FILE_PREFIX,
    JOBS_PATH,
    get_client,
)
from training.data_generation.upload_batch_file import (
    create_batch,
    logger,
    output_path,
    poll_until_done,
    save_jobs,
    upload_batch_file,
)

client = get_client()

CANCELLED_BATCH_IDS = {
    0: "batch_6a4fce6001c08190b13530e164ff50e3",
    13: "batch_6a4fd4a39f308190a21aa73cd699c0d6",
    25: "batch_6a4fd4af507c8190863d0f76722bfd81",
    35: "batch_6a4fd4ba056081909f79904e21c5b189",
    38: "batch_6a4fd4bd211c8190918eedc03a6bb061",
    41: "batch_6a4fd4bfa5a48190b403d5678a1e8ea0",
}

POLL_INTERVAL_SECONDS = 30


def load_jobs():
    with open(JOBS_PATH, "r") as f:
        return {int(k): v for k, v in json.load(f).items()}


def wait_for_cancellation(batch_ids):
    remaining = dict(batch_ids)
    finalized = {}
    while remaining:
        done = []
        for idx, bid in remaining.items():
            b = client.batches.retrieve(bid)
            logger.info(f"batch {idx} ({bid}) status={b.status}")
            if b.status != "cancelling":
                finalized[idx] = b
                done.append(idx)
        for idx in done:
            remaining.pop(idx)
        if remaining:
            time.sleep(POLL_INTERVAL_SECONDS)
    return finalized


def download_cancelled_outputs(finalized_batches, jobs):
    for idx, batch_info in finalized_batches.items():
        path = output_path(GPT_OUTPUT_DIR, idx)
        if batch_info.output_file_id:
            content = client.files.content(batch_info.output_file_id)
            os.makedirs(GPT_OUTPUT_DIR, exist_ok=True)
            with open(path, "w") as f:
                f.write(content.text)
            logger.info(f"Downloaded partial output for batch {idx} -> {path}")
        else:
            logger.warning(f"batch {idx} has no output_file_id; nothing completed before cancellation")

        counts = batch_info.request_counts
        jobs[idx] = {
            "batch_id": CANCELLED_BATCH_IDS[idx],
            "status": batch_info.status,
            "downloaded": True,
            "counts": {
                "completed": counts.completed,
                "failed": counts.failed,
                "total": counts.total,
            },
        }
    save_jobs(jobs)
    return jobs


def custom_ids_in_file(path):
    ids = set()
    if not os.path.exists(path):
        return ids
    with open(path, "r") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            ids.add(json.loads(line)["custom_id"])
    return ids


def collect_stragglers(batch_indices):
    """Returns the raw request lines (as strings) for every custom_id present in
    the input batch file but missing from its corresponding output file."""
    straggler_lines = []
    for idx in batch_indices:
        input_path = f"{GPT_INPUT_BATCH_DIR}/{GPT_INPUT_BATCH_PREFIX}{idx}.jsonl"
        out_path = output_path(GPT_OUTPUT_DIR, idx)
        output_ids = custom_ids_in_file(out_path)

        missing_count = 0
        with open(input_path, "r") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                custom_id = json.loads(line)["custom_id"]
                if custom_id not in output_ids:
                    straggler_lines.append(line)
                    missing_count += 1
        logger.info(f"batch {idx}: {missing_count} straggler request(s) missing from output")

    return straggler_lines


def next_batch_index():
    existing = glob.glob(f"{GPT_INPUT_BATCH_DIR}/{GPT_INPUT_BATCH_PREFIX}*.jsonl")
    indices = [
        int(os.path.basename(p)[len(GPT_INPUT_BATCH_PREFIX):-len(".jsonl")])
        for p in existing
    ]
    return max(indices) + 1 if indices else 0


def write_makeup_batch_file(straggler_lines):
    makeup_index = next_batch_index()
    makeup_path = f"{GPT_INPUT_BATCH_DIR}/{GPT_INPUT_BATCH_PREFIX}{makeup_index}.jsonl"
    with open(makeup_path, "w") as f:
        f.write("\n".join(straggler_lines) + "\n")
    logger.info(f"Wrote makeup batch with {len(straggler_lines)} stragglers -> {makeup_path}")
    return makeup_index, makeup_path


def submit_makeup_batch(jobs, makeup_index, makeup_path):
    file_id = upload_batch_file(makeup_path)
    batch_id = create_batch(file_id)
    batch_info = client.batches.retrieve(batch_id)
    jobs[makeup_index] = {
        "batch_id": batch_id,
        "file_id": file_id,
        "status": "submitted",
        "downloaded": False,
        "created_at": getattr(batch_info, "created_at", None),
    }
    save_jobs(jobs)
    logger.info(f"Submitted makeup batch {makeup_index} as {batch_id}")
    return jobs


def run_merge():
    from training.data_generation.merge import read_answers, read_questions, write_answers_to_csv
    from training.data_generation.constants import INPUT_CSV, OUTPUT_CSV

    questions = read_questions(INPUT_CSV)
    answers = read_answers()
    write_answers_to_csv(questions, answers)
    logger.info(f"Merge complete -> {OUTPUT_CSV}")


def main():
    logger.info("Waiting for cancelled batches to finalize...")
    finalized = wait_for_cancellation(CANCELLED_BATCH_IDS)

    jobs = load_jobs()
    jobs = download_cancelled_outputs(finalized, jobs)

    logger.info("Computing stragglers from cancelled batches...")
    straggler_lines = collect_stragglers(CANCELLED_BATCH_IDS.keys())

    if straggler_lines:
        makeup_index, makeup_path = write_makeup_batch_file(straggler_lines)
        jobs = submit_makeup_batch(jobs, makeup_index, makeup_path)

        logger.info("Waiting for makeup batch to complete...")
        poll_until_done(jobs, GPT_INPUT_BATCH_DIR, GPT_INPUT_BATCH_PREFIX, GPT_OUTPUT_DIR)
    else:
        logger.info("No stragglers found; skipping makeup batch.")

    logger.info("Running merge...")
    run_merge()

    logger.info("All done: cancelled batches finalized, makeup batch processed, merge complete.")


if __name__ == "__main__":
    main()
