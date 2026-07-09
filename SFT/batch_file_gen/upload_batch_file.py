import json
import os
import os.path
from time import sleep

from openai import NotFoundError, OpenAI

from SFT.batch_file_gen.constants import GPT_OUTPUT_DIR, GPT_OUTPUT_FILE_PREFIX, GPT_INPUT_BATCH_DIR, \
    GPT_INPUT_BATCH_PREFIX
from SFT.batch_file_gen.config import OPENAI_API_KEY, PROJECT_DIR
import logging

RECOVERY_PATH = f"{PROJECT_DIR}/SFT/batch_status.txt"
JOBS_PATH = f"{PROJECT_DIR}/SFT/batch_jobs.json"
POLL_INTERVAL_SECONDS = 30
TERMINAL_FAILURE_STATUSES = {"failed", "cancelling", "cancelled", "expired"}
IN_FLIGHT_STATUSES = {"validating", "in_progress", "finalizing"}

logging.basicConfig(level=logging.INFO,
                    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')

logger = logging.getLogger("main_logger")
client = OpenAI(api_key=OPENAI_API_KEY)


def upload_batch_file(path_to_batch_file):
    with open(path_to_batch_file, "rb") as f:
        res = client.files.create(file=f, purpose="batch")
    return res.id


def create_batch(file_id):
    res = client.batches.create(
        input_file_id=file_id,
        endpoint="/v1/chat/completions",
        completion_window="24h",
    )
    return res.id


def retrieve_batch(batch_id):
    return client.batches.retrieve(batch_id)


def load_jobs():
    if os.path.exists(JOBS_PATH):
        with open(JOBS_PATH, "r") as f:
            return {int(k): v for k, v in json.load(f).items()}
    return {}


def save_jobs(jobs):
    with open(JOBS_PATH, "w") as f:
        json.dump({str(k): v for k, v in sorted(jobs.items())}, f, indent=2)


def migrate_legacy_recovery(jobs):
    """Pick up a single in-flight batch from the old batch_status.txt format."""
    if not os.path.exists(RECOVERY_PATH):
        return jobs
    with open(RECOVERY_PATH, "r") as f:
        line = f.readline().strip()
    if not line:
        return jobs
    parts = line.split()
    if len(parts) != 3:
        return jobs
    status, batch_number, batch_id = parts
    batch_number = int(batch_number)
    if batch_number in jobs:
        return jobs
    jobs[batch_number] = {
        "batch_id": batch_id,
        "status": status,
        "downloaded": False,
    }
    logger.info(f"Migrated legacy recovery: batch {batch_number} -> {batch_id} ({status})")
    save_jobs(jobs)
    return jobs


def output_path(output_dir, batch_index):
    return os.path.join(output_dir, f"{GPT_OUTPUT_FILE_PREFIX}{batch_index}.jsonl")


def download_batch_output(batch_info, batch_index, output_dir):
    output_file_id = batch_info.output_file_id
    if not output_file_id:
        raise RuntimeError(f"batch {batch_index} completed without an output_file_id")
    file_response = client.files.content(output_file_id)
    os.makedirs(output_dir, exist_ok=True)
    path = output_path(output_dir, batch_index)
    with open(path, "w") as f:
        f.write(file_response.text)
    logger.info(f"Downloaded batch {batch_index} -> {path}")


def submit_missing_batches(jobs, gpt_input_batch_dir, prefix, batch_amt):
    for batch_index in range(batch_amt):
        if batch_index in jobs and jobs[batch_index].get("batch_id"):
            continue
        path = f"{gpt_input_batch_dir}/{prefix}{batch_index}.jsonl"
        if not os.path.isfile(path):
            logger.warning(f"Missing input file {path}; skipping")
            continue
        logger.info(f"Submitting batch {batch_index} from {path}")
        file_id = upload_batch_file(path)
        batch_id = create_batch(file_id)
        jobs[batch_index] = {
            "batch_id": batch_id,
            "file_id": file_id,
            "status": "submitted",
            "downloaded": False,
        }
        save_jobs(jobs)
        logger.info(f"Submitted batch {batch_index} as {batch_id}")
    return jobs


def resubmit_batch(jobs, batch_index, gpt_input_batch_dir, prefix):
    path = f"{gpt_input_batch_dir}/{prefix}{batch_index}.jsonl"
    logger.warning(f"Resubmitting batch {batch_index} from {path}")
    file_id = upload_batch_file(path)
    batch_id = create_batch(file_id)
    jobs[batch_index] = {
        "batch_id": batch_id,
        "file_id": file_id,
        "status": "submitted",
        "downloaded": False,
    }
    save_jobs(jobs)
    return batch_id


def poll_until_done(jobs, gpt_input_batch_dir, prefix, output_dir):
    while True:
        pending = []
        for batch_index, job in sorted(jobs.items()):
            if job.get("downloaded") and os.path.exists(output_path(output_dir, batch_index)):
                continue
            pending.append(batch_index)

        if not pending:
            logger.info("All batches downloaded.")
            return

        logger.info(f"Polling {len(pending)} unfinished batches...")
        for batch_index in pending:
            job = jobs[batch_index]
            batch_id = job["batch_id"]
            try:
                batch_info = retrieve_batch(batch_id)
            except NotFoundError:
                logger.warning(f"batch {batch_index} id {batch_id} not found; resubmitting")
                resubmit_batch(jobs, batch_index, gpt_input_batch_dir, prefix)
                continue

            status = batch_info.status
            job["status"] = status
            if status in IN_FLIGHT_STATUSES:
                counts = getattr(batch_info, "request_counts", None)
                if counts is not None:
                    logger.info(
                        f"batch {batch_index} {status}: "
                        f"{counts.completed}/{counts.total} completed, {counts.failed} failed"
                    )
                else:
                    logger.info(f"batch {batch_index} is {status}")
            elif status == "completed":
                if not job.get("downloaded") or not os.path.exists(output_path(output_dir, batch_index)):
                    download_batch_output(batch_info, batch_index, output_dir)
                job["downloaded"] = True
                logger.info(f"batch {batch_index} success")
            elif status in TERMINAL_FAILURE_STATUSES:
                logger.warning(f"batch {batch_index} ended with status {status}; resubmitting")
                resubmit_batch(jobs, batch_index, gpt_input_batch_dir, prefix)
            else:
                logger.info(f"batch {batch_index} has unexpected status {status}")

            save_jobs(jobs)
            # Keep legacy single-line recovery roughly up to date for the first pending job.
            persist_legacy_status(pending[0], jobs[pending[0]]["status"], jobs[pending[0]]["batch_id"])

        sleep(POLL_INTERVAL_SECONDS)


def persist_legacy_status(batch_number, batch_status, batch_id):
    with open(RECOVERY_PATH, "w") as f:
        f.write(f"{batch_status} {batch_number} {batch_id}\n")


def run(gpt_input_batch_dir=GPT_INPUT_BATCH_DIR, prefix=GPT_INPUT_BATCH_PREFIX, output_dir=GPT_OUTPUT_DIR):
    batch_amt = len([
        f for f in os.listdir(gpt_input_batch_dir)
        if os.path.isfile(os.path.join(gpt_input_batch_dir, f))
    ])
    jobs = migrate_legacy_recovery(load_jobs())

    logger.info(f"Submitting any missing batches out of {batch_amt} total")
    jobs = submit_missing_batches(jobs, gpt_input_batch_dir, prefix, batch_amt)

    logger.info(f"Monitoring {len(jobs)} submitted batches")
    poll_until_done(jobs, gpt_input_batch_dir, prefix, output_dir)


if __name__ == "__main__":
    run()
