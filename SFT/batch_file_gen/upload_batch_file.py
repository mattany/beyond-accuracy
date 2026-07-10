import json
import os
import os.path
from time import sleep

from openai import NotFoundError
from tqdm import tqdm

from SFT.batch_file_gen.constants import GPT_OUTPUT_DIR, GPT_OUTPUT_FILE_PREFIX, GPT_INPUT_BATCH_DIR, \
    GPT_INPUT_BATCH_PREFIX, JOBS_PATH, RECOVERY_PATH, COMPLETION_WINDOW, get_client
import logging

POLL_INTERVAL_SECONDS = 30
# "expired"/"failed" are true failures with no usable output -> resubmit the whole batch.
# "cancelled" (e.g. from a manual cancel) still has an output file for whatever completed
# before cancellation, so it's handled like "completed" rather than blindly resubmitted.
TERMINAL_FAILURE_STATUSES = {"failed", "expired"}
DONE_STATUSES = {"completed", "cancelled"}
IN_FLIGHT_STATUSES = {"validating", "in_progress", "finalizing", "cancelling"}

class TqdmLoggingHandler(logging.Handler):
    """Emits log records via tqdm.write so they don't break the progress bar's line."""

    def emit(self, record):
        try:
            tqdm.write(self.format(record))
        except Exception:
            self.handleError(record)


handler = TqdmLoggingHandler()
handler.setFormatter(logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s'))

logger = logging.getLogger("main_logger")
logger.setLevel(logging.INFO)
logger.addHandler(handler)
logger.propagate = False
client = get_client()


def upload_batch_file(path_to_batch_file):
    with open(path_to_batch_file, "rb") as f:
        res = client.files.create(file=f, purpose="batch")
    return res.id


def create_batch(file_id):
    res = client.batches.create(
        input_file_id=file_id,
        endpoint="/v1/chat/completions",
        completion_window=COMPLETION_WINDOW,
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
        batch_info = retrieve_batch(batch_id)
        jobs[batch_index] = {
            "batch_id": batch_id,
            "file_id": file_id,
            "status": "submitted",
            "downloaded": False,
            "created_at": getattr(batch_info, "created_at", None),
        }
        save_jobs(jobs)
        logger.info(f"Submitted batch {batch_index} as {batch_id}")
    return jobs


def resubmit_batch(jobs, batch_index, gpt_input_batch_dir, prefix):
    path = f"{gpt_input_batch_dir}/{prefix}{batch_index}.jsonl"
    logger.warning(f"Resubmitting batch {batch_index} from {path}")
    file_id = upload_batch_file(path)
    batch_id = create_batch(file_id)
    batch_info = retrieve_batch(batch_id)
    jobs[batch_index] = {
        "batch_id": batch_id,
        "file_id": file_id,
        "status": "submitted",
        "downloaded": False,
        "created_at": getattr(batch_info, "created_at", None),
    }
    save_jobs(jobs)
    return batch_id


def total_processed_and_count(jobs):
    """Sums processed (completed+failed) and total requests across all known jobs."""
    processed = 0
    total = 0
    for job in jobs.values():
        counts = job.get("counts")
        if not counts:
            continue
        processed += counts["completed"] + counts["failed"]
        total += counts["total"]
    return processed, total


def anchor_created_at(jobs):
    """Second-earliest created_at across jobs, since the very first batch (batch 0)
    was submitted separately/earlier than the rest of the run."""
    timestamps = sorted(job["created_at"] for job in jobs.values() if job.get("created_at"))
    if len(timestamps) >= 2:
        return timestamps[1]
    return timestamps[0] if timestamps else None


def poll_until_done(jobs, gpt_input_batch_dir, prefix, output_dir):
    pbar = tqdm(desc="Batch requests completed", unit="req")
    anchored = False
    try:
        while True:
            pending = []
            for batch_index, job in sorted(jobs.items()):
                if job.get("downloaded") and os.path.exists(output_path(output_dir, batch_index)):
                    continue
                pending.append(batch_index)

            if not pending:
                logger.info("All batches downloaded.")
                return

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
                previous_status = job.get("status")
                job["status"] = status
                if getattr(batch_info, "created_at", None):
                    job["created_at"] = batch_info.created_at
                counts = getattr(batch_info, "request_counts", None)
                if counts is not None:
                    job["counts"] = {
                        "completed": counts.completed,
                        "failed": counts.failed,
                        "total": counts.total,
                    }

                # Anchor the progress bar's clock to when the earliest batch was
                # actually created, so rate/ETA reflect real processing time on
                # OpenAI's side instead of when this script happened to start.
                if not anchored:
                    start_t = anchor_created_at(jobs)
                    if start_t is not None:
                        pbar.start_t = start_t
                        pbar.last_print_t = start_t
                        anchored = True

                if status in IN_FLIGHT_STATUSES:
                    if counts is not None:
                        logger.info(
                            f"batch {batch_index} {status}: "
                            f"{counts.completed}/{counts.total} completed, {counts.failed} failed"
                        )
                    elif status != previous_status:
                        logger.info(f"batch {batch_index} is {status}")
                elif status in DONE_STATUSES:
                    if not job.get("downloaded") or not os.path.exists(output_path(output_dir, batch_index)):
                        if batch_info.output_file_id:
                            download_batch_output(batch_info, batch_index, output_dir)
                        else:
                            logger.warning(
                                f"batch {batch_index} ended as {status} with no output file "
                                f"(0 completed requests); nothing to download"
                            )
                    job["downloaded"] = True
                    if status == "cancelled":
                        counts = job.get("counts", {})
                        logger.warning(
                            f"batch {batch_index} was cancelled: "
                            f"{counts.get('completed', 0)}/{counts.get('total', '?')} requests kept, "
                            f"remaining stragglers treated as missing"
                        )
                    else:
                        logger.info(f"batch {batch_index} success")
                elif status in TERMINAL_FAILURE_STATUSES:
                    logger.warning(f"batch {batch_index} ended with status {status}; resubmitting")
                    resubmit_batch(jobs, batch_index, gpt_input_batch_dir, prefix)
                else:
                    logger.info(f"batch {batch_index} has unexpected status {status}")

                save_jobs(jobs)
                # Keep legacy single-line recovery roughly up to date for the first pending job.
                persist_legacy_status(pending[0], jobs[pending[0]]["status"], jobs[pending[0]]["batch_id"])

                processed, total = total_processed_and_count(jobs)
                pbar.total = total
                pbar.n = processed
                pbar.refresh()

            sleep(POLL_INTERVAL_SECONDS)
    finally:
        pbar.close()


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
