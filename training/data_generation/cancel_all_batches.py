"""Cancels every non-terminal batch we know about, so a fresh run can start clean.

Sources batch ids from training/data_generation/batch_jobs.json (written by
upload_batch_file.py) and
also cross-checks against the live batch list from OpenAI, in case any batch
was created outside of that file.
"""
import json
import os

from openai import NotFoundError

from training.data_generation.constants import JOBS_PATH, RECOVERY_PATH, get_client

TERMINAL_STATUSES = {"completed", "failed", "cancelled", "expired"}

client = get_client()


def load_known_batch_ids():
    ids = set()
    if os.path.exists(JOBS_PATH):
        with open(JOBS_PATH) as f:
            jobs = json.load(f)
        for job in jobs.values():
            batch_id = job.get("batch_id")
            if batch_id:
                ids.add(batch_id)
    return ids


def cancel_batch(batch_id):
    try:
        info = client.batches.retrieve(batch_id)
    except NotFoundError:
        print(f"{batch_id}: not found, skipping")
        return
    if info.status in TERMINAL_STATUSES:
        print(f"{batch_id}: already {info.status}, skipping")
        return
    client.batches.cancel(batch_id)
    print(f"{batch_id}: cancel requested (was {info.status})")


def main():
    batch_ids = load_known_batch_ids()

    # Also sweep the live list, in case some batches aren't tracked locally.
    after = None
    while True:
        page = client.batches.list(after=after, limit=100) if after else client.batches.list(limit=100)
        for batch in page.data:
            batch_ids.add(batch.id)
        if not getattr(page, "has_more", False):
            break
        after = page.data[-1].id

    print(f"Found {len(batch_ids)} known batch ids")
    for batch_id in sorted(batch_ids):
        cancel_batch(batch_id)

    for path in (JOBS_PATH, RECOVERY_PATH):
        if os.path.exists(path):
            os.remove(path)
            print(f"Removed {path}")


if __name__ == "__main__":
    main()
