import os.path
from time import sleep

from openai import OpenAI

from SFT.batch_file_gen.constants import GPT_OUTPUT_DIR, GPT_OUTPUT_FILE_PREFIX, GPT_INPUT_BATCH_DIR, \
    GPT_INPUT_BATCH_PREFIX
from SFT.batch_file_gen.config import OPENAI_ORG_ID, OPENAI_PROJECT_ID, OPENAI_API_KEY, PROJECT_DIR
import logging

RECOVERY_PATH = f"{PROJECT_DIR}/SFT/batch_status.txt"
logging.basicConfig(level=logging.INFO,
                    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')

logger = logging.getLogger("main_logger")
# Initialize OpenAI client with organization ID, project ID, and API key
client = OpenAI(
    organization=OPENAI_ORG_ID,
    project=OPENAI_PROJECT_ID,
    api_key=OPENAI_API_KEY
)


def upload_batch_file(path_to_batch_file):
    """
    Uploads a batch file to OpenAI.

    Args:
    path_to_batch_file (str): The path to the batch file to be uploaded.

    Returns:
    str: The ID of the uploaded file.
    """
    res = client.files.create(
        file=open(path_to_batch_file, "rb"),
        purpose="batch"
    )
    return res.id



def create_batch(file_id):
    """
    Creates a batch process using the uploaded file ID.

    Args:
    file_id (str): The ID of the uploaded file.

    Returns:
    str: The ID of the created batch.
    """
    res = client.batches.create(
        input_file_id=file_id,
        endpoint="/v1/chat/completions",
        completion_window="24h"
    )
    return res.id


def upload_and_create(path_to_batch_file):
    """
    Uploads a batch file and creates a batch process.

    Args:
    path_to_batch_file (str): The path to the batch file to be uploaded.

    Returns:
    str: The ID of the created batch.
    """
    file_id = upload_batch_file(path_to_batch_file)
    batch_id = create_batch(file_id=file_id)
    return batch_id
#
def retrieve_batch(batch_id):
    """
    Retrieves the status of a batch process.

    Args:
    batch_id (str): The ID of the batch to retrieve status for.

    Returns:
    dict: The status information of the batch.
    """
    return client.batches.retrieve(batch_id)


def upload_create_and_monitor_batch(path_to_batch_file, batch_index, batch_id, output_dir=GPT_OUTPUT_DIR):
    batch_number = path_to_batch_file.split("_")[-1].split(".")[0]
    file_id = upload_batch_file(path_to_batch_file)
    batch_id = batch_id or create_batch(file_id=file_id)
    success = False
    while not success:
        batch_info = retrieve_batch(batch_id)
        batch_status = batch_info.status
        if batch_status in {"validating", "in_progress", "finalizing"}:
            logger.info(f"batch {batch_number} is {batch_status}. Waiting 5 seconds.")
            persist_batch_status_to_disk(batch_number, batch_status, batch_id)
            sleep(5)
        elif batch_status in {"failed", "cancelling", "cancelled", "expired"}:
            logger.warning(f"batch {batch_number} failed. Waiting 60 seconds and creating new batch.")
            persist_batch_status_to_disk(batch_number, batch_status, batch_id)
            sleep(60)
            batch_id = create_batch(file_id=file_id)
        elif batch_status in {"completed"}:
            success = True
            persist_batch_status_to_disk(batch_number, batch_status, batch_id)
            output_file_id = batch_info.output_file_id
            file_response = client.files.content(output_file_id)
            os.makedirs(output_dir, exist_ok=True)
            with open(os.path.join(output_dir, f"{GPT_OUTPUT_FILE_PREFIX}{batch_index}.jsonl"), "w") as f:
                f.write(file_response.text)
            logger.info(f"batch {batch_number} success")


def persist_batch_status_to_disk(batch_number, batch_status, batch_id):
    with open(RECOVERY_PATH, "w") as f:
        f.write(f"{batch_status} {batch_number} {batch_id}\n")


def run(gpt_input_batch_dir=GPT_INPUT_BATCH_DIR, prefix=GPT_INPUT_BATCH_PREFIX, output_dir=GPT_OUTPUT_DIR):

    batch_amt = len([f for f in os.listdir(gpt_input_batch_dir) if os.path.isfile(os.path.join(gpt_input_batch_dir, f))])
    batch_id = ""
    batch_number = 0
    if os.path.exists(RECOVERY_PATH):
        with open(RECOVERY_PATH, "r") as f:
            line = f.readline()
            batch_status, batch_number, batch_id = line.split()
            batch_number = int(batch_number)
    arguments = [f"{gpt_input_batch_dir}/{prefix}{batch_index}.jsonl" for batch_index in range(batch_number, batch_amt, 1)]
    for i, arg in enumerate(arguments):
        if i == 0:
            upload_create_and_monitor_batch(arg, batch_index=batch_number, batch_id=batch_id, output_dir=output_dir)
        else:
            upload_create_and_monitor_batch(arg, batch_index=batch_number + i, batch_id=None, output_dir=output_dir)



if __name__ == "__main__":
    run()