from openai import OpenAI
from config import OPENAI_ORG_ID, OPENAI_PROJECT_ID, OPENAI_API_KEY, PROJECT_DIR
import asyncio
import logging


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


async def upload_create_and_monitor_batch(path_to_batch_file):
    batch_number = path_to_batch_file[-7:-6]
    file_id = await asyncio.to_thread(upload_batch_file, path_to_batch_file)
    batch_id = await asyncio.to_thread(create_batch, file_id=file_id)
    success = False
    while not success:
        batch_info = await asyncio.to_thread(retrieve_batch, batch_id)
        batch_status = batch_info.status
        if batch_status in {"validating", "in_progress", "finalizing"}:
            logger.info(f"batch {batch_number} is {batch_status}. Waiting 5 seconds.")
            asyncio.to_thread(persist_batch_status_to_disk, batch_number, batch_status)
            await asyncio.sleep(5)
        elif batch_status in {"failed", "cancelling", "cancelled", "expired"}:
            logger.warning(f"batch {batch_number} failed. Waiting 60 seconds and creating new batch.")
            asyncio.to_thread(persist_batch_status_to_disk, batch_number, batch_status)
            await asyncio.sleep(60)
            batch_id = create_batch(file_id=file_id)
        elif batch_status in {"completed"}:
            success = True
            asyncio.to_thread(persist_batch_status_to_disk, batch_number, batch_status)
            logger.info(f"batch {batch_number} success")


def persist_batch_status_to_disk(batch_number, batch_status):
    with open(f"{PROJECT_DIR}/SFT/batch_statuses/batch_{batch_number}_status.txt", "w") as f:
        f.write(batch_status)


async def main(output_dir, prefix, batch_amt):
    tasks = []
    arguments = [f"{output_dir}/{prefix}{batch_index}.jsonl" for batch_index in range(batch_amt)]
    for arg in arguments:
        tasks.append(asyncio.create_task(upload_create_and_monitor_batch(arg)))


if __name__ == "__main__":
    # Print the list of uploaded files
    print(client.files.list())

    # Print the list of batches
    print(client.batches.list())

    # Upload and create batches sequentially due to API rate limits
    # Uncomment one line at a time to upload and create batches

    # print(upload_and_create(f"batch_files/batch_file_{0}.jsonl"))
    # print(upload_and_create(f"batch_files/batch_file_{1}.jsonl"))
    # print(upload_and_create(f"batch_files/batch_file_{2}.jsonl"))
    # print(upload_and_create(f"batch_files/batch_file_{3}.jsonl"))
    # print(upload_and_create(f"batch_files/batch_file_{4}.jsonl"))
    # print(upload_and_create(f"batch_files/batch_file_{5}.jsonl"))
    # print(upload_and_create(f"batch_files/batch_file_{6}.jsonl"))
    # print(upload_and_create(f"batch_files/batch_file_{7}.jsonl"))
    # print(upload_and_create(f"batch_files/batch_file_{8}.jsonl"))
    # print(upload_and_create(f"batch_files/batch_file_{9}.jsonl"))
    # print(upload_and_create(f"batch_files/batch_file_{10}.jsonl"))
    # print(upload_and_create(f"batch_files/batch_file_{11}.jsonl"))
    # print(upload_and_create(f"batch_files/batch_file_{12}.jsonl"))
    # print(upload_and_create(f"batch_files/batch_file_{13}.jsonl"))
    # print(upload_and_create(f"batch_files/batch_file_{14}.jsonl"))
    # print(upload_and_create(f"batch_files/batch_file_{15}.jsonl"))
    # print(upload_and_create(f"batch_files/batch_file_{16}.jsonl"))

