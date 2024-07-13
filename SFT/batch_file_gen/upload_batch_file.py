from openai import OpenAI
from config import OPENAI_ORG_ID, OPENAI_PROJECT_ID, OPENAI_API_KEY

# Initialize OpenAI client with organization ID, project ID, and API key
client = OpenAI(
    organization=OPENAI_ORG_ID,
    project=OPENAI_PROJECT_ID,
    api_key=OPENAI_API_KEY
)


def upload_batch_file(path_to_batch_file="./batch_file_test.jsonl"):
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
def get_batch_status(batch_id):
    """
    Retrieves the status of a batch process.

    Args:
    batch_id (str): The ID of the batch to retrieve status for.

    Returns:
    dict: The status information of the batch.
    """
    return client.batches.retrieve(batch_id)



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

