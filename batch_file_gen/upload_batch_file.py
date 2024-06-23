import time

from openai import OpenAI
from config import OPENAI_ORG_ID, OPENAI_PROJECT_ID, OPENAI_API_KEY

client = OpenAI(
    organization=OPENAI_ORG_ID,
    project=OPENAI_PROJECT_ID,
    api_key=OPENAI_API_KEY
)


def upload_batch_file(path_to_batch_file="./batch_file_test.jsonl"):
    res = client.files.create(
        file=open(path_to_batch_file, "rb"),
        purpose="batch"
    )
    return res.id



def create_batch(file_id="file-CRClQmhbgiTs2zHsr9g6omdy"):
    res = client.batches.create(
        input_file_id=file_id,
        endpoint="/v1/chat/completions",
        completion_window="24h"
    )
    return res.id


def upload_and_create(path_to_batch_file):
    file_id = upload_batch_file(path_to_batch_file)
    batch_id = create_batch(file_id=file_id)
    return batch_id
#
def get_batch_status(batch_id):
    return client.batches.retrieve(batch_id)



if __name__ == "__main__":
    # print(client.files.list())
    # print(client.batches.list())
    # print(upload_and_create(f"./batch_file_{0}.jsonl")) # TODO in progress
    print(upload_and_create(f"./batch_file_{1}.jsonl"))
    time.sleep(3600)
    print(upload_and_create(f"./batch_file_{2}.jsonl"))
    time.sleep(3600)
    print(upload_and_create(f"./batch_file_{3}.jsonl"))
