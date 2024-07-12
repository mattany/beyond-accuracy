import json
from pprint import pprint
import re


def extract_number(string):
    # Use a regular expression to search for a number in the string
    match = re.search(r'\d+', string)

    if match:
        # If a number is found, return it as an integer
        return int(match.group())
    else:
        # If no number is found, return None
        return None

res = {}
with open("archive/version_0/outputs/output_2.jsonl", "r") as f:
    for i, line in enumerate(f.readlines()):
        json_line = json.loads(line)
        try:
            as_dict = json.loads(json_line["response"]["body"]["choices"][0]["message"]["content"])
            as_dict = {extract_number(k): v for k, v in as_dict.items()}
            res = {**res, **as_dict}
        except json.decoder.JSONDecodeError:
            print(f"error reading line {i}")
with open("archive/version_0/outputs/output_dict_2.jsonl", "w") as f:
    f.write(json.dumps(res))