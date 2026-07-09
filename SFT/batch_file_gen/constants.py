from SFT.batch_file_gen.config import PROJECT_DIR

INPUT_CSV = f"{PROJECT_DIR}/SFT/data/ask_science.csv"
OUTPUT_CSV = f"{PROJECT_DIR}/SFT/data/ask_science_gpt_5_answers.csv"
GPT_INPUT_BATCH_DIR = f"{PROJECT_DIR}/SFT/data/input_batches"
GPT_OUTPUT_DIR = f"{PROJECT_DIR}/SFT/data/GPT5_outputs"
GPT_OUTPUT_FILE_PREFIX = "gpt_output_file_"
GPT_INPUT_BATCH_PREFIX = "sft_input_batch_file_"
# System prompt for generating high-quality scientific answers
SFT_SYSTEM_PROMPT = """You are tasked with writing high-quality scientific answers, given these criteria:
1. The explanation should have a structured flow from simple to complex concepts.
2. Establish clear connections between various parts of the explanation.
3. Assume the reader has minimal prior knowledge.
4. Usage of didactic tools such as examples, metaphors, analogy, and humor is encouraged.
5. If possible, try to paint mental images that will stay with the reader. e.g. "Consider each computer as a node and the Internet as a web."
6. Avoid domain specific jargon and unfamiliar concepts.
7. Ensure the language is unambiguous, concise, and with clearly defined terminology.
8. Use of paragraphs will be preferred over bullet points and lists. 

The answers should be around two to three paragraphs long.
"""