from openai import OpenAI
import textwrap

from SFT.batch_file_gen.gen_batch import SFT_SYSTEM_PROMPT
from config import OPENAI_API_KEY
client = OpenAI(api_key=OPENAI_API_KEY)
import time
system_prompt = SFT_SYSTEM_PROMPT

question = "When dams are being built, how do they build it with all the water still there?"

t = time.time()
completion = client.chat.completions.create(
  model="gpt-3.5-turbo",
  messages=[
    {"role": "system", "content": system_prompt},
    {"role": "user", "content": question}
  ],
  max_tokens=256
)
t = time.time() -t
print(f'Time: {t} seconds')
print(completion.usage)

def format_text(text, line_length):
    return textwrap.fill(text, width=line_length)


print(format_text(completion.choices[0].message.content, line_length=256))

