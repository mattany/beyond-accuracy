from openai import OpenAI
from config import OPENAI_ORG_ID, OPENAI_PROJECT_ID, OPENAI_API_KEY
client = OpenAI(
  organization=OPENAI_ORG_ID,
  project=OPENAI_PROJECT_ID,
  api_key=OPENAI_API_KEY
)
import time
system_prompt = """
You are tasked with writing high-quality scientific answers , given these criteria:
1. The explanation has a structured flow from simple to complex concepts.
2. Establish clear connections between various parts of the explanation.
3. Achieve a good balance between introduction, scientific content, examples, and conclusion.
4. Assume the reader has minimal prior knowledge.
5. Use examples.
6. Avoid jargon.
7. Ensure the language is unambiguous, concise, and with clearly defined terminology.
8. Use of paragraphs will be preferred over bullet points and lists. 

The answers should be short.
"""

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
print(completion.choices[0].message.content)

