from openai import OpenAI
from config import OPENAI_ORG_ID, OPENAI_PROJECT_ID, OPENAI_API_KEY
client = OpenAI(
  organization=OPENAI_ORG_ID,
  project=OPENAI_PROJECT_ID,
  api_key=OPENAI_API_KEY
)
system_prompt = """You are a science educator tasked with writing a high-quality scientific answers to the following questions, given these criteria:
1. The explanation has a structured flow from simple to complex concepts.
2. Establish clear connections between various parts of the explanation.
3. Achieve a good balance between introduction, scientific content, examples, and conclusion.
4. Assume the reader has minimal prior knowledge.
5. Use examples.
6. Avoid jargon.
7. Ensure the language is unambiguous, concise, and with clearly defined terminology.
8. Use of paragraphs will be preferred over bullet points and lists.

Your answers should be no more than 256 tokens long.
Your response should be a json with key value pairs in the form:
question_number:answer
"""


completion = client.chat.completions.create(
  model="gpt-3.5-turbo",
  response_format={ "type": "json_object" },
  messages=[
    {"role": "system", "content": system_prompt},
    {"role": "user", "content": 'Question #6220: How can carbon dioxide be converted into carbon and oxygen?\n\nQuestion #6221: What or Who is Anatomical plane?\n\nQuestion #6222: What or Who is Graphology?\n\nQuestion #6223: "In a USB cable, is it OK to swap the D+ and D- wires?"\n\nQuestion #6224: About 6 months ago hundreds of millions of genetically modified mosquitos were released in the Florida Keys. Is there any update on how that\'s going?\n\nQuestion #6225: What or Who is Gardens by the Bay?\n\nQuestion #6226: What or Who is Arbor Day?\n\nQuestion #6227: What or Who is ALS?\n\nQuestion #6228: What or Who is Geographic coordinate conversion?\n\nQuestion #6229: "If you watch a gif of a coin flipping (without ever seeing it) to make a decision, is it still a 50/50 chance, even though the video already predetermines what side the coin will flip onto?"\n\n'}
  ]
)
print(completion.choices[0].message.content)