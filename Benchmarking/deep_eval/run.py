from deepeval import assert_test
from deepeval.test_case import LLMTestCase
from deepeval.metrics import AnswerRelevancyMetric
import textwrap
from llama3_1_8b import CustomLlama3_8B

custom_llm = CustomLlama3_8B()
generation = custom_llm.generate("""System Prompt: You are tasked with writing high-quality scientific answers, given these criteria:
1. The explanation should have a structured flow from simple to complex concepts.
2. Establish clear connections between various parts of the explanation.
3. Assume the reader has minimal prior knowledge.
4. Usage of didactic tools such as examples, metaphors, analogy, and humor is encouraged.
5. If possible, try to paint mental images that will stay with the reader. e.g. "Consider each computer as a node and the Internet as a web."
6. Avoid domain specific jargon and unfamiliar concepts.
7. Ensure the language is unambiguous, concise, and with clearly defined terminology.
8. Use of paragraphs will be preferred over bullet points and lists. 

The answers should be around two to three paragraphs long.
Prompt: will the cables laying along the bottom of the sea ever deteriorate or even decompose?
""")
print("###################################################")
wrapper = textwrap.TextWrapper(width=100, replace_whitespace=False, drop_whitespace=False)

wrapped_text = "\n".join([wrapper.fill(line) for line in generation.splitlines()])

print(wrapped_text)


# def test_answer_relevancy():
#     answer_relevancy_metric = AnswerRelevancyMetric(threshold=0.5)
#     test_case = LLMTestCase(
#         input="What if these shoes don't fit?",
#         # Replace this with the actual output of your LLM application
#         actual_output="We offer a 30-day full refund at no extra cost."
#     )
#     assert_test(test_case, [answer_relevancy_metric])