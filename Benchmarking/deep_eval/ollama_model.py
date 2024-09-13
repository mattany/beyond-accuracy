import transformers
import torch
from ollama import AsyncClient
from transformers import BitsAndBytesConfig
from transformers import AutoModelForCausalLM, AutoTokenizer
# from huggingface_hub import login
from deepeval.models import DeepEvalBaseLLM

# from config import HF_TOKEN
# from mlx_lm import load, generate
# token = HF_TOKEN
# login(token)
import ollama


class OllamaModel(DeepEvalBaseLLM):
    def __init__(self, model_name="llama3.1:8b", system_prompt="", stream=True, visual=True):
        self.model_name = model_name
        self._system_prompt = system_prompt
        self._stream = stream
        self._visual = visual
    def load_model(self):
        return

    def generate(self, prompt: str) -> str:
        # messages = [
        #     {
        #         'role': 'user',
        #         'content': prompt,
        #     },
        # ]
        # if self._system_prompt:
        #     messages = [
        #         {
        #             'role': 'system',
        #             'content': self._system_prompt,
        #         },
        #     ] + messages
        stream = ollama.generate(model=self.model_name,
                                 options={
                                     "temperature": 0.0,  # Set temperature to 0 for deterministic generation
                                     "top_k": 1,  # Only consider the top 1 probable token
                                     "top_p": 0,  # Disable nucleus sampling
                                     "seed": 42  # Set a random seed for deterministic output
                                 }, system=self._system_prompt, prompt=prompt, stream=self._stream)
        chunks = []

        for chunk in stream:
            if self._visual:
                print(chunk['response'], end='', flush=True)
            chunks.append(chunk['response'])
        if self._visual:
            print("\n=========================== END OF GENERATION ===========================")
        return ''.join(chunks)

    async def a_generate(self, prompt: str) -> str:
        generation = await AsyncClient().generate(model=self.model_name,
                                     options={
                                         "temperature": 0.0,  # Set temperature to 0 for deterministic generation
                                         "top_k": 1,  # Only consider the top 1 probable token
                                         "top_p": 0,  # Disable nucleus sampling
                                         "seed": 42,  # Set a random seed for deterministic output
                                         "num_predict": 1024
                                     }, system=self._system_prompt, prompt=prompt)
        return generation['response']
    def get_model_name(self):
        return self.model_name
