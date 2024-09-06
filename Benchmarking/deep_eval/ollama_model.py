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
    def __init__(self, model_name="llama3.1:8b", system_prompt="", stream=True):
        self.model_name = model_name
        self._system_prompt = system_prompt

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
                                 }, system=self._system_prompt, prompt=prompt)
        chunks = []
        for chunk in stream:
            print(chunk['response'], end='', flush=True)
            chunks.append(chunk['response'])
        print("\n=========================== END OF GENERATION ===========================")
        return ''.join(chunks)

    async def a_generate(self, prompt: str) -> str:
        generation = await AsyncClient().generate(model=self.model_name,
                                     options={
                                         "temperature": 0.0,  # Set temperature to 0 for deterministic generation
                                         "top_k": 1,  # Only consider the top 1 probable token
                                         "top_p": 0,  # Disable nucleus sampling
                                         "seed": 42  # Set a random seed for deterministic output
                                     }, system=self._system_prompt, prompt=prompt)
        return generation['response']
    def get_model_name(self):
        return self.model_name



"""Let's start by considering a simple scenario. Imagine you're at the beach on a sunny day, and you
look up at the sky. It appears blue, right? But have you ever wondered why that is? The reason lies
in something called light scattering.

When sunlight enters Earth's atmosphere, it encounters tiny molecules of gases such as nitrogen and
oxygen. These molecules are like tiny balls bouncing around, and they scatter the light in all
directions. Now, here's the important part: shorter (blue) wavelengths of light are scattered more
than longer (red) wavelengths. This is because the smaller blue wavelengths are more easily
deflected by the gas molecules. As a result, our eyes perceive the sky as blue because we're seeing
the scattered blue light from all directions.

The scattering effect becomes even more pronounced when you consider the vast number of particles in
the atmosphere. It's like a giant game of cosmic billiards, where the tiny gas molecules act as the
cue balls, bouncing off each other and scattering the light in every direction. This is why the sky
appears blue during the daytime, especially in the morning and late afternoon when the sun is lower
in the sky. So, to summarize, the combination of sunlight, atmospheric gases, and light scattering
all come together to give us that beautiful blue sky we love so much!"""