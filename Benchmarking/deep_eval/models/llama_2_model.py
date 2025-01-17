import transformers
import torch
from transformers import BitsAndBytesConfig
from transformers import AutoModelForCausalLM, AutoTokenizer
# from huggingface_hub import login
from deepeval.models import DeepEvalBaseLLM

from models.util import create_model_and_tokenizer


# from config import HF_TOKEN
# token = HF_TOKEN
# login(token)

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


class Llama2Model(DeepEvalBaseLLM):
    def __init__(self, model_name="meta-llama/Llama-2-7b-chat-hf"):
        # quantization_config = BitsAndBytesConfig(
        #     load_in_4bit=True,
        #     bnb_4bit_compute_dtype=torch.float16,
        #     bnb_4bit_quant_type="nf4",
        #     bnb_4bit_use_double_quant=True,
        # )
        model_4bit, tokenizer = create_model_and_tokenizer(model_name)
        # model_4bit = AutoModelForCausalLM.from_pretrained(
        #     "mlx-community/Meta-Llama-3.1-8B-4bit",
        #     device_map="auto",
        #     quantization_config=quantization_config,
        # low_cpu_mem_usage=True,
        # )
        # tokenizer = AutoTokenizer.from_pretrained(
        #     "mlx-community/Meta-Llama-3.1-8B-4bit"
        # )
        self._model_name = model_name
        self.model = model_4bit
        self.tokenizer = tokenizer

    def load_model(self):
        return self.model

    def generate(self, prompt: str) -> str:
        model = self.load_model()
        inputs = self.tokenizer(prompt, return_tensors="pt").to(DEVICE)
        inputs_length = len(inputs["input_ids"][0])
        with torch.inference_mode():
            outputs = model.generate(**inputs, max_new_tokens=256, top_p=0.9, temperature=0.6)
        return self.tokenizer.decode(outputs[0][inputs_length:], skip_special_tokens=True)

    async def a_generate(self, prompt: str) -> str:
        return self.generate(prompt)

    def get_model_name(self):
        return " ".join(self._model_name.split("/")[-1].split("-"))
