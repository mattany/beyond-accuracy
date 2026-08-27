import transformers
import torch
from transformers import BitsAndBytesConfig
from transformers import AutoModelForCausalLM, AutoTokenizer
# from huggingface_hub import login
from deepeval.models import DeepEvalBaseLLM

# from evaluation.rubrics.settings import HF_TOKEN
from mlx_lm import load, generate
# token = HF_TOKEN
# login(token)

class MLXModel(DeepEvalBaseLLM):
    def __init__(self, model_name="mlx-community/Meta-Llama-3-8B-Instruct-4bit"):
        # quantization_config = BitsAndBytesConfig(
        #     load_in_4bit=True,
        #     bnb_4bit_compute_dtype=torch.float16,
        #     bnb_4bit_quant_type="nf4",
        #     bnb_4bit_use_double_quant=True,
        # )
        model_4bit, tokenizer = load(model_name)
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

        # pipeline = transformers.pipeline(
        #     "text-generation",
        #     model=model,
        #     tokenizer=self.tokenizer,
        #     use_cache=True,
        #     device_map="auto",
        #     max_length=2500,
        #     do_sample=True,
        #     top_k=5,
        #     num_return_sequences=1,
        #     eos_token_id=self.tokenizer.eos_token_id,
        #     pad_token_id=self.tokenizer.eos_token_id,
        # )

        return generate(model, self.tokenizer, prompt, max_tokens=512, verbose=True)

    async def a_generate(self, prompt: str) -> str:
        return self.generate(prompt)

    def get_model_name(self):
        return " ".join(self._model_name.split("/")[-1].split("-"))
