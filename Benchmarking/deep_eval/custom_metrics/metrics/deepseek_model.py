"""DeepSeek model wrapper for deepeval."""
import os
from openai import OpenAI, AsyncOpenAI
from deepeval.models import DeepEvalBaseLLM


class DeepSeekModel(DeepEvalBaseLLM):
    """DeepSeek model using OpenAI-compatible API with lazy client initialization."""
    
    def __init__(self, model_name: str = "deepseek-reasoner"):
        """
        Args:
            model_name: "deepseek-reasoner" (slow, chain-of-thought) or 
                       "deepseek-chat" (fast, standard chat model)
        """
        self._model_name = model_name
        self._client = None
        self._async_client = None
    
    @property
    def client(self) -> OpenAI:
        if self._client is None:
            self._client = OpenAI(
                api_key=os.environ.get("DEEPSEEK_API_KEY"),
                base_url="https://api.deepseek.com"
            )
        return self._client
    
    @property
    def async_client(self) -> AsyncOpenAI:
        if self._async_client is None:
            self._async_client = AsyncOpenAI(
                api_key=os.environ.get("DEEPSEEK_API_KEY"),
                base_url="https://api.deepseek.com"
            )
        return self._async_client
    
    def load_model(self):
        return self.client
    
    def generate(self, prompt: str) -> str:
        try:
            response = self.client.chat.completions.create(
                model=self._model_name,
                messages=[{"role": "user", "content": prompt}],
                max_tokens=2048
            )
            return response.choices[0].message.content or ""
        except Exception as e:
            print(f"DeepSeek API error: {e}")
            raise
    
    async def a_generate(self, prompt: str) -> str:
        try:
            response = await self.async_client.chat.completions.create(
                model=self._model_name,
                messages=[{"role": "user", "content": prompt}],
                max_tokens=2048
            )
            return response.choices[0].message.content or ""
        except Exception as e:
            print(f"DeepSeek async API error: {e}")
            raise
    
    def get_model_name(self) -> str:
        return self._model_name


# Pre-configured instances (lazy initialization)
deepseek_reasoner = DeepSeekModel("deepseek-reasoner")  # Slow but powerful reasoning
deepseek_chat = DeepSeekModel("deepseek-chat")          # Fast standard chat

