"""Claude model wrapper for deepeval."""
import os
from anthropic import Anthropic, AsyncAnthropic
from deepeval.models import DeepEvalBaseLLM


class ClaudeModel(DeepEvalBaseLLM):
    """Claude model using Anthropic API with lazy client initialization."""
    
    def __init__(self, model_name: str = "claude-sonnet-4-20250514"):
        """
        Args:
            model_name: Claude model name, e.g.:
                - "claude-sonnet-4-20250514" (Claude 4 Sonnet)
                - "claude-opus-4-20250514" (Claude 4 Opus)
                - "claude-3-5-sonnet-20241022" (Claude 3.5 Sonnet)
        """
        self._model_name = model_name
        self._client = None
        self._async_client = None
    
    @property
    def client(self) -> Anthropic:
        if self._client is None:
            self._client = Anthropic(
                api_key=os.environ.get("ANTHROPIC_API_KEY"),
            )
        return self._client
    
    @property
    def async_client(self) -> AsyncAnthropic:
        if self._async_client is None:
            self._async_client = AsyncAnthropic(
                api_key=os.environ.get("ANTHROPIC_API_KEY"),
            )
        return self._async_client
    
    def load_model(self):
        return self.client
    
    def generate(self, prompt: str) -> str:
        try:
            response = self.client.messages.create(
                model=self._model_name,
                max_tokens=2048,
                messages=[{"role": "user", "content": prompt}]
            )
            return response.content[0].text or ""
        except Exception as e:
            print(f"Claude API error: {e}")
            raise
    
    async def a_generate(self, prompt: str) -> str:
        try:
            response = await self.async_client.messages.create(
                model=self._model_name,
                max_tokens=2048,
                messages=[{"role": "user", "content": prompt}]
            )
            return response.content[0].text or ""
        except Exception as e:
            print(f"Claude async API error: {e}")
            raise
    
    def get_model_name(self) -> str:
        return self._model_name


# Pre-configured instances (lazy initialization)
claude_opus_4 = ClaudeModel("claude-opus-4-20250514")  # Claude 4 Opus
claude_sonnet_4 = ClaudeModel("claude-sonnet-4-20250514")  # Claude 4 Sonnet
claude_35_sonnet = ClaudeModel("claude-3-5-sonnet-20241022")  # Claude 3.5 Sonnet

