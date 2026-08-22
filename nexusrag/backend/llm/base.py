from dataclasses import dataclass
from typing import Optional, Dict, Any

@dataclass
class LLMResult:
    text: str
    provider_name: str
    model_name: str
    tokens_used: int = 0
    is_fallback: bool = False
    error_message: Optional[str] = None

class BaseLLMProvider:
    def generate(self, system_prompt: str, user_prompt: str) -> LLMResult:
        raise NotImplementedError
