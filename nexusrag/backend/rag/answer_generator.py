from typing import List, Tuple
from nexusrag.backend.retrieval.retriever import RetrievalResult
from nexusrag.backend.llm.base import BaseLLMProvider, LLMResult
from .prompt_builder import PromptBuilder

class AnswerGenerator:
    def __init__(self, llm_provider: BaseLLMProvider):
        self.llm_provider = llm_provider

    def generate(self, query: str, evidence: List[RetrievalResult]) -> Tuple[str, LLMResult]:
        system_prompt = PromptBuilder.build_system_prompt()
        user_prompt = PromptBuilder.build_user_prompt(query, evidence)

        result = self.llm_provider.generate(system_prompt, user_prompt)
        
        # If external LLM failed, fallback to local grounded synthesis
        if result.error_message or not result.text:
            from nexusrag.backend.llm.provider import OfflineDeterministicLLMProvider
            fallback_provider = OfflineDeterministicLLMProvider()
            fallback_res = fallback_provider.generate(system_prompt, user_prompt)
            fallback_res.error_message = result.error_message
            return fallback_res.text, fallback_res

        return result.text, result
