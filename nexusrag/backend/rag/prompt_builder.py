from typing import List
from nexusrag.backend.retrieval.retriever import RetrievalResult
from .context_builder import ContextBuilder

class PromptBuilder:
    SYSTEM_PROMPT = """You are NexusRAG, an Evidence-First Enterprise Knowledge Intelligence Assistant.
Your core mission is to provide accurate, grounded answers strictly derived from the provided retrieved document sources.

STRICT GROUNDING RULES:
1. Answer ONLY using the explicit facts present in the RETRIEVED SOURCES.
2. Do NOT extrapolate, assume, or use outside/pre-trained knowledge.
3. For EVERY factual claim or statement, you MUST attach bracketed citations corresponding to the source number, e.g., [1], [2].
4. If different sources contain conflicting rules or different versions (e.g. Version 1.0 vs Version 2.0), EXPLICITLY state the conflict and distinguish between the document years/versions.
5. If the retrieved evidence does not contain sufficient information to answer the user's question, you MUST clearly state:
   "I couldn't find sufficient evidence in the uploaded documents to answer this question."
6. Never fabricate source numbers, pages, or document names."""

    @staticmethod
    def build_system_prompt() -> str:
        return PromptBuilder.SYSTEM_PROMPT

    @staticmethod
    def build_user_prompt(query: str, evidence: List[RetrievalResult]) -> str:
        context_text = ContextBuilder.build_context(evidence)
        return (
            f"QUESTION: {query}\n\n"
            f"RETRIEVED DOCUMENT SOURCES:\n"
            f"{context_text}\n\n"
            f"INSTRUCTIONS:\n"
            f"Provide a clear, concise, and grounded response answering the question with inline citations like [1], [2]. "
            f"If sources conflict across document versions, clearly highlight the difference. "
            f"If the sources do not contain the answer, state that insufficient evidence was found."
        )
