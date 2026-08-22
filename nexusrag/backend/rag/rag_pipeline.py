import os
from dataclasses import dataclass, field, asdict
from typing import List, Dict, Any, Optional

from nexusrag.backend.retrieval.retriever import VectorRetriever, RetrievalResult
from nexusrag.backend.llm.base import BaseLLMProvider, LLMResult
from nexusrag.backend.llm.provider import LLMProviderFactory
from .citation_parser import CitationParser, Citation
from .confidence import ConfidenceEstimator, ConfidenceScore
from .answer_generator import AnswerGenerator

@dataclass
class RAGAnswer:
    query: str
    answer: str
    confidence: ConfidenceScore
    citations: List[Citation]
    evidence: List[RetrievalResult]
    is_abstention: bool = False
    has_conflict: bool = False
    llm_provider: str = "Offline Grounded Engine"
    llm_model: str = "local-grounded-engine"
    tokens_used: int = 0
    error_message: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        return {
            "query": self.query,
            "answer": self.answer,
            "confidence": self.confidence.to_dict(),
            "citations": [c.to_dict() for c in self.citations],
            "evidence": [e.__dict__ for e in self.evidence],
            "is_abstention": self.is_abstention,
            "has_conflict": self.has_conflict,
            "llm_provider": self.llm_provider,
            "llm_model": self.llm_model,
            "tokens_used": self.tokens_used,
            "error_message": self.error_message
        }

class RAGPipeline:
    def __init__(
        self,
        retriever: VectorRetriever,
        llm_provider: Optional[BaseLLMProvider] = None,
        similarity_threshold: float = 0.15
    ):
        self.retriever = retriever
        self.llm_provider = llm_provider or LLMProviderFactory.create("offline")
        self.similarity_threshold = similarity_threshold
        self.answer_generator = AnswerGenerator(self.llm_provider)

    def run(self, query: str, top_k: int = 5) -> RAGAnswer:
        # 1. Query Validation
        clean_query = query.strip() if query else ""
        if not clean_query:
            return RAGAnswer(
                query=query,
                answer="Please provide a valid question to search your indexed documents.",
                confidence=ConfidenceScore("Low", 0, 0.0, 0, False, "Empty or invalid query."),
                citations=[],
                evidence=[],
                is_abstention=True,
                llm_provider="Validation",
                llm_model="None"
            )

        if len(clean_query) < 3:
            return RAGAnswer(
                query=query,
                answer="Question is too short. Please enter a more specific question.",
                confidence=ConfidenceScore("Low", 0, 0.0, 0, False, "Query under minimum character length."),
                citations=[],
                evidence=[],
                is_abstention=True,
                llm_provider="Validation",
                llm_model="None"
            )

        # 2. Check if documents are indexed
        if self.retriever.vector_store.get_total_vectors() == 0:
            return RAGAnswer(
                query=clean_query,
                answer="No indexed documents are available. Upload and index documents in the Documents page before asking questions.",
                confidence=ConfidenceScore("Low", 0, 0.0, 0, False, "Vector store is empty."),
                citations=[],
                evidence=[],
                is_abstention=True,
                llm_provider="Validation",
                llm_model="None"
            )

        # 3. Semantic Retrieval
        retrieved_chunks = self.retriever.retrieve(clean_query, top_k=top_k)

        # 4. Evidence Filtering & Abstention Check
        filtered_chunks = [c for c in retrieved_chunks if c.similarity_score >= self.similarity_threshold]

        if not filtered_chunks:
            return RAGAnswer(
                query=clean_query,
                answer="I couldn't find sufficient evidence in the uploaded documents to answer this question.",
                confidence=ConfidenceScore("Low", 15, retrieved_chunks[0].similarity_score if retrieved_chunks else 0.0, 0, False, "Evidence similarity score below required threshold."),
                citations=[],
                evidence=retrieved_chunks[:2] if retrieved_chunks else [],
                is_abstention=True,
                llm_provider="Abstention Filter",
                llm_model="None"
            )

        # 5. Check for Cross-Document Version Conflicts in Evidence
        has_conflict = self._detect_potential_conflict(filtered_chunks, clean_query)

        # 6. LLM Grounded Answer Generation
        answer_text, llm_res = self.answer_generator.generate(clean_query, filtered_chunks)

        # 7. Citation Parsing
        citations = CitationParser.extract_citations(answer_text, filtered_chunks)

        # 8. Confidence Estimation
        confidence = ConfidenceEstimator.estimate_confidence(
            evidence=filtered_chunks,
            is_abstention=False,
            has_conflict=has_conflict
        )

        return RAGAnswer(
            query=clean_query,
            answer=answer_text,
            confidence=confidence,
            citations=citations,
            evidence=filtered_chunks,
            is_abstention=False,
            has_conflict=has_conflict,
            llm_provider=llm_res.provider_name,
            llm_model=llm_res.model_name,
            tokens_used=llm_res.tokens_used,
            error_message=llm_res.error_message
        )

    def _detect_potential_conflict(self, chunks: List[RetrievalResult], query: str) -> bool:
        # Detect if retrieved chunks span multiple versions or years for the same department/topic
        versions = set(c.version for c in chunks if c.version)
        years = set(c.year for c in chunks if c.year)
        return len(versions) > 1 or len(years) > 1
