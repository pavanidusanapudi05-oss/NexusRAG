import re
from dataclasses import dataclass, field
from typing import List, Dict, Any, Optional
from nexusrag.backend.reranking.reranker import RerankedEvidence
from .llm_client import LLMClientAdapter

@dataclass
class GroundedAnswer:
    answer: str
    confidence_score: float
    confidence_label: str # High, Medium, Low, Insufficient Evidence
    evidence_cards: List[Dict[str, Any]]
    routed_intent: str
    query_context: str
    is_abstention: bool = False

class EvidenceFirstGenerator:
    def __init__(self, llm_adapter: Optional[LLMClientAdapter] = None, abstention_threshold: float = 0.15):
        self.llm_adapter = llm_adapter or LLMClientAdapter()
        self.abstention_threshold = abstention_threshold

    def generate_answer(
        self,
        query: str,
        evidence: List[RerankedEvidence],
        routed_intent: str = "general_qa",
        graph_context: Optional[str] = None
    ) -> GroundedAnswer:
        # Check for empty or weak evidence
        if not evidence or (evidence and evidence[0].relevance_score < self.abstention_threshold):
            return GroundedAnswer(
                answer="I could not find sufficient evidence in the available documents to answer this reliably.",
                confidence_score=0.1,
                confidence_label="Insufficient Evidence",
                evidence_cards=[],
                routed_intent=routed_intent,
                query_context=query,
                is_abstention=True
            )

        # Build structured evidence context
        context_parts = []
        evidence_cards = []

        for rank, ev in enumerate(evidence):
            c = ev.chunk
            card = {
                "rank": rank + 1,
                "doc_name": c.doc_name,
                "version": c.version,
                "year": c.year,
                "department": c.department,
                "page_number": c.page_number,
                "section_title": c.section_title,
                "relevance_score": ev.relevance_score,
                "matched_terms": ev.matched_terms,
                "excerpt": c.content
            }
            evidence_cards.append(card)

            context_parts.append(
                f"[EVIDENCE {rank+1}]\n"
                f"Document: {c.doc_name} (Version {c.version}, Year {c.year})\n"
                f"Section: {c.section_title} | Page: {c.page_number}\n"
                f"Department: {c.department}\n"
                f"Content:\n{c.content}\n"
            )

        evidence_str = "\n---\n".join(context_parts)
        kg_str = f"\nKNOWLEDGE GRAPH CONTEXT:\n{graph_context}\n" if graph_context else ""

        system_instruction = (
            "You are NexusRAG, an evidence-first enterprise knowledge intelligence system. "
            "Your answers MUST be strictly grounded in the provided evidence. "
            "Always include citations in the format [Document Name, Page X, Section Y] for every claim. "
            "If evidence is insufficient or conflicting, explicitly state it. "
            "Never invent facts, document names, or page numbers."
        )

        prompt = (
            f"USER QUERY: {query}\n\n"
            f"INTENT: {routed_intent}\n"
            f"{kg_str}\n"
            f"=== RETRIEVED EVIDENCE ===\n"
            f"{evidence_str}\n\n"
            f"Provide a clear, grounded, and concise answer with explicit citations."
        )

        raw_answer = self.llm_adapter.generate(prompt, system_instruction=system_instruction)

        # Calculate confidence
        top_score = evidence[0].relevance_score if evidence else 0.0
        avg_score = sum(e.relevance_score for e in evidence) / len(evidence) if evidence else 0.0
        confidence = min(0.99, max(0.2, (top_score * 0.6) + (avg_score * 0.4)))

        if confidence >= 0.75:
            conf_label = "High Confidence"
        elif confidence >= 0.45:
            conf_label = "Medium Confidence"
        else:
            conf_label = "Low Confidence"

        return GroundedAnswer(
            answer=raw_answer,
            confidence_score=round(confidence, 2),
            confidence_label=conf_label,
            evidence_cards=evidence_cards,
            routed_intent=routed_intent,
            query_context=query,
            is_abstention=False
        )
