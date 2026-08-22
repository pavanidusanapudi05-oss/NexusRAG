import re
from enum import Enum
from dataclasses import dataclass
from typing import Dict, Any, List, Optional

class QueryIntent(str, Enum):
    VERSION_COMPARISON = "version_comparison"
    CROSS_DOC_COMPARISON = "cross_doc_comparison"
    CONFLICT_DETECTION = "conflict_detection"
    REGULATION_IMPACT = "regulation_impact"
    REQUIREMENT_LOOKUP = "requirement_lookup"
    GENERAL_QA = "general_qa"
    SUMMARIZATION = "summarization"

@dataclass
class RoutedQuery:
    original_query: str
    expanded_query: str
    intent: QueryIntent
    target_docs: List[str]
    requires_kg: bool
    requires_diff: bool
    confidence: float
    reasoning: str

class QueryRouter:
    @staticmethod
    def route_query(query: str, available_docs: Optional[List[str]] = None) -> RoutedQuery:
        q_lower = query.lower()
        target_docs = []
        requires_kg = False
        requires_diff = False
        
        # Detect target docs mentioned
        if available_docs:
            for doc in available_docs:
                d_stem = re.sub(r'\.[^.]+$', '', doc).lower().replace('_', ' ')
                if d_stem in q_lower or any(word in q_lower for word in d_stem.split() if len(word) > 4):
                    target_docs.append(doc)

        # 1. Version Comparison (e.g. "What changed between 2025 and 2026?", "Compare version 1 and 2", "What requirements were added?")
        if (("2025" in q_lower and "2026" in q_lower) or 
            "what changed" in q_lower or 
            "difference between" in q_lower or 
            "version comparison" in q_lower or 
            "requirements were added" in q_lower or 
            "requirements were removed" in q_lower or 
            "latest policy" in q_lower or
            "which version" in q_lower):
            return RoutedQuery(
                original_query=query,
                expanded_query=query,
                intent=QueryIntent.VERSION_COMPARISON,
                target_docs=target_docs or ["Employee_Operations_Policy_2025.pdf", "Employee_Operations_Policy_2026.pdf"],
                requires_kg=True,
                requires_diff=True,
                confidence=0.95,
                reasoning="Query compares document versions or asks for chronological policy changes."
            )

        # 2. Conflict Detection (e.g. "Are there any conflicting requirements?", "conflicts between documents")
        if "conflict" in q_lower or "contradict" in q_lower or "discrepanc" in q_lower or "inconsistent" in q_lower:
            return RoutedQuery(
                original_query=query,
                expanded_query=query,
                intent=QueryIntent.CONFLICT_DETECTION,
                target_docs=target_docs,
                requires_kg=True,
                requires_diff=False,
                confidence=0.92,
                reasoning="Query asks for requirement conflicts across enterprise documents."
            )

        # 3. Cross-Document Comparison (e.g. "Compare requirements across these three documents", "Compare Policy A and Regulation B")
        if "compare" in q_lower or "across documents" in q_lower or "across these three" in q_lower or "matrix" in q_lower:
            return RoutedQuery(
                original_query=query,
                expanded_query=query,
                intent=QueryIntent.CROSS_DOC_COMPARISON,
                target_docs=target_docs,
                requires_kg=True,
                requires_diff=False,
                confidence=0.90,
                reasoning="Query requests cross-document requirement synthesis."
            )

        # 4. Regulation Impact Analysis (e.g. "How does the regulation affect this process?", "How does SR-402 affect Process X?")
        if ("regulation affect" in q_lower or 
            "sr-402 affect" in q_lower or 
            "sr402 affect" in q_lower or 
            "impact of regulation" in q_lower or 
            "how does regulation" in q_lower or
            ("process x" in q_lower and "regulation" in q_lower)):
            return RoutedQuery(
                original_query=query,
                expanded_query=f"{query} Regulation SR-402 Process X Cybersecurity Requirements",
                intent=QueryIntent.REGULATION_IMPACT,
                target_docs=target_docs,
                requires_kg=True,
                requires_diff=False,
                confidence=0.94,
                reasoning="Query explores regulatory ripple effects across processes via Knowledge Graph."
            )

        # 5. Requirement Lookup (e.g. "Which document supports this requirement?", "Who approves remote work?")
        if "which document supports" in q_lower or "which policy supports" in q_lower or "where is it stated" in q_lower:
            return RoutedQuery(
                original_query=query,
                expanded_query=query,
                intent=QueryIntent.REQUIREMENT_LOOKUP,
                target_docs=target_docs,
                requires_kg=False,
                requires_diff=False,
                confidence=0.88,
                reasoning="Query looks for definitive document authority for a specific requirement."
            )

        # 6. Summarization
        if "summarize" in q_lower or "overview" in q_lower or "executive summary" in q_lower:
            return RoutedQuery(
                original_query=query,
                expanded_query=query,
                intent=QueryIntent.SUMMARIZATION,
                target_docs=target_docs,
                requires_kg=False,
                requires_diff=False,
                confidence=0.85,
                reasoning="Query requests high-level document summarization."
            )

        # Default: General Grounded QA
        return RoutedQuery(
            original_query=query,
            expanded_query=query,
            intent=QueryIntent.GENERAL_QA,
            target_docs=target_docs,
            requires_kg=False,
            requires_diff=False,
            confidence=0.80,
            reasoning="Standard grounded factual retrieval."
        )
