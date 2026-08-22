from dataclasses import dataclass, asdict
from typing import List, Dict, Any
from nexusrag.backend.retrieval.retriever import RetrievalResult

@dataclass
class ConfidenceScore:
    level: str             # "High", "Medium", "Low"
    score_percentage: int  # 0 to 100
    top_similarity: float
    evidence_count: int
    has_conflict: bool
    explanation: str

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

class ConfidenceEstimator:
    @staticmethod
    def estimate_confidence(
        evidence: List[RetrievalResult],
        is_abstention: bool = False,
        has_conflict: bool = False
    ) -> ConfidenceScore:
        if is_abstention or not evidence:
            return ConfidenceScore(
                level="Low",
                score_percentage=15,
                top_similarity=0.0,
                evidence_count=0,
                has_conflict=False,
                explanation="Insufficient relevant evidence found in indexed documents."
            )

        top_sim = evidence[0].similarity_score
        avg_sim = sum(e.similarity_score for e in evidence) / len(evidence)

        # Baseline percentage calculation based on similarity score (0.0 to 1.0)
        # Cosine similarity for relevant dense / TF-IDF chunks typically ranges 0.3 - 0.95
        base_pct = int(min(98, max(20, (top_sim * 0.7 + avg_sim * 0.3) * 100)))

        # Adjustments
        if len(evidence) >= 3:
            base_pct = min(98, base_pct + 5)
        if has_conflict:
            base_pct = max(50, base_pct - 10)

        if base_pct >= 75:
            level = "High"
            explanation = f"Strong evidence alignment across {len(evidence)} retrieved chunks (Top similarity: {top_sim:.4f})."
        elif base_pct >= 50:
            level = "Medium"
            explanation = f"Moderate evidence support (Top similarity: {top_sim:.4f}). Review citations for context."
        else:
            level = "Low"
            explanation = f"Weak evidence support (Top similarity: {top_sim:.4f}). Answer may be incomplete."

        if has_conflict:
            explanation += " Note: Potential policy or version discrepancies detected in evidence."

        return ConfidenceScore(
            level=level,
            score_percentage=base_pct,
            top_similarity=round(top_sim, 4),
            evidence_count=len(evidence),
            has_conflict=has_conflict,
            explanation=explanation
        )
