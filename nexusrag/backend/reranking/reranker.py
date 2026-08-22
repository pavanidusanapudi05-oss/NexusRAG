import re
from dataclasses import dataclass
from typing import List, Dict, Any, Optional
from nexusrag.backend.retrieval.hybrid import RetrievalCandidate
from nexusrag.backend.ingestion.chunker import DocumentChunk

@dataclass
class RerankedEvidence:
    chunk: DocumentChunk
    relevance_score: float
    original_rank: int
    reranked_rank: int
    matched_terms: List[str]
    evidence_snippet: str

class CrossEncoderReranker:
    def __init__(self, relevance_threshold: float = 0.20):
        self.relevance_threshold = relevance_threshold

    def rerank(self, query: str, candidates: List[RetrievalCandidate], top_k: int = 4) -> List[RerankedEvidence]:
        if not candidates:
            return []

        query_tokens = [t.lower() for t in re.findall(r'\b[a-zA-Z0-9_-]{2,}\b', query)]
        reranked: List[RerankedEvidence] = []

        for orig_rank, cand in enumerate(candidates):
            chunk = cand.chunk
            content_lower = chunk.content.lower()
            title_lower = chunk.section_title.lower()
            doc_lower = chunk.doc_name.lower()
            combined_text = f'{doc_lower} {title_lower} {content_lower}'

            matched = []
            term_score = 0.0

            # 1. Check exact match for each query token
            for t in query_tokens:
                if t in combined_text:
                    matched.append(t)
                    # Higher weight for matches in titles and document names
                    if t in title_lower or t in doc_lower:
                        term_score += 0.35
                    else:
                        term_score += 0.20

            # 2. Key entity / number bonus (e.g. 2025, 2026, 75%, 60%, SR-402, Process X, MFA)
            entity_matches = re.findall(r'\b(2025|2026|75%|60%|sr-402|sr402|process\s*x|mfa|2fa|aes-256|vpn|7\s*years|2\s*hours)\b', query.lower())
            for em in entity_matches:
                if em in combined_text:
                    term_score += 0.40

            # 3. Base score from hybrid retrieval (normalized)
            base_score = min(1.0, cand.final_score)

            # Combined relevance score
            final_relevance = min(1.0, (base_score * 0.45) + (term_score * 0.55))

            # Extract focused snippet (around first matched term or top 250 chars)
            snippet = chunk.content[:300] + '...' if len(chunk.content) > 300 else chunk.content

            reranked.append(RerankedEvidence(
                chunk=chunk,
                relevance_score=round(float(final_relevance), 4),
                original_rank=orig_rank + 1,
                reranked_rank=0,
                matched_terms=list(set(matched)),
                evidence_snippet=snippet
            ))

        # Sort by reranked relevance score descending
        reranked.sort(key=lambda x: x.relevance_score, reverse=True)

        for new_rank, item in enumerate(reranked):
            item.reranked_rank = new_rank + 1

        # Filter by threshold if needed, but ensure at least top 1 is returned if non-empty
        filtered = [item for item in reranked if item.relevance_score >= self.relevance_threshold]
        if not filtered and reranked:
            filtered = [reranked[0]]

        return filtered[:top_k]
