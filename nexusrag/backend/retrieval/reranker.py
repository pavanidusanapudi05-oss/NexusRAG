import re
from typing import List
from nexusrag.backend.retrieval.retriever import RetrievalResult

class PrecisionReranker:
    def __init__(self, enabled: bool = True):
        self.enabled = enabled

    def rerank(self, query: str, candidates: List[RetrievalResult], top_k: int = 5) -> List[RetrievalResult]:
        if not self.enabled or not candidates or not query.strip():
            return candidates[:top_k]

        q_lower = query.lower()
        q_tokens = set(re.findall(r'[a-zA-Z0-9_\-\.]+', q_lower))
        numbers_and_codes = re.findall(r'\b(202[0-9]|sr[-_]?[0-9]+|\d+[%$]|v\d+)\b', q_lower)

        scored_candidates = []
        for cand in candidates:
            base_score = cand.similarity_score
            text_lower = cand.text.lower()
            sec_lower = (cand.section_title or "").lower()
            doc_lower = cand.document_name.lower()
            combined = f"{doc_lower} {sec_lower} {text_lower}"

            boost = 0.0

            # 1. Exact phrase boost
            if query.lower() in text_lower or query.lower() in sec_lower:
                boost += 0.25

            # 2. Section title match boost
            sec_tokens = set(re.findall(r'[a-zA-Z0-9_\-\.]+', sec_lower))
            sec_overlap = len(q_tokens.intersection(sec_tokens))
            if sec_overlap > 0:
                boost += 0.15 * min(sec_overlap, 3)

            # 3. Exact numerical / code match boost
            for nc in numbers_and_codes:
                clean_nc = nc.replace("-", "").replace("_", "")
                if clean_nc in combined.replace("-", "").replace("_", ""):
                    boost += 0.20

            # 4. Token overlap density
            tokens_in_text = set(re.findall(r'[a-zA-Z0-9_\-\.]+', text_lower))
            overlap_ratio = len(q_tokens.intersection(tokens_in_text)) / max(1, len(q_tokens))
            boost += 0.15 * overlap_ratio

            final_score = round(float(base_score + boost), 4)
            cand.similarity_score = final_score
            scored_candidates.append(cand)

        scored_candidates.sort(key=lambda c: c.similarity_score, reverse=True)
        return scored_candidates[:top_k]
