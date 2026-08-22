import re
import math
from typing import List, Dict, Any, Tuple, Optional
from nexusrag.backend.models.chunk import DocumentChunk
from nexusrag.backend.retrieval.retriever import RetrievalResult

class BM25KeywordSearch:
    def __init__(self, k1: float = 1.5, b: float = 0.75):
        self.k1 = k1
        self.b = b
        self.corpus: List[Dict[str, Any]] = []
        self.doc_len: List[int] = []
        self.avg_doc_len: float = 0.0
        self.doc_freqs: List[Dict[str, int]] = []
        self.idf: Dict[str, float] = {}
        self.vocab: set = set()

    @staticmethod
    def tokenize(text: str) -> List[str]:
        return re.findall(r'[a-zA-Z0-9_\-\.]+', text.lower())

    def index_chunks(self, chunks: List[Dict[str, Any]]):
        self.corpus = chunks
        if not chunks:
            self.doc_len = []
            self.avg_doc_len = 0.0
            self.doc_freqs = []
            self.idf = {}
            return

        self.doc_len = []
        self.doc_freqs = []
        df_counts: Dict[str, int] = {}

        for c in chunks:
            text = c.get("text", "")
            tokens = self.tokenize(text)
            self.doc_len.append(len(tokens))

            freqs: Dict[str, int] = {}
            for t in tokens:
                freqs[t] = freqs.get(t, 0) + 1
            self.doc_freqs.append(freqs)

            for unique_t in set(tokens):
                df_counts[unique_t] = df_counts.get(unique_t, 0) + 1

        total_docs = len(chunks)
        self.avg_doc_len = sum(self.doc_len) / total_docs if total_docs > 0 else 0.0

        # Compute IDF
        self.idf = {}
        for term, df in df_counts.items():
            self.idf[term] = math.log(1.0 + (total_docs - df + 0.5) / (df + 0.5))

    def search(self, query: str, top_k: int = 5) -> List[Tuple[Dict[str, Any], float]]:
        if not self.corpus or not query.strip():
            return []

        q_tokens = self.tokenize(query)
        if not q_tokens:
            return []

        scores: List[float] = [0.0] * len(self.corpus)

        for i, freqs in enumerate(self.doc_freqs):
            d_len = self.doc_len[i]
            score = 0.0
            for t in q_tokens:
                if t not in freqs:
                    continue
                tf = freqs[t]
                idf = self.idf.get(t, 0.0)
                numerator = tf * (self.k1 + 1)
                denominator = tf + self.k1 * (1 - self.b + self.b * (d_len / (self.avg_doc_len or 1.0)))
                score += idf * (numerator / denominator)
            scores[i] = score

        max_score = max(scores) if scores and max(scores) > 0 else 1.0
        normalized_scores = [s / max_score for s in scores]

        top_indices = sorted(range(len(scores)), key=lambda idx: scores[idx], reverse=True)[:top_k]
        results = []
        for idx in top_indices:
            if scores[idx] > 0:
                results.append((self.corpus[idx], normalized_scores[idx]))
        return results
