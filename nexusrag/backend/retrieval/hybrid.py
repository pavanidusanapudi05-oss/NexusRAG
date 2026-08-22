from dataclasses import dataclass
from typing import List, Dict, Tuple, Optional
from nexusrag.backend.ingestion.chunker import DocumentChunk
from nexusrag.backend.vectorstore.index import VectorStore
from .bm25_retriever import BM25Retriever

@dataclass
class RetrievalCandidate:
    chunk: DocumentChunk
    dense_score: float
    bm25_score: float
    dense_rank: int
    bm25_rank: int
    rrf_score: float
    final_score: float

class HybridRetriever:
    def __init__(self, vector_store: VectorStore, bm25_retriever: BM25Retriever, alpha: float = 0.65, rrf_k: int = 60):
        self.vector_store = vector_store
        self.bm25_retriever = bm25_retriever
        self.alpha = alpha  # weight for dense similarity vs (1-alpha) for BM25
        self.rrf_k = rrf_k  # RRF constant

    def retrieve(self, query: str, top_k: int = 10) -> List[RetrievalCandidate]:
        # 1. Fetch dense candidates (pool 2x top_k)
        candidate_pool_size = max(top_k * 2, 20)
        dense_results = self.vector_store.search(query, top_k=candidate_pool_size)
        bm25_results = self.bm25_retriever.search(query, top_k=candidate_pool_size)

        dense_map = {chunk.chunk_id: (chunk, score, rank + 1) for rank, (chunk, score) in enumerate(dense_results)}
        bm25_map = {chunk.chunk_id: (chunk, score, rank + 1) for rank, (chunk, score) in enumerate(bm25_results)}

        all_chunk_ids = set(dense_map.keys()).union(set(bm25_map.keys()))
        candidates: List[RetrievalCandidate] = []

        for cid in all_chunk_ids:
            chunk = None
            d_score, d_rank = 0.0, 999
            b_score, b_rank = 0.0, 999

            if cid in dense_map:
                chunk, d_score, d_rank = dense_map[cid]
            if cid in bm25_map:
                chunk_b, b_score, b_rank = bm25_map[cid]
                if chunk is None:
                    chunk = chunk_b

            # Compute RRF score: 1 / (k + rank)
            rrf_dense = 1.0 / (self.rrf_k + d_rank) if d_rank <= candidate_pool_size else 0.0
            rrf_bm25 = 1.0 / (self.rrf_k + b_rank) if b_rank <= candidate_pool_size else 0.0
            rrf_total = (self.alpha * rrf_dense) + ((1.0 - self.alpha) * rrf_bm25)

            # Combined weighted score
            combined = (self.alpha * max(0.0, d_score)) + ((1.0 - self.alpha) * max(0.0, b_score))
            # Fuse RRF and combined score
            final_score = (combined * 0.7) + (rrf_total * 30.0 * 0.3)

            candidates.append(RetrievalCandidate(
                chunk=chunk,
                dense_score=round(float(d_score), 4),
                bm25_score=round(float(b_score), 4),
                dense_rank=d_rank,
                bm25_rank=b_rank,
                rrf_score=round(float(rrf_total), 6),
                final_score=round(float(final_score), 4)
            ))

        # Sort descending by final score
        candidates.sort(key=lambda x: x.final_score, reverse=True)
        return candidates[:top_k]
