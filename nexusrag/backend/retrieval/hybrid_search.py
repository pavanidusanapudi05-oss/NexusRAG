from typing import List, Dict, Any, Optional
from nexusrag.backend.retrieval.vector_store import LocalVectorStore
from nexusrag.backend.retrieval.embedding_service import BaseEmbeddingProvider
from nexusrag.backend.retrieval.retriever import RetrievalResult
from .semantic_search import SemanticSearch
from .keyword_search import BM25KeywordSearch
from .reranker import PrecisionReranker

class HybridSearch:
    def __init__(
        self,
        vector_store: LocalVectorStore,
        embedding_provider: BaseEmbeddingProvider,
        semantic_weight: float = 0.7,
        keyword_weight: float = 0.3,
        reranker_enabled: bool = True
    ):
        self.vector_store = vector_store
        self.embedding_provider = embedding_provider
        self.semantic_weight = semantic_weight
        self.keyword_weight = keyword_weight
        self.semantic_searcher = SemanticSearch(vector_store, embedding_provider)
        self.keyword_searcher = BM25KeywordSearch()
        self.reranker = PrecisionReranker(enabled=reranker_enabled)
        self._sync_keyword_index()

    def _sync_keyword_index(self):
        if self.vector_store.chunks_data:
            self.keyword_searcher.index_chunks(self.vector_store.chunks_data)

    def search(
        self,
        query: str,
        top_k: int = 5,
        doc_filter: Optional[str] = None,
        version_filter: Optional[str] = None,
        year_filter: Optional[str] = None
    ) -> List[RetrievalResult]:
        if not query.strip() or self.vector_store.get_total_vectors() == 0:
            return []

        self._sync_keyword_index()

        # 1. Semantic
        semantic_results = self.semantic_searcher.search(query, top_k=top_k * 3)

        # 2. Keyword
        keyword_results = self.keyword_searcher.search(query, top_k=top_k * 3)

        # 3. Fuse
        fused_candidates: Dict[str, Dict[str, Any]] = {}

        for chunk_data, sem_score in semantic_results:
            cid = chunk_data.get("chunk_id", "")
            fused_candidates[cid] = {
                "chunk_data": chunk_data,
                "sem_score": float(sem_score),
                "lex_score": 0.0
            }

        for chunk_data, lex_score in keyword_results:
            cid = chunk_data.get("chunk_id", "")
            if cid in fused_candidates:
                fused_candidates[cid]["lex_score"] = float(lex_score)
            else:
                fused_candidates[cid] = {
                    "chunk_data": chunk_data,
                    "sem_score": 0.0,
                    "lex_score": float(lex_score)
                }

        results: List[RetrievalResult] = []
        for cid, item in fused_candidates.items():
            chunk_data = item["chunk_data"]
            meta = chunk_data.get("metadata", {})

            if doc_filter and meta.get("document_name") != doc_filter:
                continue
            if version_filter and str(meta.get("version")) != str(version_filter):
                continue
            if year_filter and str(meta.get("year")) != str(year_filter):
                continue

            combined_score = (
                self.semantic_weight * item["sem_score"] +
                self.keyword_weight * item["lex_score"]
            )

            results.append(RetrievalResult(
                chunk_id=chunk_data.get("chunk_id", ""),
                document_id=chunk_data.get("document_id", ""),
                text=chunk_data.get("text", ""),
                similarity_score=round(combined_score, 4),
                document_name=meta.get("document_name", "Document"),
                page_number=meta.get("page_number", 1),
                section_title=meta.get("section_title"),
                sheet_name=meta.get("sheet_name"),
                version=meta.get("version", "1.0"),
                year=meta.get("year", "2026"),
                department=meta.get("department", "General"),
                char_count=chunk_data.get("char_count", len(chunk_data.get("text", ""))),
                token_count=chunk_data.get("token_count", max(1, len(chunk_data.get("text", "").split()))),
                metadata=meta
            ))

        results.sort(key=lambda r: r.similarity_score, reverse=True)
        return self.reranker.rerank(query, results, top_k=top_k)

    def retrieve(self, query: str, top_k: int = 5) -> List[RetrievalResult]:
        return self.search(query, top_k=top_k)
