from dataclasses import dataclass, field
from typing import List, Dict, Any, Optional
from .embedding_service import BaseEmbeddingProvider
from .vector_store import LocalVectorStore

@dataclass
class RetrievalResult:
    chunk_id: str
    document_id: str
    text: str
    similarity_score: float
    document_name: str
    page_number: Optional[int] = 1
    section_title: Optional[str] = None
    sheet_name: Optional[str] = None
    version: Optional[str] = "1.0"
    year: Optional[str] = "2026"
    department: Optional[str] = "General"
    char_count: int = 0
    token_count: int = 0
    metadata: Dict[str, Any] = field(default_factory=dict)

class VectorRetriever:
    def __init__(self, vector_store: LocalVectorStore, embedding_provider: BaseEmbeddingProvider, top_k: int = 5):
        self.vector_store = vector_store
        self.embedding_provider = embedding_provider
        self.top_k = top_k

    def retrieve(self, query: str, top_k: Optional[int] = None) -> List[RetrievalResult]:
        k = top_k or self.top_k
        if not query.strip():
            return []

        # If using local dense provider, ensure it is fitted with stored texts
        if hasattr(self.embedding_provider, "fit") and not getattr(self.embedding_provider, "is_fitted", False):
            all_texts = [d.get("text", "") for d in self.vector_store.chunks_data]
            if all_texts:
                self.embedding_provider.fit(all_texts)

        query_embedding = self.embedding_provider.embed_query(query)
        raw_results = self.vector_store.similarity_search(query_embedding, top_k=k)

        results: List[RetrievalResult] = []
        for chunk_data, score in raw_results:
            meta = chunk_data.get("metadata", {})
            results.append(RetrievalResult(
                chunk_id=chunk_data.get("chunk_id", ""),
                document_id=chunk_data.get("document_id", ""),
                text=chunk_data.get("text", ""),
                similarity_score=round(float(score), 4),
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

        return results
