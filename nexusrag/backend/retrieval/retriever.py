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

    def __init__(
        self,
        vector_store: LocalVectorStore,
        embedding_provider: BaseEmbeddingProvider,
        top_k: int = 5
    ):
        self.vector_store = vector_store
        self.embedding_provider = embedding_provider
        self.top_k = top_k

    def retrieve(
        self,
        query: str,
        top_k: Optional[int] = None
    ) -> List[RetrievalResult]:

        if not query or not query.strip():
            return []

        k = max(1, int(top_k if top_k is not None else self.top_k))

        if not getattr(self.vector_store, "chunks_data", None):
            return []

        # 1. Embed query
        query_embedding = self.embedding_provider.embed_query(query.strip())

        # 2. Perform similarity search safely
        raw_results = self.vector_store.similarity_search(
            query_embedding,
            top_k=k
        )

        results: List[RetrievalResult] = []

        # 3. Parse output securely
        for item in raw_results:
            if not isinstance(item, dict):
                continue

            meta = item.get("metadata", {})
            if not isinstance(meta, dict):
                meta = {}

            text = str(item.get("text", ""))

            results.append(
                RetrievalResult(
                    chunk_id=str(item.get("chunk_id", "unknown")),
                    document_id=str(item.get("document_id", "unknown")),
                    text=text,
                    similarity_score=round(float(item.get("score", 0.0)), 4),
                    document_name=meta.get("document_name", "Document"),
                    page_number=meta.get("page_number", 1),
                    section_title=meta.get("section_title"),
                    sheet_name=meta.get("sheet_name"),
                    version=meta.get("version", "1.0"),
                    year=meta.get("year", "2026"),
                    department=meta.get("department", "General"),
                    char_count=item.get("char_count", len(text)),
                    token_count=item.get("token_count", max(1, len(text.split()))),
                    metadata=meta
                )
            )

        return results