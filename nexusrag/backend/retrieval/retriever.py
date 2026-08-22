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

    # ---------------------------------------------------------
    # CHECK DIMENSIONS
    # ---------------------------------------------------------

    def _dimensions_match(self, query_embedding) -> bool:
        if (
            self.vector_store.vectors is None
            or len(self.vector_store.vectors) == 0
        ):
            return True

        stored_vectors = self.vector_store.vectors

        if stored_vectors.ndim == 1:
            stored_dimension = stored_vectors.shape[0]
        else:
            stored_dimension = stored_vectors.shape[1]

        query_dimension = (
            query_embedding.shape[1]
            if getattr(query_embedding, "ndim", 1) == 2
            else query_embedding.shape[0]
        )

        return stored_dimension == query_dimension

    # ---------------------------------------------------------
    # REBUILD LOCAL VECTOR STORE
    # ---------------------------------------------------------

    def _rebuild_local_vectors(self) -> None:
        if not hasattr(self.embedding_provider, "fit"):
            return

        chunks_data = self.vector_store.chunks_data or []
        if not chunks_data:
            return

        texts = [str(item.get("text", "")) for item in chunks_data]
        if not any(text.strip() for text in texts):
            return

        # Fit and embed using the exact stored corpus
        self.embedding_provider.fit(texts)
        new_vectors = self.embedding_provider.embed_texts(texts)

        if new_vectors is None or len(new_vectors) != len(chunks_data):
            raise ValueError(
                "Failed to rebuild vectors. "
                "Embedding count does not match stored chunks."
            )

        self.vector_store.vectors = new_vectors
        if hasattr(self.vector_store, "save"):
            self.vector_store.save()

    # ---------------------------------------------------------
    # RETRIEVE
    # ---------------------------------------------------------

    def retrieve(
        self,
        query: str,
        top_k: Optional[int] = None
    ) -> List[RetrievalResult]:

        if not query or not query.strip():
            return []

        k = top_k if top_k is not None else self.top_k
        k = max(1, int(k))

        # Check empty store
        if (
            self.vector_store.vectors is None
            or not getattr(self.vector_store, "chunks_data", None)
        ):
            return []

        # -----------------------------------------------------
        # CREATE QUERY EMBEDDING
        # -----------------------------------------------------
        # Note: Do NOT call _fit_local_provider() here to avoid resetting vocabulary
        query_embedding = self.embedding_provider.embed_query(query.strip())

        # -----------------------------------------------------
        # DIMENSION VALIDATION & AUTO REBUILD
        # -----------------------------------------------------
        if not self._dimensions_match(query_embedding):
            if hasattr(self.embedding_provider, "fit"):
                self._rebuild_local_vectors()
                query_embedding = self.embedding_provider.embed_query(query.strip())

            if not self._dimensions_match(query_embedding):
                stored_shape = (
                    self.vector_store.vectors.shape
                    if self.vector_store.vectors is not None
                    else None
                )
                query_shape = (
                    query_embedding.shape
                    if query_embedding is not None
                    else None
                )
                raise ValueError(
                    "Embedding dimension mismatch. "
                    f"Stored shape={stored_shape}, query shape={query_shape}. "
                    "Please re-index your documents."
                )

        # -----------------------------------------------------
        # SIMILARITY SEARCH
        # -----------------------------------------------------
        raw_results = self.vector_store.similarity_search(
            query_embedding,
            top_k=k
        )

        results: List[RetrievalResult] = []

        # -----------------------------------------------------
        # CONVERT RESULTS (Safe unpacking for dict & tuple)
        # -----------------------------------------------------
        for item in raw_results:
            # Handle item if returned as dictionary from vector_store
            if isinstance(item, dict):
                chunk_data = item
                score = item.get("score", 0.0)
            elif isinstance(item, tuple):
                chunk_data = item[0] if len(item) > 0 else {}
                score = item[1] if len(item) > 1 else 0.0
            else:
                chunk_data = getattr(item, "metadata", {})
                score = 0.0

            meta = chunk_data.get("metadata", {})
            if not isinstance(meta, dict):
                meta = {}

            text = str(chunk_data.get("text", ""))

            results.append(
                RetrievalResult(
                    chunk_id=str(chunk_data.get("chunk_id", chunk_data.get("document_id", ""))),
                    document_id=str(chunk_data.get("document_id", "")),
                    text=text,
                    similarity_score=round(float(score), 4),
                    document_name=meta.get("document_name", "Document"),
                    page_number=meta.get("page_number", 1),
                    section_title=meta.get("section_title"),
                    sheet_name=meta.get("sheet_name"),
                    version=meta.get("version", "1.0"),
                    year=meta.get("year", "2026"),
                    department=meta.get("department", "General"),
                    char_count=chunk_data.get("char_count", len(text)),
                    token_count=chunk_data.get("token_count", max(1, len(text.split()))),
                    metadata=meta
                )
            )

        return results