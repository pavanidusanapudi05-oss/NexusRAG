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
    # FIT LOCAL EMBEDDING PROVIDER
    # ---------------------------------------------------------

    def _fit_local_provider(self) -> None:

        if not hasattr(
            self.embedding_provider,
            "fit"
        ):
            return

        chunks_data = (
            self.vector_store.chunks_data
            or []
        )

        texts = []

        for item in chunks_data:

            text = str(
                item.get(
                    "text",
                    ""
                )
            ).strip()

            if text:
                texts.append(text)

        if not texts:
            return

        self.embedding_provider.fit(
            texts
        )

    # ---------------------------------------------------------
    # CHECK DIMENSIONS
    # ---------------------------------------------------------

    def _dimensions_match(
        self,
        query_embedding
    ) -> bool:

        if (
            self.vector_store.vectors is None
            or len(self.vector_store.vectors) == 0
        ):
            return True

        stored_vectors = (
            self.vector_store.vectors
        )

        if stored_vectors.ndim == 1:
            stored_dimension = (
                stored_vectors.shape[0]
            )
        else:
            stored_dimension = (
                stored_vectors.shape[1]
            )

        query_dimension = (
            query_embedding.shape[1]
            if query_embedding.ndim == 2
            else query_embedding.shape[0]
        )

        return (
            stored_dimension
            == query_dimension
        )

    # ---------------------------------------------------------
    # REBUILD LOCAL VECTOR STORE
    # ---------------------------------------------------------

    def _rebuild_local_vectors(self) -> None:

        if not hasattr(
            self.embedding_provider,
            "fit"
        ):
            return

        chunks_data = (
            self.vector_store.chunks_data
            or []
        )

        if not chunks_data:
            return

        texts = [
            str(
                item.get(
                    "text",
                    ""
                )
            )
            for item in chunks_data
        ]

        if not any(
            text.strip()
            for text in texts
        ):
            return

        # Fit using exactly the same corpus
        # stored in the vector store.
        self.embedding_provider.fit(
            texts
        )

        new_vectors = (
            self.embedding_provider.embed_texts(
                texts
            )
        )

        if (
            new_vectors is None
            or len(new_vectors)
            != len(chunks_data)
        ):
            raise ValueError(
                "Failed to rebuild vectors. "
                "Embedding count does not match "
                "stored chunks."
            )

        self.vector_store.vectors = (
            new_vectors
        )

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

        k = (
            top_k
            if top_k is not None
            else self.top_k
        )

        k = max(
            1,
            int(k)
        )

        # -----------------------------------------------------
        # EMPTY STORE
        # -----------------------------------------------------

        if (
            self.vector_store.vectors is None
            or not self.vector_store.chunks_data
        ):
            return []

        # -----------------------------------------------------
        # LOCAL TF-IDF
        # -----------------------------------------------------

        if hasattr(
            self.embedding_provider,
            "fit"
        ):

            self._fit_local_provider()

        # -----------------------------------------------------
        # CREATE QUERY EMBEDDING
        # -----------------------------------------------------

        query_embedding = (
            self.embedding_provider.embed_query(
                query.strip()
            )
        )

        # -----------------------------------------------------
        # DIMENSION VALIDATION
        # -----------------------------------------------------

        if not self._dimensions_match(
            query_embedding
        ):

            # Old vectors may have been created
            # using a different TF-IDF vocabulary.
            if hasattr(
                self.embedding_provider,
                "fit"
            ):

                self._rebuild_local_vectors()

                # Create the query embedding again
                # after rebuilding with the same vocabulary.
                query_embedding = (
                    self.embedding_provider.embed_query(
                        query.strip()
                    )
                )

            # If dimensions still do not match,
            # fail with a clear error instead of
            # NumPy's cryptic shapes error.
            if not self._dimensions_match(
                query_embedding
            ):

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
                    "Embedding dimension mismatch after "
                    "rebuilding the vector store. "
                    f"Stored shape={stored_shape}, "
                    f"query shape={query_shape}. "
                    "Please clear the old vector store "
                    "and re-index the documents."
                )

        # -----------------------------------------------------
        # SIMILARITY SEARCH
        # -----------------------------------------------------

        raw_results = (
            self.vector_store.similarity_search(
                query_embedding,
                top_k=k
            )
        )

        results: List[
            RetrievalResult
        ] = []

        # -----------------------------------------------------
        # CONVERT RESULTS
        # -----------------------------------------------------

        for chunk_data, score in raw_results:

            meta = chunk_data.get(
                "metadata",
                {}
            )

            if not isinstance(
                meta,
                dict
            ):
                meta = {}

            text = str(
                chunk_data.get(
                    "text",
                    ""
                )
            )

            results.append(
                RetrievalResult(
                    chunk_id=str(
                        chunk_data.get(
                            "chunk_id",
                            ""
                        )
                    ),

                    document_id=str(
                        chunk_data.get(
                            "document_id",
                            ""
                        )
                    ),

                    text=text,

                    similarity_score=round(
                        float(score),
                        4
                    ),

                    document_name=meta.get(
                        "document_name",
                        "Document"
                    ),

                    page_number=meta.get(
                        "page_number",
                        1
                    ),

                    section_title=meta.get(
                        "section_title"
                    ),

                    sheet_name=meta.get(
                        "sheet_name"
                    ),

                    version=meta.get(
                        "version",
                        "1.0"
                    ),

                    year=meta.get(
                        "year",
                        "2026"
                    ),

                    department=meta.get(
                        "department",
                        "General"
                    ),

                    char_count=chunk_data.get(
                        "char_count",
                        len(text)
                    ),

                    token_count=chunk_data.get(
                        "token_count",
                        max(
                            1,
                            len(
                                text.split()
                            )
                        )
                    ),

                    metadata=meta
                )
            )

        return results