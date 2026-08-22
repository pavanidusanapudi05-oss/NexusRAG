import numpy as np
```python
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

    metadata: Dict[str, Any] = field(
        default_factory=dict
    )


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

    # ============================================================
    # ENSURE VECTOR DIMENSIONS
    # ============================================================

    def _ensure_vector_dimensions(self):

        vectors = self.vector_store.vectors
        chunks = self.vector_store.chunks_data

        if vectors is None or not chunks:
            return

        # Local TF-IDF provider has a fit() method.
        if not hasattr(
            self.embedding_provider,
            "fit"
        ):
            return

        all_texts = [
            str(chunk.get("text", ""))
            for chunk in chunks
        ]

        if not all_texts:
            return

        # Always fit the TF-IDF vocabulary using
        # exactly the same corpus as the stored chunks.
        self.embedding_provider.fit(
            all_texts
        )

        expected_dimension = getattr(
            self.embedding_provider,
            "dimension",
            0
        )

        stored_vectors = np.asarray(
            vectors
        )

        if stored_vectors.ndim == 1:
            stored_vectors = stored_vectors.reshape(
                1,
                -1
            )

        stored_dimension = (
            stored_vectors.shape[1]
        )

        # --------------------------------------------------------
        # REBUILD IF DIMENSIONS DIFFER
        # --------------------------------------------------------

        if stored_dimension != expected_dimension:

            print(
                "[NexusRAG] Vector dimension mismatch detected."
            )

            print(
                f"[NexusRAG] Stored dimension: "
                f"{stored_dimension}"
            )

            print(
                f"[NexusRAG] Expected dimension: "
                f"{expected_dimension}"
            )

            new_vectors = (
                self.embedding_provider.embed_texts(
                    all_texts
                )
            )

            if len(new_vectors) != len(chunks):

                raise ValueError(
                    "Vector rebuild failed because the "
                    "number of embeddings does not match "
                    "the number of stored chunks."
                )

            self.vector_store.vectors = (
                new_vectors
            )

            self.vector_store.save()

            print(
                "[NexusRAG] Vector store rebuilt successfully."
            )

        else:

            self.vector_store.vectors = (
                stored_vectors
            )

    # ============================================================
    # RETRIEVE
    # ============================================================

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

        # Make sure stored vectors and the current
        # embedding provider use the same dimension.
        self._ensure_vector_dimensions()

        # If the provider is not fitted yet,
        # fit it using the complete stored corpus.
        if (
            hasattr(
                self.embedding_provider,
                "fit"
            )
            and not getattr(
                self.embedding_provider,
                "is_fitted",
                False
            )
        ):

            all_texts = [
                str(chunk.get("text", ""))
                for chunk
                in self.vector_store.chunks_data
            ]

            if all_texts:
                self.embedding_provider.fit(
                    all_texts
                )

        # --------------------------------------------------------
        # CREATE QUERY EMBEDDING
        # --------------------------------------------------------

        query_embedding = (
            self.embedding_provider.embed_query(
                query
            )
        )

        # --------------------------------------------------------
        # FINAL DIMENSION SAFETY CHECK
        # --------------------------------------------------------

        if (
            self.vector_store.vectors is not None
            and query_embedding is not None
        ):

            stored_vectors = (
                self.vector_store.vectors
            )

            if stored_vectors.ndim == 1:

                stored_dimension = len(
                    stored_vectors
                )

            else:

                stored_dimension = (
                    stored_vectors.shape[1]
                )

            if query_embedding.ndim == 1:

                query_dimension = (
                    query_embedding.shape[0]
                )

            else:

                query_dimension = (
                    query_embedding.shape[1]
                )

            if (
                stored_dimension
                != query_dimension
            ):

                raise ValueError(
                    "Embedding dimension mismatch "
                    "after automatic rebuild. "
                    f"Stored={stored_dimension}, "
                    f"Query={query_dimension}. "
                    "Please clear and re-index the documents."
                )

        # --------------------------------------------------------
        # SIMILARITY SEARCH
        # --------------------------------------------------------

        raw_results = (
            self.vector_store.similarity_search(
                query_embedding,
                top_k=k
            )
        )

        results: List[RetrievalResult] = []

        # --------------------------------------------------------
        # CONVERT RAW RESULTS
        # --------------------------------------------------------

        for chunk_data, score in raw_results:

            metadata = chunk_data.get(
                "metadata",
                {}
            )

            text = chunk_data.get(
                "text",
                ""
            )

            results.append(
                RetrievalResult(

                    chunk_id=chunk_data.get(
                        "chunk_id",
                        ""
                    ),

                    document_id=chunk_data.get(
                        "document_id",
                        ""
                    ),

                    text=text,

                    similarity_score=round(
                        float(score),
                        4
                    ),

                    document_name=metadata.get(
                        "document_name",
                        "Document"
                    ),

                    page_number=metadata.get(
                        "page_number",
                        1
                    ),

                    section_title=metadata.get(
                        "section_title"
                    ),

                    sheet_name=metadata.get(
                        "sheet_name"
                    ),

                    version=metadata.get(
                        "version",
                        "1.0"
                    ),

                    year=metadata.get(
                        "year",
                        "2026"
                    ),

                    department=metadata.get(
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
                            len(text.split())
                        )
                    ),

                    metadata=metadata
                )
            )

        return results
```
