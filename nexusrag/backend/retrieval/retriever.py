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

    def _ensure_vector_dimensions(self):
        """
        Make sure stored vectors and the current embedding provider
        use exactly the same dimension.

        If old vectors were created with an older TF-IDF vocabulary,
        rebuild them automatically from the stored chunk texts.
        """

        if (
            self.vector_store.vectors is None
            or not self.vector_store.chunks_data
        ):
            return

        if not hasattr(
            self.embedding_provider,
            "fit"
        ):
            return

        texts = [
            d.get("text", "")
            for d in self.vector_store.chunks_data
        ]

        texts = [
            str(text)
            for text in texts
            if str(text).strip()
        ]

        if not texts:
            return

        # Always fit the local provider against the SAME stored
        # document corpus used by the vector store.
        self.embedding_provider.fit(texts)

        expected_dimension = getattr(
            self.embedding_provider,
            "dimension",
            None
        )

        stored_dimension = (
            self.vector_store.vectors.shape[1]
            if self.vector_store.vectors.ndim == 2
            else None
        )

        if (
            expected_dimension is not None
            and stored_dimension != expected_dimension
        ):

            print(
                "[NexusRAG] Rebuilding vector store: "
                f"stored={stored_dimension}, "
                f"expected={expected_dimension}"
            )

            new_vectors = (
                self.embedding_provider.embed_texts(
                    texts
                )
            )

            # The filtered text list should normally match all chunks.
            # If empty/filtered texts caused a mismatch, rebuild using
            # every chunk instead.
            if len(new_vectors) != len(
                self.vector_store.chunks_data
            ):

                all_texts = [
                    str(d.get("text", ""))
                    for d in self.vector_store.chunks_data
                ]

                self.embedding_provider.fit(
                    all_texts
                )

                new_vectors = (
                    self.embedding_provider.embed_texts(
                        all_texts
                    )
                )

            self.vector_store.vectors = new_vectors
            self.vector_store.save()

            print(
                "[NexusRAG] Vector store rebuilt successfully."
            )

    def retrieve(
        self,
        query: str,
        top_k: Optional[int] = None
    ) -> List[RetrievalResult]:

        k = top_k or self.top_k

        if not query.strip():
            return []

        # Repair old/incompatible local TF-IDF vectors
        # before creating the query embedding.
        self._ensure_vector_dimensions()

        if (
            hasattr(self.embedding_provider, "fit")
            and not getattr(
                self.embedding_provider,
                "is_fitted",
                False
            )
        ):

            all_texts = [
                d.get("text", "")
                for d in self.vector_store.chunks_data
            ]

            if all_texts:
                self.embedding_provider.fit(
                    all_texts
                )

        query_embedding = (
            self.embedding_provider.embed_query(
                query
            )
        )

        raw_results = (
            self.vector_store.similarity_search(
                query_embedding,
                top_k=k
            )
        )

        results: List[RetrievalResult] = []

        for chunk_data, score in raw_results:

            meta = chunk_data.get(
                "metadata",
                {}
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
                    text=chunk_data.get(
                        "text",
                        ""
                    ),
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
                        len(
                            chunk_data.get(
                                "text",
                                ""
                            )
                        )
                    ),
                    token_count=chunk_data.get(
                        "token_count",
                        max(
                            1,
                            len(
                                chunk_data.get(
                                    "text",
                                    ""
                                ).split()
                            )
                        )
                    ),
                    metadata=meta
                )
            )

        return results