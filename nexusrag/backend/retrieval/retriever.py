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

```
def __init__(
    self,
    vector_store: LocalVectorStore,
    embedding_provider: BaseEmbeddingProvider,
    top_k: int = 5
):

    self.vector_store = vector_store
    self.embedding_provider = embedding_provider
    self.top_k = top_k

def _prepare_local_embeddings(self) -> None:
    """
    Ensure the local TF-IDF provider is fitted using exactly
    the same corpus that exists in the vector store.
    """

    if not hasattr(
        self.embedding_provider,
        "fit"
    ):
        return

    if not self.vector_store.chunks_data:
        return

    texts = [
        str(
            item.get(
                "text",
                ""
            )
        )
        for item in self.vector_store.chunks_data
    ]

    texts = [
        text
        for text in texts
        if text.strip()
    ]

    if not texts:
        return

    self.embedding_provider.fit(
        texts
    )

def _rebuild_vector_store_if_needed(self) -> None:
    """
    Rebuild local TF-IDF vectors when the stored vector dimension
    does not match the current fitted vocabulary dimension.
    """

    if not hasattr(
        self.embedding_provider,
        "fit"
    ):
        return

    if (
        self.vector_store.vectors is None
        or not self.vector_store.chunks_data
    ):
        return

    self._prepare_local_embeddings()

    expected_dimension = getattr(
        self.embedding_provider,
        "dimension",
        0
    )

    if expected_dimension <= 0:
        return

    stored_vectors = self.vector_store.vectors

    if stored_vectors.ndim == 1:
        stored_dimension = len(
            stored_vectors
        )
    else:
        stored_dimension = (
            stored_vectors.shape[1]
        )

    if stored_dimension == expected_dimension:
        return

    print(
        "[NexusRAG] Rebuilding incompatible "
        "local vector store: "
        f"{stored_dimension} -> "
        f"{expected_dimension}"
    )

    all_texts = [
        str(
            item.get(
                "text",
                ""
            )
        )
        for item in self.vector_store.chunks_data
    ]

    new_vectors = (
        self.embedding_provider.embed_texts(
            all_texts
        )
    )

    if len(new_vectors) != len(
        self.vector_store.chunks_data
    ):
        raise ValueError(
            "Failed to rebuild vector store. "
            "Number of generated vectors does not "
            "match stored chunks."
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

    clean_query = (
        query.strip()
        if query
        else ""
    )

    if not clean_query:
        return []

    # -----------------------------------------------------
    # LOCAL TF-IDF PROVIDER
    # -----------------------------------------------------

    if hasattr(
        self.embedding_provider,
        "fit"
    ):

        self._prepare_local_embeddings()

        self._rebuild_vector_store_if_needed()

    # -----------------------------------------------------
    # MAKE QUERY EMBEDDING
    # -----------------------------------------------------

    query_embedding = (
        self.embedding_provider.embed_query(
            clean_query
        )
    )

    # -----------------------------------------------------
    # FINAL SAFETY CHECK
    # -----------------------------------------------------

    if (
        self.vector_store.vectors is not None
        and len(self.vector_store.vectors) > 0
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

        query_dimension = (
            query_embedding.shape[1]
            if query_embedding.ndim == 2
            else query_embedding.shape[0]
        )

        if (
            stored_dimension
            != query_dimension
        ):

            raise ValueError(
                "Embedding dimension mismatch "
                "after automatic rebuild: "
                f"stored={stored_dimension}, "
                f"query={query_dimension}. "
                "Please re-index the documents."
            )

    # -----------------------------------------------------
    # SIMILARITY SEARCH
    # -----------------------------------------------------

    k = (
        top_k
        if top_k is not None
        else self.top_k
    )

    raw_results = (
        self.vector_store.similarity_search(
            query_embedding,
            top_k=k
        )
    )

    results: List[RetrievalResult] = []

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
                        len(
                            text.split()
                        )
                    )
                ),
                metadata=metadata
            )
        )

    return results
```
