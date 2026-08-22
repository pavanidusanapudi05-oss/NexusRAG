import json
from pathlib import Path
from typing import List, Dict, Any, Tuple, Optional

import numpy as np

class LocalVectorStore:

```
def __init__(
    self,
    persist_dir: Optional[Path] = None
):

    self.persist_dir = (
        persist_dir
        or Path("nexusrag/data/vector_store")
    )

    self.persist_dir.mkdir(
        parents=True,
        exist_ok=True
    )

    self.chunk_ids: List[str] = []
    self.document_ids: List[str] = []
    self.chunks_data: List[Dict[str, Any]] = []
    self.vectors: Optional[np.ndarray] = None

    self.load()

def add_chunks(
    self,
    document_id: str,
    chunks,
    embeddings: np.ndarray
) -> None:

    if (
        not chunks
        or embeddings is None
        or len(embeddings) == 0
    ):
        return

    embeddings = np.asarray(
        embeddings,
        dtype=np.float32
    )

    if embeddings.ndim == 1:
        embeddings = embeddings.reshape(
            1,
            -1
        )

    if len(embeddings) != len(chunks):
        raise ValueError(
            "Number of embeddings does not match "
            "number of chunks: "
            f"{len(embeddings)} != {len(chunks)}"
        )

    self.delete_document_vectors(
        document_id,
        auto_save=False
    )

    new_chunk_ids = [
        chunk.chunk_id
        for chunk in chunks
    ]

    new_document_ids = [
        document_id
    ] * len(chunks)

    new_chunks_data = [
        chunk.to_dict()
        for chunk in chunks
    ]

    if (
        self.vectors is None
        or len(self.chunk_ids) == 0
    ):

        self.chunk_ids = new_chunk_ids
        self.document_ids = new_document_ids
        self.chunks_data = new_chunks_data
        self.vectors = embeddings.copy()

    else:

        existing = np.asarray(
            self.vectors,
            dtype=np.float32
        )

        if existing.ndim == 1:
            existing = existing.reshape(
                1,
                -1
            )

        existing_dimension = existing.shape[1]
        new_dimension = embeddings.shape[1]

        if existing_dimension != new_dimension:

            print(
                "[NexusRAG] Embedding dimension changed: "
                f"{existing_dimension} -> {new_dimension}"
            )

            # A local TF-IDF vocabulary changed.
            # Rebuild the persisted store using the
            # newly indexed document.
            self.chunk_ids = new_chunk_ids
            self.document_ids = new_document_ids
            self.chunks_data = new_chunks_data
            self.vectors = embeddings.copy()

        else:

            self.chunk_ids.extend(
                new_chunk_ids
            )

            self.document_ids.extend(
                new_document_ids
            )

            self.chunks_data.extend(
                new_chunks_data
            )

            self.vectors = np.vstack(
                [
                    existing,
                    embeddings
                ]
            )

    self.save()

def delete_document_vectors(
    self,
    document_id: str,
    auto_save: bool = True
) -> int:

    if (
        not self.document_ids
        or self.vectors is None
    ):
        return 0

    vectors = np.asarray(
        self.vectors,
        dtype=np.float32
    )

    if vectors.ndim == 1:
        vectors = vectors.reshape(
            1,
            -1
        )

    usable_count = min(
        len(vectors),
        len(self.document_ids),
        len(self.chunk_ids),
        len(self.chunks_data)
    )

    if usable_count == 0:
        return 0

    keep_indices = [
        index
        for index in range(usable_count)
        if self.document_ids[index] != document_id
    ]

    deleted_count = (
        usable_count
        - len(keep_indices)
    )

    if not keep_indices:

        self.chunk_ids = []
        self.document_ids = []
        self.chunks_data = []
        self.vectors = None

    else:

        self.chunk_ids = [
            self.chunk_ids[index]
            for index in keep_indices
        ]

        self.document_ids = [
            self.document_ids[index]
            for index in keep_indices
        ]

        self.chunks_data = [
            self.chunks_data[index]
            for index in keep_indices
        ]

        self.vectors = vectors[
            keep_indices
        ]

    if auto_save:
        self.save()

    return deleted_count

def similarity_search(
    self,
    query_embedding: np.ndarray,
    top_k: int = 5
) -> List[
    Tuple[Dict[str, Any], float]
]:

    if (
        self.vectors is None
        or len(self.chunk_ids) == 0
    ):
        return []

    stored_vectors = np.asarray(
        self.vectors,
        dtype=np.float32
    )

    query_vec = np.asarray(
        query_embedding,
        dtype=np.float32
    )

    if stored_vectors.ndim == 1:
        stored_vectors = stored_vectors.reshape(
            1,
            -1
        )

    if query_vec.ndim == 1:
        query_vec = query_vec.reshape(
            1,
            -1
        )

    stored_dimension = (
        stored_vectors.shape[1]
    )

    query_dimension = (
        query_vec.shape[1]
    )

    # Never allow numpy.dot() to produce the
    # confusing "shapes not aligned" error.
    if stored_dimension != query_dimension:

        raise ValueError(
            "Embedding dimension mismatch: "
            f"stored vectors={stored_dimension}, "
            f"query embedding={query_dimension}. "
            "The vector store must be rebuilt "
            "with the current embedding configuration."
        )

    scores = np.dot(
        stored_vectors,
        query_vec.T
    ).flatten()

    if len(scores) == 0:
        return []

    top_k = max(
        1,
        min(
            int(top_k),
            len(scores)
        )
    )

    top_indices = np.argsort(
        scores
    )[::-1][:top_k]

    results = []

    for index in top_indices:

        index = int(index)

        if index >= len(
            self.chunks_data
        ):
            continue

        results.append(
            (
                self.chunks_data[index],
                float(scores[index])
            )
        )

    return results

def get_total_vectors(self) -> int:

    if not self.chunk_ids:
        return 0

    return len(
        self.chunk_ids
    )

def get_indexed_document_ids(
    self
) -> List[str]:

    return list(
        set(
            self.document_ids
        )
    )

def save(self) -> None:

    vector_file = (
        self.persist_dir
        / "vectors.npy"
    )

    metadata_file = (
        self.persist_dir
        / "metadata.json"
    )

    if (
        self.vectors is not None
        and len(self.chunk_ids) > 0
    ):

        np.save(
            str(vector_file),
            self.vectors
        )

        data = {
            "chunk_ids": self.chunk_ids,
            "document_ids": self.document_ids,
            "chunks_data": self.chunks_data
        }

        with open(
            metadata_file,
            "w",
            encoding="utf-8"
        ) as file:

            json.dump(
                data,
                file,
                indent=2
            )

    else:

        if vector_file.exists():
            vector_file.unlink()

        if metadata_file.exists():
            metadata_file.unlink()

def load(self) -> bool:

    vector_file = (
        self.persist_dir
        / "vectors.npy"
    )

    metadata_file = (
        self.persist_dir
        / "metadata.json"
    )

    if not (
        vector_file.exists()
        and metadata_file.exists()
    ):

        self.chunk_ids = []
        self.document_ids = []
        self.chunks_data = []
        self.vectors = None

        return False

    try:

        vectors = np.load(
            str(vector_file)
        )

        with open(
            metadata_file,
            "r",
            encoding="utf-8"
        ) as file:

            data = json.load(file)

        if vectors.ndim == 1:
            vectors = vectors.reshape(
                1,
                -1
            )

        chunk_ids = data.get(
            "chunk_ids",
            []
        )

        document_ids = data.get(
            "document_ids",
            []
        )

        chunks_data = data.get(
            "chunks_data",
            []
        )

        if not (
            len(vectors)
            == len(chunk_ids)
            == len(document_ids)
            == len(chunks_data)
        ):

            print(
                "[NexusRAG] Vector store metadata "
                "is inconsistent. Resetting store."
            )

            self.clear()

            return False

        self.vectors = vectors
        self.chunk_ids = chunk_ids
        self.document_ids = document_ids
        self.chunks_data = chunks_data

        return True

    except Exception as error:

        print(
            f"[NexusRAG] Error loading vector store: {error}"
        )

        self.chunk_ids = []
        self.document_ids = []
        self.chunks_data = []
        self.vectors = None

        return False

def clear(self) -> None:

    self.chunk_ids = []
    self.document_ids = []
    self.chunks_data = []
    self.vectors = None

    vector_file = (
        self.persist_dir
        / "vectors.npy"
    )

    metadata_file = (
        self.persist_dir
        / "metadata.json"
    )

    for file in [
        vector_file,
        metadata_file
    ]:

        try:
            if file.exists():
                file.unlink()
        except Exception:
            pass
```