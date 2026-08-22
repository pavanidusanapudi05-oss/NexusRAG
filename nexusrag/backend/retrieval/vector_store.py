import json
from pathlib import Path
from typing import List, Dict, Any, Tuple, Optional

import numpy as np

from nexusrag.backend.models.chunk import DocumentChunk


class LocalVectorStore:
    """
    Local persistent vector store.

    Stores:
    - chunk IDs
    - document IDs
    - chunk metadata
    - embedding vectors

    Includes dimension validation so old/incompatible
    vector indexes cannot crash similarity search.
    """

    def __init__(self, persist_dir: Optional[Path] = None):
        self.persist_dir = persist_dir or Path("nexusrag/data/vector_store")
        self.persist_dir.mkdir(parents=True, exist_ok=True)

        self.chunk_ids: List[str] = []
        self.document_ids: List[str] = []
        self.chunks_data: List[Dict[str, Any]] = []
        self.vectors: Optional[np.ndarray] = None

        self.load()

    def add_chunks(
        self,
        document_id: str,
        chunks: List[DocumentChunk],
        embeddings: np.ndarray
    ):
        if not chunks or embeddings is None or len(embeddings) == 0:
            return

        embeddings = np.asarray(embeddings, dtype=np.float32)

        if embeddings.ndim == 1:
            embeddings = embeddings.reshape(1, -1)

        if len(chunks) != len(embeddings):
            raise ValueError(
                f"Chunk/embedding count mismatch: "
                f"{len(chunks)} chunks vs {len(embeddings)} embeddings"
            )

        # Remove previous vectors for this document.
        self.delete_document_vectors(
            document_id,
            auto_save=False
        )

        new_chunk_ids = [c.chunk_id for c in chunks]
        new_doc_ids = [document_id] * len(chunks)
        new_chunks_data = [c.to_dict() for c in chunks]

        if self.vectors is None or len(self.chunk_ids) == 0:
            self.chunk_ids = list(new_chunk_ids)
            self.document_ids = list(new_doc_ids)
            self.chunks_data = list(new_chunks_data)
            self.vectors = embeddings
        else:
            # Prevent mixing different embedding dimensions.
            if self.vectors.ndim != 2 or embeddings.ndim != 2:
                raise ValueError("Invalid vector dimensions.")

            if self.vectors.shape[1] != embeddings.shape[1]:
                raise ValueError(
                    "Embedding dimension mismatch while adding vectors: "
                    f"existing={self.vectors.shape[1]}, "
                    f"new={embeddings.shape[1]}"
                )

            self.chunk_ids.extend(new_chunk_ids)
            self.document_ids.extend(new_doc_ids)
            self.chunks_data.extend(new_chunks_data)

            self.vectors = np.vstack(
                [self.vectors, embeddings]
            ).astype(np.float32)

        self.save()

    def delete_document_vectors(
        self,
        document_id: str,
        auto_save: bool = True
    ) -> int:

        if not self.document_ids or self.vectors is None:
            return 0

        indices_to_keep = [
            i
            for i, doc_id in enumerate(self.document_ids)
            if doc_id != document_id
        ]

        deleted_count = (
            len(self.document_ids)
            - len(indices_to_keep)
        )

        if deleted_count == 0:
            return 0

        self.chunk_ids = [
            self.chunk_ids[i]
            for i in indices_to_keep
        ]

        self.document_ids = [
            self.document_ids[i]
            for i in indices_to_keep
        ]

        self.chunks_data = [
            self.chunks_data[i]
            for i in indices_to_keep
        ]

        if not indices_to_keep:
            self.vectors = None
        else:
            self.vectors = self.vectors[
                indices_to_keep
            ]

        if auto_save:
            self.save()

        return deleted_count

    def similarity_search(
        self,
        query_embedding: np.ndarray,
        top_k: int = 5
    ) -> List[Tuple[Dict[str, Any], float]]:

        if (
            self.vectors is None
            or len(self.chunk_ids) == 0
        ):
            return []

        vectors = np.asarray(
            self.vectors,
            dtype=np.float32
        )

        query_vec = np.asarray(
            query_embedding,
            dtype=np.float32
        )

        if query_vec.ndim == 1:
            query_vec = query_vec.reshape(1, -1)

        if query_vec.ndim != 2:
            raise ValueError(
                f"Query embedding must be 2D, got shape "
                f"{query_vec.shape}"
            )

        if vectors.ndim != 2:
            raise ValueError(
                f"Stored vectors must be 2D, got shape "
                f"{vectors.shape}"
            )

        stored_dim = vectors.shape[1]
        query_dim = query_vec.shape[1]

        # IMPORTANT:
        # Never allow incompatible vectors into np.dot().
        if stored_dim != query_dim:
            raise ValueError(
                "Embedding dimension mismatch. "
                f"Stored vectors have dimension {stored_dim}, "
                f"but query embedding has dimension {query_dim}. "
                "The vector index must be rebuilt."
            )

        # Cosine similarity because embeddings are L2 normalized.
        scores = np.dot(
            vectors,
            query_vec.T
        ).flatten()

        top_k = max(1, int(top_k))

        top_indices = np.argsort(scores)[::-1][:top_k]

        results = []

        for idx in top_indices:
            score = float(scores[idx])

            results.append(
                (
                    self.chunks_data[idx],
                    score
                )
            )

        return results

    def get_total_vectors(self) -> int:
        return (
            len(self.chunk_ids)
            if self.chunk_ids
            else 0
        )

    def get_indexed_document_ids(self) -> List[str]:
        return list(set(self.document_ids))

    def save(self):
        vec_file = self.persist_dir / "vectors.npy"
        meta_file = self.persist_dir / "metadata.json"

        if (
            self.vectors is not None
            and len(self.chunk_ids) > 0
        ):
            np.save(
                str(vec_file),
                self.vectors
            )

            data = {
                "chunk_ids": self.chunk_ids,
                "document_ids": self.document_ids,
                "chunks_data": self.chunks_data
            }

            with open(
                meta_file,
                "w",
                encoding="utf-8"
            ) as f:
                json.dump(
                    data,
                    f,
                    indent=2,
                    ensure_ascii=False
                )

        else:
            if vec_file.exists():
                vec_file.unlink()

            if meta_file.exists():
                meta_file.unlink()

    def load(self) -> bool:
        vec_file = self.persist_dir / "vectors.npy"
        meta_file = self.persist_dir / "metadata.json"

        if not (
            vec_file.exists()
            and meta_file.exists()
        ):
            self.clear()
            return False

        try:
            vectors = np.load(
                str(vec_file),
                allow_pickle=False
            )

            with open(
                meta_file,
                "r",
                encoding="utf-8"
            ) as f:
                data = json.load(f)

            self.vectors = np.asarray(
                vectors,
                dtype=np.float32
            )

            self.chunk_ids = data.get(
                "chunk_ids",
                []
            )

            self.document_ids = data.get(
                "document_ids",
                []
            )

            self.chunks_data = data.get(
                "chunks_data",
                []
            )

            # Validate index consistency.
            if (
                self.vectors.ndim != 2
                or len(self.chunk_ids)
                != len(self.vectors)
                or len(self.document_ids)
                != len(self.vectors)
                or len(self.chunks_data)
                != len(self.vectors)
            ):
                print(
                    "[Warning] Invalid vector store detected. "
                    "Clearing incompatible index."
                )
                self.clear()
                return False

            print(
                f"[VectorStore] Loaded "
                f"{len(self.vectors)} vectors "
                f"with dimension "
                f"{self.vectors.shape[1]}"
            )

            return True

        except Exception as e:
            print(
                f"[VectorStore] Error loading vector store: {e}"
            )

            self.clear()
            return False

    def clear(self):
        self.chunk_ids = []
        self.document_ids = []
        self.chunks_data = []
        self.vectors = None

        vec_file = self.persist_dir / "vectors.npy"
        meta_file = self.persist_dir / "metadata.json"

        for file_path in [
            vec_file,
            meta_file
        ]:
            try:
                if file_path.exists():
                    file_path.unlink()
            except Exception:
                pass