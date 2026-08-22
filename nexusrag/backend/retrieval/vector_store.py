import json
from pathlib import Path
from typing import List, Dict, Any, Tuple, Optional

import numpy as np

from nexusrag.backend.models.chunk import DocumentChunk


class LocalVectorStore:

    def __init__(self, persist_dir: Optional[Path] = None):

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

    # ---------------------------------------------------------
    # ADD CHUNKS
    # ---------------------------------------------------------

    def add_chunks(
        self,
        document_id: str,
        chunks: List[DocumentChunk],
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

        # Make sure embeddings are 2-D
        if embeddings.ndim == 1:
            embeddings = embeddings.reshape(
                1, -1
            )

        if len(embeddings) != len(chunks):
            raise ValueError(
                "Number of embeddings does not match "
                "number of chunks: "
                f"{len(embeddings)} != {len(chunks)}"
            )

        # Remove old vectors belonging to this document.
        self.delete_document_vectors(
            document_id,
            auto_save=False
        )

        new_chunk_ids = [
            c.chunk_id
            for c in chunks
        ]

        new_doc_ids = [
            document_id
        ] * len(chunks)

        new_chunks_data = [
            c.to_dict()
            for c in chunks
        ]

        # -----------------------------------------------------
        # EMPTY VECTOR STORE
        # -----------------------------------------------------

        if (
            self.vectors is None
            or len(self.chunk_ids) == 0
        ):

            self.chunk_ids = list(
                new_chunk_ids
            )

            self.document_ids = list(
                new_doc_ids
            )

            self.chunks_data = list(
                new_chunks_data
            )

            self.vectors = embeddings.copy()

        # -----------------------------------------------------
        # EXISTING VECTOR STORE
        # -----------------------------------------------------

        else:

            existing = np.asarray(
                self.vectors,
                dtype=np.float32
            )

            if existing.ndim == 1:
                existing = existing.reshape(
                    1, -1
                )

            # If dimensions differ, do NOT perform vstack.
            # Existing vectors were created using a different
            # embedding vocabulary/model.
            if existing.shape[1] != embeddings.shape[1]:

                print(
                    "[NexusRAG] Vector dimension changed: "
                    f"{existing.shape[1]} -> "
                    f"{embeddings.shape[1]}"
                )

                # Rebuild the store using ONLY the new
                # document being indexed. Other documents
                # will be re-indexed when requested.
                self.chunk_ids = list(
                    new_chunk_ids
                )

                self.document_ids = list(
                    new_doc_ids
                )

                self.chunks_data = list(
                    new_chunks_data
                )

                self.vectors = embeddings.copy()

            else:

                self.chunk_ids.extend(
                    new_chunk_ids
                )

                self.document_ids.extend(
                    new_doc_ids
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

    # ---------------------------------------------------------
    # DELETE DOCUMENT VECTORS
    # ---------------------------------------------------------

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

        # Make sure metadata and vectors have
        # consistent lengths.
        total_vectors = len(
            self.vectors
        )

        total_documents = len(
            self.document_ids
        )

        usable_count = min(
            total_vectors,
            total_documents
        )

        if usable_count == 0:
            return 0

        indices_to_keep = [
            i
            for i in range(usable_count)
            if self.document_ids[i]
            != document_id
        ]

        deleted_count = (
            usable_count
            - len(indices_to_keep)
        )

        # -----------------------------------------------------
        # KEEP NOTHING
        # -----------------------------------------------------

        if len(indices_to_keep) == 0:

            self.chunk_ids = []
            self.document_ids = []
            self.chunks_data = []
            self.vectors = None

        # -----------------------------------------------------
        # KEEP SOME VECTORS
        # -----------------------------------------------------

        else:

            self.chunk_ids = [
                self.chunk_ids[i]
                for i in indices_to_keep
                if i < len(self.chunk_ids)
            ]

            self.document_ids = [
                self.document_ids[i]
                for i in indices_to_keep
            ]

            self.chunks_data = [
                self.chunks_data[i]
                for i in indices_to_keep
                if i < len(self.chunks_data)
            ]

            self.vectors = self.vectors[
                indices_to_keep
            ]

        if auto_save:
            self.save()

        return deleted_count

    # ---------------------------------------------------------
    # SIMILARITY SEARCH
    # ---------------------------------------------------------

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

        query_vec = np.asarray(
            query_embedding,
            dtype=np.float32
        )

        if query_vec.ndim == 1:
            query_vec = query_vec.reshape(
                1, -1
            )

        stored_vectors = np.asarray(
            self.vectors,
            dtype=np.float32
        )

        if stored_vectors.ndim == 1:
            stored_vectors = stored_vectors.reshape(
                1, -1
            )

        # -----------------------------------------------------
        # CRITICAL DIMENSION CHECK
        # -----------------------------------------------------

        if (
            stored_vectors.shape[1]
            != query_vec.shape[1]
        ):

            raise ValueError(
                "Embedding dimension mismatch. "
                f"Stored vectors have dimension "
                f"{stored_vectors.shape[1]}, "
                f"but query has dimension "
                f"{query_vec.shape[1]}. "
                "Please re-index the documents."
            )

        # Cosine similarity because embeddings
        # are L2 normalized.
        scores = np.dot(
            stored_vectors,
            query_vec.T
        ).flatten()

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

        for idx in top_indices:

            if idx >= len(
                self.chunks_data
            ):
                continue

            score = float(
                scores[idx]
            )

            results.append(
                (
                    self.chunks_data[idx],
                    score
                )
            )

        return results

    # ---------------------------------------------------------
    # STATS
    # ---------------------------------------------------------

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

    # ---------------------------------------------------------
    # SAVE
    # ---------------------------------------------------------

    def save(self) -> None:

        vec_file = (
            self.persist_dir
            / "vectors.npy"
        )

        meta_file = (
            self.persist_dir
            / "metadata.json"
        )

        if (
            self.vectors is not None
            and len(self.chunk_ids) > 0
        ):

            np.save(
                str(vec_file),
                self.vectors
            )

            data = {
                "chunk_ids":
                    self.chunk_ids,

                "document_ids":
                    self.document_ids,

                "chunks_data":
                    self.chunks_data
            }

            with open(
                meta_file,
                "w",
                encoding="utf-8"
            ) as f:

                json.dump(
                    data,
                    f,
                    indent=2
                )

        else:

            if vec_file.exists():
                vec_file.unlink()

            if meta_file.exists():
                meta_file.unlink()

    # ---------------------------------------------------------
    # LOAD
    # ---------------------------------------------------------

    def load(self) -> bool:

        vec_file = (
            self.persist_dir
            / "vectors.npy"
        )

        meta_file = (
            self.persist_dir
            / "metadata.json"
        )

        if not (
            vec_file.exists()
            and meta_file.exists()
        ):

            self.chunk_ids = []
            self.document_ids = []
            self.chunks_data = []
            self.vectors = None

            return False

        try:

            self.vectors = np.load(
                str(vec_file)
            )

            with open(
                meta_file,
                "r",
                encoding="utf-8"
            ) as f:

                data = json.load(f)

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

            # Validate loaded store.
            if (
                self.vectors is not None
                and self.vectors.ndim == 1
            ):

                self.vectors = (
                    self.vectors.reshape(
                        1, -1
                    )
                )

            # If metadata and vector count are
            # inconsistent, reset the store.
            if (
                self.vectors is not None
                and len(self.vectors)
                != len(self.chunk_ids)
            ):

                print(
                    "[NexusRAG] Invalid vector store "
                    "detected. Resetting."
                )

                self.chunk_ids = []
                self.document_ids = []
                self.chunks_data = []
                self.vectors = None

                return False

            return True

        except Exception as e:

            print(
                f"Error loading vector store: {e}"
            )

            self.chunk_ids = []
            self.document_ids = []
            self.chunks_data = []
            self.vectors = None

            return False

    # ---------------------------------------------------------
    # CLEAR
    # ---------------------------------------------------------

    def clear(self) -> None:

        self.chunk_ids = []
        self.document_ids = []
        self.chunks_data = []
        self.vectors = None

        for f in self.persist_dir.glob("*"):

            try:
                f.unlink()

            except Exception:
                pass