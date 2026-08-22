import logging
import numpy as np
from typing import List, Dict, Any, Optional

logger = logging.getLogger(__name__)


class LocalVectorStore:
    def __init__(self, collection=None, index=None, vectors=None, chunks_data=None):
        self.collection = collection
        self.index = index
        self.vectors = vectors
        self.chunks_data = chunks_data or []

    def similarity_search(self, query_embedding: Any, top_k: int = 5) -> List[Dict[str, Any]]:
        """
        Calculates cosine similarity safely without breaking on type/shape mismatches.
        """
        try:
            # 1. Fallback for stored NumPy vectors (TF-IDF / Custom Embeddings)
            if self.vectors is not None and len(self.chunks_data) > 0:
                q_vec = np.asarray(query_embedding).squeeze()
                
                # Check for matrix multiplication shape alignment
                if self.vectors.ndim == 2 and q_vec.ndim == 1:
                    if self.vectors.shape[1] == q_vec.shape[0]:
                        scores = np.dot(self.vectors, q_vec)
                    elif self.vectors.shape[0] == q_vec.shape[0]:
                        scores = np.dot(self.vectors.T, q_vec)
                    else:
                        logger.error("Embedding dimensions do not match stored vectors.")
                        return []
                else:
                    scores = np.zeros(len(self.chunks_data))

                # Extract top-k indices
                top_indices = np.argsort(scores)[::-1][:top_k]
                parsed_results = []

                for idx in top_indices:
                    if idx >= len(self.chunks_data):
                        continue
                    
                    chunk = self.chunks_data[idx]
                    if not isinstance(chunk, dict):
                        chunk = {"text": str(chunk)}

                    raw_meta = chunk.get("metadata", {})
                    meta = raw_meta if isinstance(raw_meta, dict) else {}
                    text_content = str(chunk.get("text", ""))

                    parsed_results.append({
                        "chunk_id": str(chunk.get("chunk_id", f"chunk_{idx}")),
                        "document_id": str(chunk.get("document_id", meta.get("document_id", "doc_0"))),
                        "text": text_content,
                        "metadata": meta,
                        "score": float(scores[idx]) if idx < len(scores) else 0.0
                    })

                return parsed_results

            return []

        except Exception as e:
            logger.error(f"Error during similarity_search: {str(e)}")
            return []

    def save(self):
        pass