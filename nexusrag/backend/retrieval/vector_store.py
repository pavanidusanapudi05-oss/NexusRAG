import logging
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
        Safely performs similarity search and extracts document metadata
        without crashing on tuple unpacking or missing key structure.
        """
        try:
            results = None
            if self.collection:
                results = self.collection.query(
                    query_embeddings=[query_embedding],
                    n_results=top_k
                )
            elif self.index:
                results = self.index.search(query_embedding, k=top_k)
            else:
                logger.warning("No vector store collection or index initialized.")
                return []

            parsed_results = []

            # 1. Handle ChromaDB / Dictionary structured outputs
            if isinstance(results, dict):
                ids = results.get("ids", [[]])[0] if results.get("ids") else []
                documents = results.get("documents", [[]])[0] if results.get("documents") else []
                metadatas = results.get("metadatas", [[]])[0] if results.get("metadatas") else []
                distances = results.get("distances", [[]])[0] if results.get("distances") else []

                for i in range(len(ids)):
                    doc_id = ids[i] if i < len(ids) else "unknown_id"
                    doc_text = documents[i] if i < len(documents) else ""
                    meta = metadatas[i] if (i < len(metadatas) and isinstance(metadatas[i], dict)) else {}
                    dist = distances[i] if i < len(distances) else 0.0

                    document_id = meta.get("document_id") or meta.get("id") or doc_id

                    parsed_results.append({
                        "chunk_id": doc_id,
                        "document_id": document_id,
                        "text": doc_text,
                        "metadata": meta,
                        "score": dist
                    })

            # 2. Handle List structured outputs (FAISS / Tuples / Custom Dicts)
            elif isinstance(results, list):
                for item in results:
                    if item is None:
                        continue

                    if isinstance(item, dict):
                        doc_text = item.get("text") or item.get("page_content", "")
                        meta = item.get("metadata", {})
                        doc_id = item.get("chunk_id") or item.get("id", "unknown_id")
                        score = item.get("score", 0.0)
                    elif isinstance(item, tuple):
                        doc = item[0] if len(item) > 0 else {}
                        score = item[1] if len(item) > 1 else 0.0
                        if isinstance(doc, dict):
                            doc_text = doc.get("text", "")
                            meta = doc.get("metadata", {})
                            doc_id = doc.get("id", "unknown_id")
                        else:
                            doc_text = getattr(doc, "page_content", str(doc))
                            meta = getattr(doc, "metadata", {})
                            doc_id = "unknown_id"
                    else:
                        doc_text = getattr(item, "page_content", str(item))
                        meta = getattr(item, "metadata", {})
                        doc_id = "unknown_id"
                        score = 0.0

                    if not isinstance(meta, dict):
                        meta = {}

                    document_id = meta.get("document_id") or meta.get("id") or doc_id

                    parsed_results.append({
                        "chunk_id": doc_id,
                        "document_id": document_id,
                        "text": doc_text,
                        "metadata": meta,
                        "score": score
                    })

            return parsed_results

        except Exception as e:
            logger.error(f"Error during similarity_search: {str(e)}")
            return []