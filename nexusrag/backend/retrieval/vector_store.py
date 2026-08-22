import logging
from typing import List, Dict, Any, Optional

logger = logging.getLogger(__name__)

class VectorStore:
    def __init__(self, collection=None, index=None):
        self.collection = collection
        self.index = index

    def similarity_search(self, query_embedding: Any, top_k: int = 5) -> List[Dict[str, Any]]:
        """
        Safely performs similarity search and extracts document metadata
        without crashing on missing keys or None objects.
        """
        try:
            # Vector DB query execution
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

            # Handle ChromaDB / Dictionary structured outputs
            if isinstance(results, dict):
                ids = results.get("ids", [[]])[0]
                documents = results.get("documents", [[]])[0]
                metadatas = results.get("metadatas", [[]])[0]
                distances = results.get("distances", [[]])[0] if "distances" in results else [0.0] * len(ids)

                for doc_id, doc_text, meta, dist in zip(ids, documents, metadatas, distances):
                    safe_meta = meta if isinstance(meta, dict) else {}
                    
                    # Safe document_id extraction (Fixes Line 74 Crash)
                    document_id = safe_meta.get("document_id") or safe_meta.get("id") or doc_id

                    parsed_results.append({
                        "document_id": document_id,
                        "text": doc_text,
                        "metadata": safe_meta,
                        "score": dist
                    })

            # Handle List/Tuple structured outputs (FAISS / LangChain / Custom)
            elif isinstance(results, list):
                for item in results:
                    if item is None:
                        continue
                    
                    # If item is a tuple (doc, score)
                    if isinstance(item, tuple):
                        doc, score = item[0], item[1] if len(item) > 1 else 0.0
                    else:
                        doc, score = item, 0.0

                    # Extract text and metadata safely
                    if isinstance(doc, dict):
                        doc_text = doc.get("text") or doc.get("page_content", "")
                        meta = doc.get("metadata", doc)
                    else:
                        doc_text = getattr(doc, "page_content", str(doc))
                        meta = getattr(doc, "metadata", {})

                    if not isinstance(meta, dict):
                        meta = {}

                    # Safe document_id extraction (Fixes Line 74 Crash)
                    document_id = meta.get("document_id") or meta.get("id") or "unknown_id"

                    parsed_results.append({
                        "document_id": document_id,
                        "text": doc_text,
                        "metadata": meta,
                        "score": score
                    })

            return parsed_results

        except Exception as e:
            logger.error(f"Error during similarity_search: {str(e)}")
            return []