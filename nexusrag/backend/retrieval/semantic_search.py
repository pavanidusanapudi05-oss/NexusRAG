from typing import List, Dict, Any, Tuple
import numpy as np
from nexusrag.backend.retrieval.vector_store import LocalVectorStore
from nexusrag.backend.retrieval.embedding_service import BaseEmbeddingProvider

class SemanticSearch:
    def __init__(self, vector_store: LocalVectorStore, embedding_provider: BaseEmbeddingProvider):
        self.vector_store = vector_store
        self.embedding_provider = embedding_provider

    def search(self, query: str, top_k: int = 5) -> List[Tuple[Dict[str, Any], float]]:
        if not query.strip() or self.vector_store.get_total_vectors() == 0:
            return []

        if hasattr(self.embedding_provider, "fit") and not getattr(self.embedding_provider, "is_fitted", False):
            all_texts = [d.get("text", "") for d in self.vector_store.chunks_data]
            if all_texts:
                self.embedding_provider.fit(all_texts)

        query_emb = self.embedding_provider.embed_query(query)
        return self.vector_store.similarity_search(query_emb, top_k=top_k)
