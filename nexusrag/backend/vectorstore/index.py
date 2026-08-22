import os
import json
from pathlib import Path
import numpy as np
from typing import List, Dict, Any, Tuple, Optional
from nexusrag.backend.ingestion.chunker import DocumentChunk
from .embedding import BaseEmbeddingProvider, EmbeddingFactory

class VectorStore:
    def __init__(self, persist_dir: Optional[str] = None, embedding_provider: Optional[BaseEmbeddingProvider] = None):
        self.persist_dir = Path(persist_dir) if persist_dir else Path('nexusrag/vectorstore')
        self.persist_dir.mkdir(parents=True, exist_ok=True)
        self.embedding_provider = embedding_provider or EmbeddingFactory.get_provider('offline')
        
        self.chunks: List[DocumentChunk] = []
        self.vectors: Optional[np.ndarray] = None
        self.chunk_ids: List[str] = []

    def add_chunks(self, chunks: List[DocumentChunk]):
        if not chunks:
            return
        
        new_texts = [c.content for c in chunks]
        new_vectors = self.embedding_provider.embed_texts(new_texts)
        
        if self.vectors is None or len(self.chunks) == 0:
            self.chunks = list(chunks)
            self.vectors = new_vectors
            self.chunk_ids = [c.chunk_id for c in chunks]
        else:
            self.chunks.extend(chunks)
            self.vectors = np.vstack([self.vectors, new_vectors])
            self.chunk_ids.extend([c.chunk_id for c in chunks])
            
        self.save()

    def search(self, query: str, top_k: int = 10) -> List[Tuple[DocumentChunk, float]]:
        if self.vectors is None or len(self.chunks) == 0:
            return []
            
        query_vector = self.embedding_provider.embed_query(query)
        # Cosine similarity between normalized vectors = dot product
        scores = np.dot(self.vectors, query_vector.T).flatten()
        
        # Sort descending
        top_indices = np.argsort(scores)[::-1][:top_k]
        results = []
        for idx in top_indices:
            score = float(scores[idx])
            results.append((self.chunks[idx], score))
        return results

    def save(self):
        if self.vectors is None or len(self.chunks) == 0:
            return
        
        # Save vectors
        vec_file = self.persist_dir / 'vectors.npy'
        np.save(str(vec_file), self.vectors)
        
        # Save chunks metadata
        meta_file = self.persist_dir / 'chunks.json'
        data = [c.to_dict() for c in self.chunks]
        with open(meta_file, 'w', encoding='utf-8') as f:
            json.dump(data, f, indent=2)

    def load(self) -> bool:
        vec_file = self.persist_dir / 'vectors.npy'
        meta_file = self.persist_dir / 'chunks.json'
        
        if not (vec_file.exists() and meta_file.exists()):
            return False
            
        try:
            self.vectors = np.load(str(vec_file))
            with open(meta_file, 'r', encoding='utf-8') as f:
                data = json.load(f)
            self.chunks = [DocumentChunk.from_dict(d) for d in data]
            self.chunk_ids = [c.chunk_id for c in self.chunks]
            
            # Re-fit local vectorizer if using offline provider
            if hasattr(self.embedding_provider, 'fit'):
                texts = [c.content for c in self.chunks]
                self.embedding_provider.fit(texts)
            return True
        except Exception as e:
            print(f'Error loading vector store: {e}')
            return False

    def clear(self):
        self.chunks = []
        self.vectors = None
        self.chunk_ids = []
        for f in self.persist_dir.glob('*'):
            if f.is_file():
                try:
                    f.unlink()
                except Exception:
                    pass
