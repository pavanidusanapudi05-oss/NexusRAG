import re
from typing import List, Tuple, Dict, Any, Optional
from rank_bm25 import BM25Okapi
from nexusrag.backend.ingestion.chunker import DocumentChunk

class BM25Retriever:
    def __init__(self, chunks: Optional[List[DocumentChunk]] = None):
        self.chunks: List[DocumentChunk] = []
        self.bm25: Optional[BM25Okapi] = None
        self.corpus_tokens: List[List[str]] = []
        if chunks:
            self.index_chunks(chunks)

    @staticmethod
    def tokenize(text: str) -> List[str]:
        # Lowercase and extract alphanumeric words
        clean_text = text.lower().replace('-', ' ').replace('_', ' ')
        tokens = re.findall(r'\b[a-z0-9]{2,}\b', clean_text)
        
        # Also include un-hyphenated compact tokens (e.g. sr402, v2)
        compact_tokens = re.findall(r'\b[a-z0-9]+\b', text.lower().replace('-', ''))
        all_tokens = list(set(tokens + compact_tokens))
        return all_tokens

    def index_chunks(self, chunks: List[DocumentChunk]):
        self.chunks = list(chunks)
        self.corpus_tokens = [self.tokenize(c.content + ' ' + c.section_title + ' ' + c.doc_name) for c in self.chunks]
        if self.corpus_tokens and any(len(t) > 0 for t in self.corpus_tokens):
            self.bm25 = BM25Okapi(self.corpus_tokens)
        else:
            self.bm25 = None

    def search(self, query: str, top_k: int = 10) -> List[Tuple[DocumentChunk, float]]:
        if not self.bm25 or not self.chunks:
            return []
        
        query_tokens = self.tokenize(query)
        if not query_tokens:
            return []
            
        scores = self.bm25.get_scores(query_tokens)
        
        # Normalize BM25 scores between 0 and 1
        max_score = max(scores) if len(scores) > 0 else 1.0
        if max_score <= 0:
            max_score = 1.0
            
        top_indices = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)[:top_k]
        results = []
        for idx in top_indices:
            raw_score = float(scores[idx])
            norm_score = raw_score / max_score if raw_score > 0 else 0.0
            results.append((self.chunks[idx], norm_score))
        return results
