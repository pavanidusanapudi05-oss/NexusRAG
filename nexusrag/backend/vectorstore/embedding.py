import os
import numpy as np
from typing import List, Optional
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import TruncatedSVD
from sklearn.preprocessing import normalize

class BaseEmbeddingProvider:
    def embed_texts(self, texts: List[str]) -> np.ndarray:
        raise NotImplementedError
        
    def embed_query(self, query: str) -> np.ndarray:
        raise NotImplementedError

class LocalDenseEmbeddingProvider(BaseEmbeddingProvider):
    def __init__(self, n_components: int = 64):
        self.vectorizer = TfidfVectorizer(
            max_features=2500,
            ngram_range=(1, 2),
            sublinear_tf=True,
            stop_words='english'
        )
        self.svd = TruncatedSVD(n_components=n_components, random_state=42)
        self.is_fitted = False

    def fit(self, texts: List[str]):
        if not texts:
            return
        tfidf_matrix = self.vectorizer.fit_transform(texts)
        n_comp = min(self.svd.n_components, max(2, tfidf_matrix.shape[1] - 1))
        self.svd.n_components = n_comp
        self.svd.fit(tfidf_matrix)
        self.is_fitted = True

    def embed_texts(self, texts: List[str]) -> np.ndarray:
        if not texts:
            return np.empty((0, self.svd.n_components))
        if not self.is_fitted:
            self.fit(texts)
        tfidf_matrix = self.vectorizer.transform(texts)
        dense_vectors = self.svd.transform(tfidf_matrix)
        # Normalize to unit length for cosine similarity
        normed = normalize(dense_vectors, norm='l2', axis=1)
        return normed

    def embed_query(self, query: str) -> np.ndarray:
        if not self.is_fitted:
            return np.zeros((1, self.svd.n_components))
        tfidf_matrix = self.vectorizer.transform([query])
        dense_vector = self.svd.transform(tfidf_matrix)
        normed = normalize(dense_vector, norm='l2', axis=1)
        return normed

class GeminiEmbeddingProvider(BaseEmbeddingProvider):
    def __init__(self, api_key: str, model_name: str = 'models/text-embedding-004'):
        import google.generativeai as genai
        self.api_key = api_key
        self.model_name = model_name
        genai.configure(api_key=api_key)
        self.genai = genai

    def embed_texts(self, texts: List[str]) -> np.ndarray:
        embeddings = []
        for text in texts:
            res = self.genai.embed_content(
                model=self.model_name,
                content=text,
                task_type='retrieval_document'
            )
            embeddings.append(res['embedding'])
        arr = np.array(embeddings, dtype=np.float32)
        return normalize(arr, norm='l2', axis=1)

    def embed_query(self, query: str) -> np.ndarray:
        res = self.genai.embed_content(
            model=self.model_name,
            content=query,
            task_type='retrieval_query'
        )
        arr = np.array([res['embedding']], dtype=np.float32)
        return normalize(arr, norm='l2', axis=1)

class EmbeddingFactory:
    @staticmethod
    def get_provider(provider_type: str = 'offline', api_key: str = '') -> BaseEmbeddingProvider:
        if provider_type == 'gemini' and api_key:
            try:
                return GeminiEmbeddingProvider(api_key=api_key)
            except Exception:
                return LocalDenseEmbeddingProvider()
        return LocalDenseEmbeddingProvider()
