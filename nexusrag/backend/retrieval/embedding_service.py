import os
import numpy as np
from typing import List
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import normalize


class BaseEmbeddingProvider:

    def embed_texts(self, texts: List[str]) -> np.ndarray:
        raise NotImplementedError

    def embed_query(self, query: str) -> np.ndarray:
        raise NotImplementedError


class LocalDenseEmbeddingProvider(BaseEmbeddingProvider):
    """
    Local TF-IDF embedding provider.

    The same vectorizer vocabulary is used for both
    documents and queries, preventing dimension mismatch.
    """

    def __init__(self, max_features: int = 2048):
        self.max_features = max_features

        self.vectorizer = TfidfVectorizer(
            max_features=max_features,
            ngram_range=(1, 2),
            sublinear_tf=True,
            stop_words="english"
        )

        self.is_fitted = False
        self.dimension = 0

    def fit(self, texts: List[str]):
        if not texts:
            return

        clean_texts = [
            str(text) if text is not None else ""
            for text in texts
        ]

        self.vectorizer.fit(clean_texts)

        self.dimension = len(
            self.vectorizer.vocabulary_
        )

        self.is_fitted = True

    def embed_texts(self, texts: List[str]) -> np.ndarray:

        if not texts:
            return np.empty(
                (
                    0,
                    self.dimension
                    if self.dimension > 0
                    else self.max_features
                ),
                dtype=np.float32
            )

        if not self.is_fitted:
            self.fit(texts)

        matrix = self.vectorizer.transform(texts)

        embeddings = matrix.toarray().astype(
            np.float32
        )

        return normalize(
            embeddings,
            norm="l2",
            axis=1
        )

    def embed_query(self, query: str) -> np.ndarray:

        if not self.is_fitted:
            raise ValueError(
                "Embedding provider is not fitted. "
                "Index documents before searching."
            )

        query = str(query).strip()

        if not query:
            return np.zeros(
                (1, self.dimension),
                dtype=np.float32
            )

        # IMPORTANT:
        # Use the SAME vocabulary that was fitted
        # on the indexed documents.
        matrix = self.vectorizer.transform(
            [query]
        )

        embedding = matrix.toarray().astype(
            np.float32
        )

        return normalize(
            embedding,
            norm="l2",
            axis=1
        )


class GeminiEmbeddingProvider(BaseEmbeddingProvider):

    def __init__(
        self,
        api_key: str,
        model_name: str = "models/text-embedding-004"
    ):
        import google.generativeai as genai

        self.api_key = api_key
        self.model_name = model_name

        genai.configure(api_key=api_key)

        self.genai = genai

    def embed_texts(
        self,
        texts: List[str]
    ) -> np.ndarray:

        embeddings = []

        for text in texts:

            res = self.genai.embed_content(
                model=self.model_name,
                content=text,
                task_type="retrieval_document"
            )

            embeddings.append(
                res["embedding"]
            )

        arr = np.array(
            embeddings,
            dtype=np.float32
        )

        return normalize(
            arr,
            norm="l2",
            axis=1
        )

    def embed_query(
        self,
        query: str
    ) -> np.ndarray:

        res = self.genai.embed_content(
            model=self.model_name,
            content=query,
            task_type="retrieval_query"
        )

        arr = np.array(
            [res["embedding"]],
            dtype=np.float32
        )

        return normalize(
            arr,
            norm="l2",
            axis=1
        )


class OpenAIEmbeddingProvider(BaseEmbeddingProvider):

    def __init__(
        self,
        api_key: str,
        model_name: str = "text-embedding-3-small"
    ):
        from openai import OpenAI

        self.client = OpenAI(
            api_key=api_key
        )

        self.model_name = model_name

    def embed_texts(
        self,
        texts: List[str]
    ) -> np.ndarray:

        resp = self.client.embeddings.create(
            input=texts,
            model=self.model_name
        )

        embeddings = [
            d.embedding
            for d in resp.data
        ]

        arr = np.array(
            embeddings,
            dtype=np.float32
        )

        return normalize(
            arr,
            norm="l2",
            axis=1
        )

    def embed_query(
        self,
        query: str
    ) -> np.ndarray:

        resp = self.client.embeddings.create(
            input=[query],
            model=self.model_name
        )

        arr = np.array(
            [resp.data[0].embedding],
            dtype=np.float32
        )

        return normalize(
            arr,
            norm="l2",
            axis=1
        )


class EmbeddingService:

    @staticmethod
    def create_provider(
        provider_type: str = "local_dense",
        api_key: str = "",
        model_name: str = ""
    ) -> BaseEmbeddingProvider:

        prov = provider_type.lower()

        if prov == "gemini" and api_key:

            try:
                return GeminiEmbeddingProvider(
                    api_key=api_key,
                    model_name=(
                        model_name
                        or "models/text-embedding-004"
                    )
                )

            except Exception as e:

                print(
                    f"[Warning] Failed to initialize "
                    f"Gemini embeddings ({e}). "
                    f"Falling back to local dense embeddings."
                )

        elif prov == "openai" and api_key:

            try:
                return OpenAIEmbeddingProvider(
                    api_key=api_key,
                    model_name=(
                        model_name
                        or "text-embedding-3-small"
                    )
                )

            except Exception as e:

                print(
                    f"[Warning] Failed to initialize "
                    f"OpenAI embeddings ({e}). "
                    f"Falling back to local dense embeddings."
                )

        return LocalDenseEmbeddingProvider()