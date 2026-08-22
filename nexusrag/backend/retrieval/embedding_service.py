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

    IMPORTANT:
    The same fitted TF-IDF vocabulary is always used for
    both document embeddings and query embeddings.

    This prevents dimension mismatch errors such as:

        shapes (30,277) and (944,1) not aligned
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

    # ============================================================
    # FIT
    # ============================================================

    def fit(self, texts: List[str]) -> None:

        clean_texts = [
            str(text).strip()
            for text in texts
            if text is not None and str(text).strip()
        ]

        if not clean_texts:
            self.is_fitted = False
            self.dimension = 0
            return

        self.vectorizer.fit(clean_texts)

        self.is_fitted = True

        self.dimension = len(
            self.vectorizer.vocabulary_
        )

    # ============================================================
    # EMBED DOCUMENT TEXTS
    # ============================================================

    def embed_texts(
        self,
        texts: List[str]
    ) -> np.ndarray:

        if not texts:

            dimension = (
                self.dimension
                if self.dimension > 0
                else self.max_features
            )

            return np.empty(
                (0, dimension),
                dtype=np.float32
            )

        # If provider has not been fitted yet,
        # fit it using these texts.
        if not self.is_fitted:

            self.fit(texts)

        matrix = self.vectorizer.transform(
            [
                str(text)
                for text in texts
            ]
        )

        embeddings = matrix.toarray().astype(
            np.float32
        )

        embeddings = normalize(
            embeddings,
            norm="l2",
            axis=1
        )

        return embeddings.astype(
            np.float32
        )

    # ============================================================
    # EMBED QUERY
    # ============================================================

    def embed_query(
        self,
        query: str
    ) -> np.ndarray:

        if not query or not query.strip():

            dimension = (
                self.dimension
                if self.dimension > 0
                else self.max_features
            )

            return np.zeros(
                (1, dimension),
                dtype=np.float32
            )

        if not self.is_fitted:

            raise ValueError(
                "Local TF-IDF embedding provider is not fitted. "
                "Fit it using the indexed document corpus first."
            )

        matrix = self.vectorizer.transform(
            [query]
        )

        embedding = matrix.toarray().astype(
            np.float32
        )

        embedding = normalize(
            embedding,
            norm="l2",
            axis=1
        )

        return embedding.astype(
            np.float32
        )


# ================================================================
# GEMINI EMBEDDING PROVIDER
# ================================================================

class GeminiEmbeddingProvider(BaseEmbeddingProvider):

    def __init__(
        self,
        api_key: str,
        model_name: str = "models/text-embedding-004"
    ):

        import google.generativeai as genai

        self.api_key = api_key
        self.model_name = model_name

        genai.configure(
            api_key=api_key
        )

        self.genai = genai

    def embed_texts(
        self,
        texts: List[str]
    ) -> np.ndarray:

        embeddings = []

        for text in texts:

            response = self.genai.embed_content(
                model=self.model_name,
                content=text,
                task_type="retrieval_document"
            )

            embeddings.append(
                response["embedding"]
            )

        arr = np.asarray(
            embeddings,
            dtype=np.float32
        )

        return normalize(
            arr,
            norm="l2",
            axis=1
        ).astype(np.float32)

    def embed_query(
        self,
        query: str
    ) -> np.ndarray:

        response = self.genai.embed_content(
            model=self.model_name,
            content=query,
            task_type="retrieval_query"
        )

        arr = np.asarray(
            [response["embedding"]],
            dtype=np.float32
        )

        return normalize(
            arr,
            norm="l2",
            axis=1
        ).astype(np.float32)


# ================================================================
# OPENAI EMBEDDING PROVIDER
# ================================================================

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

        response = self.client.embeddings.create(
            input=texts,
            model=self.model_name
        )

        embeddings = [
            item.embedding
            for item in response.data
        ]

        arr = np.asarray(
            embeddings,
            dtype=np.float32
        )

        return normalize(
            arr,
            norm="l2",
            axis=1
        ).astype(np.float32)

    def embed_query(
        self,
        query: str
    ) -> np.ndarray:

        response = self.client.embeddings.create(
            input=[query],
            model=self.model_name
        )

        arr = np.asarray(
            [response.data[0].embedding],
            dtype=np.float32
        )

        return normalize(
            arr,
            norm="l2",
            axis=1
        ).astype(np.float32)


# ================================================================
# EMBEDDING SERVICE FACTORY
# ================================================================

class EmbeddingService:

    @staticmethod
    def create_provider(
        provider_type: str = "local_dense",
        api_key: str = "",
        model_name: str = ""
    ) -> BaseEmbeddingProvider:

        provider = (
            provider_type or "local_dense"
        ).lower().strip()

        # --------------------------------------------------------
        # GEMINI
        # --------------------------------------------------------

        if provider == "gemini" and api_key:

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
                    "[NexusRAG] Gemini embedding initialization "
                    f"failed: {e}. Falling back to local TF-IDF."
                )

        # --------------------------------------------------------
        # OPENAI
        # --------------------------------------------------------

        if provider == "openai" and api_key:

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
                    "[NexusRAG] OpenAI embedding initialization "
                    f"failed: {e}. Falling back to local TF-IDF."
                )

        # --------------------------------------------------------
        # LOCAL
        # --------------------------------------------------------

        return LocalDenseEmbeddingProvider(
            max_features=2048
        )