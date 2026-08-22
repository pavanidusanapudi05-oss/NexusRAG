import numpy as np
from typing import List
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import normalize

class BaseEmbeddingProvider:

```
def embed_texts(self, texts: List[str]) -> np.ndarray:
    raise NotImplementedError

def embed_query(self, query: str) -> np.ndarray:
    raise NotImplementedError
```

class LocalDenseEmbeddingProvider(BaseEmbeddingProvider):

```
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

    clean_texts = [
        str(text)
        for text in texts
        if str(text).strip()
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

    if not self.is_fitted:
        self.fit(texts)

    if not self.is_fitted:
        raise ValueError(
            "Cannot create embeddings because "
            "the TF-IDF provider has no usable text corpus."
        )

    matrix = (
        self.vectorizer
        .transform(
            [
                str(text)
                for text in texts
            ]
        )
        .toarray()
        .astype(np.float32)
    )

    return normalize(
        matrix,
        norm="l2",
        axis=1
    ).astype(np.float32)

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
            "Embedding provider is not fitted. "
            "The document corpus must be fitted "
            "before creating query embeddings."
        )

    matrix = (
        self.vectorizer
        .transform([query])
        .toarray()
        .astype(np.float32)
    )

    return normalize(
        matrix,
        norm="l2",
        axis=1
    ).astype(np.float32)
```

class GeminiEmbeddingProvider(BaseEmbeddingProvider):

```
def __init__(
    self,
    api_key: str,
    model_name: str = "models/text-embedding-004"
):

    import google.generativeai as genai

    if not api_key:
        raise ValueError(
            "Gemini API key is missing."
        )

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

        result = self.genai.embed_content(
            model=self.model_name,
            content=str(text),
            task_type="retrieval_document"
        )

        embeddings.append(
            result["embedding"]
        )

    if not embeddings:
        return np.empty(
            (0, 0),
            dtype=np.float32
        )

    matrix = np.asarray(
        embeddings,
        dtype=np.float32
    )

    return normalize(
        matrix,
        norm="l2",
        axis=1
    ).astype(np.float32)

def embed_query(
    self,
    query: str
) -> np.ndarray:

    result = self.genai.embed_content(
        model=self.model_name,
        content=query,
        task_type="retrieval_query"
    )

    matrix = np.asarray(
        [result["embedding"]],
        dtype=np.float32
    )

    return normalize(
        matrix,
        norm="l2",
        axis=1
    ).astype(np.float32)
```

class OpenAIEmbeddingProvider(BaseEmbeddingProvider):

```
def __init__(
    self,
    api_key: str,
    model_name: str = "text-embedding-3-small"
):

    from openai import OpenAI

    if not api_key:
        raise ValueError(
            "OpenAI API key is missing."
        )

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

    matrix = np.asarray(
        embeddings,
        dtype=np.float32
    )

    return normalize(
        matrix,
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

    matrix = np.asarray(
        [response.data[0].embedding],
        dtype=np.float32
    )

    return normalize(
        matrix,
        norm="l2",
        axis=1
    ).astype(np.float32)
```

class EmbeddingService:

```
@staticmethod
def create_provider(
    provider_type: str = "local_dense",
    api_key: str = "",
    model_name: str = ""
) -> BaseEmbeddingProvider:

    provider = (
        provider_type or "local_dense"
    ).strip().lower()

    if provider == "gemini":

        if not api_key:
            print(
                "[NexusRAG] Gemini API key not found. "
                "Using local_dense embeddings."
            )
        else:
            try:
                return GeminiEmbeddingProvider(
                    api_key=api_key,
                    model_name=(
                        model_name
                        or "models/text-embedding-004"
                    )
                )
            except Exception as error:
                print(
                    "[NexusRAG] Gemini embedding initialization "
                    f"failed: {error}"
                )

    elif provider == "openai":

        if not api_key:
            print(
                "[NexusRAG] OpenAI API key not found. "
                "Using local_dense embeddings."
            )
        else:
            try:
                return OpenAIEmbeddingProvider(
                    api_key=api_key,
                    model_name=(
                        model_name
                        or "text-embedding-3-small"
                    )
                )
            except Exception as error:
                print(
                    "[NexusRAG] OpenAI embedding initialization "
                    f"failed: {error}"
                )

    return LocalDenseEmbeddingProvider()
```
