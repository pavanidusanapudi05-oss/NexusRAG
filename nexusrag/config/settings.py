import os
from pathlib import Path
from dataclasses import dataclass, field


@dataclass
class NexusSettings:

    # ============================================================
    # BASE PATHS
    # ============================================================

    base_dir: Path = field(
        default_factory=lambda: Path(
            os.getenv(
                "NEXUS_BASE_DIR",
                Path.cwd()
            )
        )
    )

    data_dir: Path = field(
        default_factory=lambda: Path(
            os.getenv(
                "NEXUS_DATA_DIR",
                str(Path.cwd() / "nexusrag" / "data")
            )
        )
    )

    docs_dir: Path = field(
        default_factory=lambda: Path(
            os.getenv(
                "NEXUS_DOCS_DIR",
                str(
                    Path.cwd()
                    / "nexusrag"
                    / "data"
                    / "documents"
                )
            )
        )
    )

    vector_db_path: Path = field(
        default_factory=lambda: Path(
            os.getenv(
                "VECTOR_DB_PATH",
                str(
                    Path.cwd()
                    / "nexusrag"
                    / "data"
                    / "vector_store"
                )
            )
        )
    )

    sqlite_db_path: Path = field(
        default_factory=lambda: Path(
            os.getenv(
                "SQLITE_DB_PATH",
                str(
                    Path.cwd()
                    / "nexusrag"
                    / "data"
                    / "nexusrag.db"
                )
            )
        )
    )

    graph_db_path: Path = field(
        default_factory=lambda: Path(
            os.getenv(
                "GRAPH_DB_PATH",
                str(
                    Path.cwd()
                    / "nexusrag"
                    / "data"
                    / "graph"
                    / "knowledge_graph.json"
                )
            )
        )
    )

    # ============================================================
    # INGESTION & CHUNKING
    # ============================================================

    chunk_size: int = field(
        default_factory=lambda: int(
            os.getenv("CHUNK_SIZE", "600")
        )
    )

    chunk_overlap: int = field(
        default_factory=lambda: int(
            os.getenv("CHUNK_OVERLAP", "120")
        )
    )

    max_upload_size_mb: int = field(
        default_factory=lambda: int(
            os.getenv("MAX_UPLOAD_SIZE_MB", "50")
        )
    )

    # ============================================================
    # RETRIEVAL & RERANKING
    # ============================================================

    retrieval_top_k: int = field(
        default_factory=lambda: int(
            os.getenv(
                "RETRIEVAL_TOP_K",
                os.getenv("TOP_K", "5")
            )
        )
    )

    semantic_weight: float = field(
        default_factory=lambda: float(
            os.getenv("SEMANTIC_WEIGHT", "0.7")
        )
    )

    keyword_weight: float = field(
        default_factory=lambda: float(
            os.getenv("KEYWORD_WEIGHT", "0.3")
        )
    )

    reranker_enabled: bool = field(
        default_factory=lambda:
            os.getenv(
                "RERANKING_ENABLED",
                "true"
            ).lower()
            in ("true", "1", "yes")
    )

    similarity_threshold: float = field(
        default_factory=lambda: float(
            os.getenv(
                "SIMILARITY_THRESHOLD",
                "0.15"
            )
        )
    )

    # ============================================================
    # LLM CONFIGURATION
    # ============================================================

    llm_provider: str = field(
        default_factory=lambda:
            os.getenv(
                "LLM_PROVIDER",
                "offline"
            )
    )

    llm_model: str = field(
        default_factory=lambda:
            os.getenv(
                "LLM_MODEL",
                "gemini-2.0-flash"
            )
    )

    gemini_api_key: str = field(
        default_factory=lambda:
            os.getenv(
                "GEMINI_API_KEY",
                ""
            )
    )

    openai_api_key: str = field(
        default_factory=lambda:
            os.getenv(
                "OPENAI_API_KEY",
                ""
            )
    )

    gemini_model: str = field(
        default_factory=lambda:
            os.getenv(
                "GEMINI_MODEL",
                "gemini-2.0-flash"
            )
    )

    openai_model: str = field(
        default_factory=lambda:
            os.getenv(
                "OPENAI_MODEL",
                "gpt-4o-mini"
            )
    )

    # ============================================================
    # EMBEDDING CONFIGURATION
    # ============================================================

    embedding_provider: str = field(
        default_factory=lambda:
            os.getenv(
                "EMBEDDING_PROVIDER",
                "local_dense"
            )
    )

    embedding_model: str = field(
        default_factory=lambda:
            os.getenv(
                "EMBEDDING_MODEL",
                "models/text-embedding-004"
            )
    )


# ================================================================
# COMPATIBILITY ALIASES
# ================================================================

Settings = NexusSettings

settings = NexusSettings()

__all__ = [
    "NexusSettings",
    "Settings",
    "settings"
]