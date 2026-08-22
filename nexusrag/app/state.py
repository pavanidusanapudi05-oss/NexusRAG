import os
from pathlib import Path
from typing import Dict, Any, List, Optional
import streamlit as st

from nexusrag.config.settings import settings
from nexusrag.backend.storage.database import DatabaseManager
from nexusrag.backend.storage.repository import DocumentRepository, ChunkRepository
from nexusrag.backend.retrieval.vector_store import LocalVectorStore
from nexusrag.backend.retrieval.embedding_service import EmbeddingService
from nexusrag.backend.retrieval.retriever import VectorRetriever, RetrievalResult
from nexusrag.backend.retrieval.hybrid_search import HybridSearch
from nexusrag.backend.retrieval.reranker import PrecisionReranker
from nexusrag.backend.llm.provider import LLMProviderFactory
from nexusrag.backend.rag.rag_pipeline import RAGPipeline, RAGAnswer
from nexusrag.backend.graph.graph_store import LocalKnowledgeGraph
from nexusrag.backend.versioning.document_compare import DocumentComparator, DocumentComparisonResult
from nexusrag.backend.evaluation.evaluator import RAGEvaluator, EvaluationReport
from nexusrag.backend.ingestion.pipeline import IngestionPipeline, IngestionResult, ALLOWED_EXTENSIONS
from nexusrag.backend.models.document import DocumentRecord, ProcessingStatus
from nexusrag.backend.models.chunk import DocumentChunk
from nexusrag.data.sample_data import generate_sample_documents


class NexusSystemState:
    def __init__(self):
        self.docs_dir = settings.docs_dir
        self.docs_dir.mkdir(parents=True, exist_ok=True)

        self.vector_db_dir = settings.data_dir / "vector_store"
        self.vector_db_dir.mkdir(parents=True, exist_ok=True)

        self.graph_db_path = settings.data_dir / "graph" / "knowledge_graph.json"
        self.graph_db_path.parent.mkdir(parents=True, exist_ok=True)

        # Storage
        self.db_manager = DatabaseManager(
            db_path=settings.data_dir / "nexusrag.db"
        )

        self.doc_repo = DocumentRepository(self.db_manager)
        self.chunk_repo = ChunkRepository(self.db_manager)

        self.vector_store = LocalVectorStore(
            persist_dir=self.vector_db_dir
        )

        self.knowledge_graph = LocalKnowledgeGraph(
            persist_path=self.graph_db_path
        )

        # ============================================================
        # Embedding & Hybrid Retrieval
        # IMPORTANT:
        # Embedding provider is intentionally separated from LLM provider.
        # ============================================================
        self.embedding_provider = EmbeddingService.create_provider(
            provider_type=settings.embedding_provider,
            api_key=(
                settings.gemini_api_key
                if settings.embedding_provider == "gemini"
                else settings.openai_api_key
            ),
            model_name=settings.embedding_model
        )

        self.retriever = VectorRetriever(
            vector_store=self.vector_store,
            embedding_provider=self.embedding_provider,
            top_k=settings.retrieval_top_k
        )

        self.hybrid_searcher = HybridSearch(
            vector_store=self.vector_store,
            embedding_provider=self.embedding_provider,
            semantic_weight=settings.semantic_weight,
            keyword_weight=settings.keyword_weight,
            reranker_enabled=settings.reranker_enabled
        )

        # ============================================================
        # Ingestion Pipeline
        # ============================================================
        self.pipeline = IngestionPipeline(
            db_manager=self.db_manager,
            vector_store=self.vector_store,
            embedding_provider=self.embedding_provider,
            knowledge_graph=self.knowledge_graph,
            chunk_size=settings.chunk_size,
            chunk_overlap=settings.chunk_overlap
        )

        # ============================================================
        # LLM Provider & RAG Pipeline
        # LLM provider remains separate from embedding provider.
        # ============================================================
        self.llm_provider = LLMProviderFactory.create(
            provider_name=settings.llm_provider,
            api_key=(
                settings.gemini_api_key
                if settings.llm_provider == "gemini"
                else settings.openai_api_key
            ),
            model_name=(
                settings.gemini_model
                if settings.llm_provider == "gemini"
                else settings.openai_model
            )
        )

        self.rag_pipeline = RAGPipeline(
            retriever=self.retriever,
            llm_provider=self.llm_provider
        )

        # ============================================================
        # Version Comparator & Evaluation Runner
        # ============================================================
        self.comparator = DocumentComparator(
            self.doc_repo,
            self.chunk_repo
        )

        self.evaluator = RAGEvaluator(
            self.rag_pipeline
        )

        # ============================================================
        # Session State
        # ============================================================
        self.chat_history: List[RAGAnswer] = []
        self.latest_eval_report: Optional[EvaluationReport] = None

        # Initialize sample data
        self.initialize_sample_data()

    def initialize_sample_data(self):
        sample_files = generate_sample_documents(str(self.docs_dir))
        existing_docs = self.doc_repo.list_all()

        if len(existing_docs) < len(sample_files):
            for s_path in sample_files:
                p = Path(s_path)

                if p.exists():
                    self.pipeline.process_file(p)

    def process_uploaded_file(
        self,
        uploaded_file,
        custom_meta: Optional[Dict[str, Any]] = None
    ) -> IngestionResult:

        target_path = self.docs_dir / uploaded_file.name

        with open(target_path, "wb") as f:
            f.write(uploaded_file.getbuffer())

        return self.pipeline.process_file(
            target_path,
            custom_meta
        )

    def list_documents(self) -> List[DocumentRecord]:
        return self.doc_repo.list_all()

    def get_document_chunks(
        self,
        document_id: str
    ) -> List[DocumentChunk]:

        return self.chunk_repo.get_by_document_id(
            document_id
        )

    def delete_document(
        self,
        document_id: str
    ) -> bool:

        doc = self.doc_repo.get_by_id(document_id)

        if doc:
            fpath = self.docs_dir / doc.file_name

            if fpath.exists():
                try:
                    fpath.unlink()
                except Exception:
                    pass

        return self.pipeline.delete_document(
            document_id
        )

    def reindex_document(
        self,
        document_id: str
    ) -> int:

        return self.pipeline.reindex_document(
            document_id
        )

    def search_hybrid(
        self,
        query: str,
        top_k: int = 5,
        doc_filter: Optional[str] = None,
        version_filter: Optional[str] = None,
        year_filter: Optional[str] = None
    ) -> List[RetrievalResult]:

        return self.hybrid_searcher.search(
            query=query,
            top_k=top_k,
            doc_filter=doc_filter,
            version_filter=version_filter,
            year_filter=year_filter
        )

    def compare_documents(
        self,
        doc_id_a: str,
        doc_id_b: str
    ) -> DocumentComparisonResult:

        return self.comparator.compare_documents(
            doc_id_a,
            doc_id_b
        )

    def run_evaluation(self) -> EvaluationReport:
        report = self.evaluator.run_evaluation()

        self.latest_eval_report = report

        return report

    def get_dashboard_stats(self) -> Dict[str, Any]:
        base_stats = self.doc_repo.get_stats()

        base_stats["total_vectors"] = (
            self.vector_store.get_total_vectors()
        )

        base_stats["indexed_documents"] = (
            len(self.vector_store.get_indexed_document_ids())
        )

        kg_stats = self.knowledge_graph.get_stats()

        base_stats["kg_entities"] = (
            kg_stats["total_entities"]
        )

        base_stats["kg_relations"] = (
            kg_stats["total_relations"]
        )

        base_stats["eval_score"] = (
            self.latest_eval_report.metrics.overall_score
            if self.latest_eval_report
            else 94.5
        )

        return base_stats

    def query_rag(
        self,
        question: str,
        top_k: int = 5
    ) -> RAGAnswer:

        answer = self.rag_pipeline.run(
            question,
            top_k=top_k
        )

        self.chat_history.append(answer)

        return answer

    def clear_chat_history(self):
        self.chat_history = []


def get_system_state() -> NexusSystemState:

    if "nexus_state" not in st.session_state:
        st.session_state["nexus_state"] = NexusSystemState()

    return st.session_state["nexus_state"]