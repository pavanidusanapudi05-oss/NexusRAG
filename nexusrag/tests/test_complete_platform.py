import unittest
import tempfile
import gc
from pathlib import Path

from nexusrag.backend.storage.database import DatabaseManager
from nexusrag.backend.storage.repository import DocumentRepository, ChunkRepository
from nexusrag.backend.retrieval.embedding_service import EmbeddingService
from nexusrag.backend.retrieval.vector_store import LocalVectorStore
from nexusrag.backend.retrieval.hybrid_search import HybridSearch
from nexusrag.backend.retrieval.reranker import PrecisionReranker
from nexusrag.backend.llm.provider import OfflineDeterministicLLMProvider, LLMProviderFactory
from nexusrag.backend.rag.rag_pipeline import RAGPipeline
from nexusrag.backend.graph.graph_store import LocalKnowledgeGraph
from nexusrag.backend.versioning.document_compare import DocumentComparator
from nexusrag.backend.evaluation.evaluator import RAGEvaluator
from nexusrag.backend.ingestion.pipeline import IngestionPipeline
from nexusrag.data.sample_data import generate_sample_documents

class TestCompletePlatform(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.temp_dir = tempfile.TemporaryDirectory()
        cls.docs_dir = Path(cls.temp_dir.name) / "documents"
        cls.docs_dir.mkdir(parents=True, exist_ok=True)
        cls.vec_dir = Path(cls.temp_dir.name) / "vector_store"
        cls.vec_dir.mkdir(parents=True, exist_ok=True)
        cls.graph_path = Path(cls.temp_dir.name) / "graph" / "kg.json"
        cls.db_path = Path(cls.temp_dir.name) / "test_complete_nexus.db"

        cls.sample_files = generate_sample_documents(str(cls.docs_dir))

        cls.db_manager = DatabaseManager(db_path=cls.db_path)
        cls.doc_repo = DocumentRepository(cls.db_manager)
        cls.chunk_repo = ChunkRepository(cls.db_manager)
        cls.vector_store = LocalVectorStore(persist_dir=cls.vec_dir)
        cls.embedding_provider = EmbeddingService.create_provider("local_dense")
        cls.knowledge_graph = LocalKnowledgeGraph(persist_path=cls.graph_path)

        cls.pipeline = IngestionPipeline(
            db_manager=cls.db_manager,
            vector_store=cls.vector_store,
            embedding_provider=cls.embedding_provider,
            knowledge_graph=cls.knowledge_graph,
            chunk_size=500,
            chunk_overlap=100
        )

        for s_file in cls.sample_files:
            cls.pipeline.process_file(Path(s_file))

        cls.hybrid_searcher = HybridSearch(
            vector_store=cls.vector_store,
            embedding_provider=cls.embedding_provider,
            semantic_weight=0.7,
            keyword_weight=0.3,
            reranker_enabled=True
        )

        cls.llm_provider = LLMProviderFactory.create("offline")
        cls.rag_pipeline = RAGPipeline(
            retriever=cls.hybrid_searcher,
            llm_provider=cls.llm_provider
        )

        cls.comparator = DocumentComparator(cls.doc_repo, cls.chunk_repo)
        cls.evaluator = RAGEvaluator(cls.rag_pipeline)

    @classmethod
    def tearDownClass(cls):
        cls.db_manager.close()
        gc.collect()
        try:
            cls.temp_dir.cleanup()
        except Exception:
            pass

    def test_01_hybrid_search_fusion(self):
        results = self.hybrid_searcher.search("SR-402 MFA encryption standards", top_k=3)
        self.assertGreater(len(results), 0)
        self.assertIn("Enterprise_Security_Regulation_SR402.pdf", [r.document_name for r in results])
        self.assertGreater(results[0].similarity_score, 0.0)
        print(f"\n[PASS] HybridSearch retrieved {len(results)} chunks with top score: {results[0].similarity_score:.4f}")

    def test_02_precision_reranker_boosting(self):
        reranker = PrecisionReranker(enabled=True)
        raw_results = self.hybrid_searcher.search("2026 remote work allowance 3 days", top_k=4)
        reranked = reranker.rerank("2026 remote work allowance 3 days", raw_results, top_k=2)
        self.assertEqual(len(reranked), 2)
        self.assertTrue("Employee_Operations_Policy_2026.pdf" in reranked[0].document_name or "Remote Work" in (reranked[0].section_title or ""))
        print("[PASS] PrecisionReranker prioritized high-relevance chunks.")

    def test_03_version_comparator(self):
        doc_2025 = self.doc_repo.get_by_name("Employee_Operations_Policy_2025.pdf")
        doc_2026 = self.doc_repo.get_by_name("Employee_Operations_Policy_2026.pdf")
        self.assertIsNotNone(doc_2025)
        self.assertIsNotNone(doc_2026)

        comp = self.comparator.compare_documents(doc_2025.document_id, doc_2026.document_id)
        self.assertGreater(comp.total_sections, 0)
        self.assertGreater(comp.modified_count, 0)
        print(f"[PASS] DocumentComparator detected {comp.modified_count} modified sections between 2025 and 2026.")

    def test_04_knowledge_graph_entities_and_relations(self):
        stats = self.knowledge_graph.get_stats()
        self.assertGreater(stats["total_entities"], 0)
        self.assertGreater(stats["total_relations"], 0)
        entities = self.knowledge_graph.list_entities(entity_type="Department")
        self.assertGreater(len(entities), 0)
        rels = self.knowledge_graph.list_relations()
        self.assertGreater(len(rels), 0)
        print(f"[PASS] KnowledgeGraph populated with {stats['total_entities']} entities and {stats['total_relations']} relations.")

    def test_05_evaluation_benchmark_execution(self):
        report = self.evaluator.run_evaluation()
        self.assertGreater(report.total_tests, 0)
        self.assertGreater(report.passed_tests, 0)
        self.assertGreater(report.metrics.overall_score, 70.0)
        self.assertGreaterEqual(report.metrics.precision_at_k, 0.5)
        print(f"[PASS] RAGEvaluator ran {report.total_tests} test cases with score: {report.metrics.overall_score:.1f}% ({report.passed_tests}/{report.total_tests} passed).")

    def test_06_rag_pipeline_end_to_end(self):
        ans = self.rag_pipeline.run("What are the MFA requirements in Regulation SR-402?", top_k=3)
        self.assertFalse(ans.is_abstention)
        self.assertIn("MFA", ans.answer.upper())
        self.assertGreater(len(ans.citations), 0)
        self.assertIn(ans.confidence.level, ["High", "Medium"])
        print(f"[PASS] RAGPipeline generated grounded answer: {ans.answer[:100]}... [Confidence: {ans.confidence.level}]")

    def test_07_knowledge_graph_cascade_deletion(self):
        doc_2025 = self.doc_repo.get_by_name("Employee_Operations_Policy_2025.pdf")
        initial_ent_count = len(self.knowledge_graph.entities)
        
        self.pipeline.delete_document(doc_2025.document_id)
        post_del_ent_count = len(self.knowledge_graph.entities)
        self.assertLess(post_del_ent_count, initial_ent_count)
        print(f"[PASS] Cascade delete purged graph entities for deleted document ({initial_ent_count} -> {post_del_ent_count}).")

if __name__ == "__main__":
    unittest.main()
