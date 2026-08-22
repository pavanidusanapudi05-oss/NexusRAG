import unittest
import tempfile
import gc
from pathlib import Path

from nexusrag.backend.storage.database import DatabaseManager
from nexusrag.backend.storage.repository import DocumentRepository, ChunkRepository
from nexusrag.backend.retrieval.embedding_service import EmbeddingService
from nexusrag.backend.retrieval.vector_store import LocalVectorStore
from nexusrag.backend.retrieval.retriever import VectorRetriever, RetrievalResult
from nexusrag.backend.llm.provider import OfflineDeterministicLLMProvider, LLMProviderFactory
from nexusrag.backend.rag.context_builder import ContextBuilder
from nexusrag.backend.rag.prompt_builder import PromptBuilder
from nexusrag.backend.rag.citation_parser import CitationParser
from nexusrag.backend.rag.confidence import ConfidenceEstimator
from nexusrag.backend.rag.rag_pipeline import RAGPipeline
from nexusrag.backend.ingestion.pipeline import IngestionPipeline
from nexusrag.data.sample_data import generate_sample_documents

class TestPhase3RAG(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.temp_dir = tempfile.TemporaryDirectory()
        cls.docs_dir = Path(cls.temp_dir.name) / "documents"
        cls.docs_dir.mkdir(parents=True, exist_ok=True)
        cls.vec_dir = Path(cls.temp_dir.name) / "vector_store"
        cls.vec_dir.mkdir(parents=True, exist_ok=True)
        cls.db_path = Path(cls.temp_dir.name) / "test_nexus_phase3.db"

        cls.sample_files = generate_sample_documents(str(cls.docs_dir))

        cls.db_manager = DatabaseManager(db_path=cls.db_path)
        cls.doc_repo = DocumentRepository(cls.db_manager)
        cls.chunk_repo = ChunkRepository(cls.db_manager)
        cls.vector_store = LocalVectorStore(persist_dir=cls.vec_dir)
        cls.embedding_provider = EmbeddingService.create_provider("local_dense")

        cls.pipeline = IngestionPipeline(
            db_manager=cls.db_manager,
            vector_store=cls.vector_store,
            embedding_provider=cls.embedding_provider,
            chunk_size=500,
            chunk_overlap=100
        )
        cls.retriever = VectorRetriever(cls.vector_store, cls.embedding_provider, top_k=4)
        cls.llm_provider = LLMProviderFactory.create("offline")
        cls.rag_pipeline = RAGPipeline(cls.retriever, cls.llm_provider, similarity_threshold=0.15)

        # Ingest test documents
        cls.pipeline.process_file(cls.docs_dir / "Employee_Operations_Policy_2025.pdf")
        cls.pipeline.process_file(cls.docs_dir / "Employee_Operations_Policy_2026.pdf")
        cls.pipeline.process_file(cls.docs_dir / "Enterprise_Security_Regulation_SR402.pdf")

    @classmethod
    def tearDownClass(cls):
        cls.db_manager.close()
        gc.collect()
        try:
            cls.temp_dir.cleanup()
        except Exception:
            pass

    def test_01_query_validation(self):
        ans1 = self.rag_pipeline.run("")
        self.assertTrue(ans1.is_abstention)
        self.assertIn("valid question", ans1.answer)

        ans2 = self.rag_pipeline.run("    ")
        self.assertTrue(ans2.is_abstention)

        ans3 = self.rag_pipeline.run("hi")
        self.assertTrue(ans3.is_abstention)
        self.assertIn("too short", ans3.answer)
        print("\n[PASS] Query validation rejected empty, whitespace, and short queries.")

    def test_02_context_builder(self):
        chunks = self.retriever.retrieve("remote work policy", top_k=2)
        self.assertGreater(len(chunks), 0)
        context = ContextBuilder.build_context(chunks)
        self.assertIn("=== SOURCE [1] ===", context)
        self.assertIn("Document:", context)
        self.assertIn("Page:", context)
        self.assertIn("Version:", context)
        print(f"[PASS] ContextBuilder formatted {len(chunks)} sources with full metadata preservation.")

    def test_03_prompt_builder(self):
        sys_prompt = PromptBuilder.build_system_prompt()
        self.assertIn("STRICT GROUNDING RULES", sys_prompt)
        self.assertIn("bracketed citations", sys_prompt)

        chunks = self.retriever.retrieve("attendance hours", top_k=2)
        user_prompt = PromptBuilder.build_user_prompt("What are the attendance hours?", chunks)
        self.assertIn("QUESTION: What are the attendance hours?", user_prompt)
        self.assertIn("RETRIEVED DOCUMENT SOURCES:", user_prompt)
        print("[PASS] PromptBuilder assembled grounded system and user prompts.")

    def test_04_citation_parser(self):
        chunks = self.retriever.retrieve("remote work policy", top_k=2)
        sample_answer = "Remote work is authorized up to 3 days per week [1] with manager approval [2]."
        citations = CitationParser.extract_citations(sample_answer, chunks)
        self.assertGreater(len(citations), 0)
        self.assertEqual(citations[0].source_id, 1)
        self.assertTrue(len(citations[0].document_name) > 0)
        self.assertIsNotNone(citations[0].page_number)
        print(f"[PASS] CitationParser mapped citations to: {[c.citation_label for c in citations]}")

    def test_05_confidence_estimator(self):
        chunks = self.retriever.retrieve("remote work authorization", top_k=3)
        conf = ConfidenceEstimator.estimate_confidence(chunks, is_abstention=False, has_conflict=False)
        self.assertIn(conf.level, ["High", "Medium"])
        self.assertGreater(conf.score_percentage, 50)
        self.assertGreater(conf.top_similarity, 0.0)

        abst_conf = ConfidenceEstimator.estimate_confidence([], is_abstention=True)
        self.assertEqual(abst_conf.level, "Low")
        self.assertLess(abst_conf.score_percentage, 30)
        print(f"[PASS] ConfidenceEstimator generated confidence: {conf.level} ({conf.score_percentage}%) - {conf.explanation}")

    def test_06_rag_pipeline_grounded_answer(self):
        query = "What is the mandatory attendance percentage required in the 2026 policy?"
        ans = self.rag_pipeline.run(query, top_k=3)
        self.assertFalse(ans.is_abstention)
        self.assertIn("60%", ans.answer)
        self.assertGreater(len(ans.citations), 0)
        self.assertGreater(len(ans.evidence), 0)
        print(f"[PASS] RAGPipeline produced grounded answer with {len(ans.citations)} citations: {ans.answer[:100]}...")

    def test_07_insufficient_evidence_abstention(self):
        query = "What is the formula for quantum gravitational waves in deep space?"
        ans = self.rag_pipeline.run(query, top_k=3)
        self.assertTrue(ans.is_abstention)
        self.assertIn("couldn't find sufficient evidence", ans.answer.lower())
        print(f"[PASS] RAGPipeline safely abstained on out-of-domain query: '{ans.answer}'")

    def test_08_conflicting_evidence_detection(self):
        query = "Compare the changes and differences between 2025 and 2026 attendance policy rules."
        ans = self.rag_pipeline.run(query, top_k=4)
        self.assertFalse(ans.is_abstention)
        self.assertTrue(ans.has_conflict or "2025" in ans.answer)
        self.assertIn("75%", ans.answer)
        self.assertIn("60%", ans.answer)
        print(f"[PASS] RAGPipeline detected cross-version policy differences: {ans.answer[:120]}...")

    def test_09_llm_provider_offline_fallback(self):
        provider = LLMProviderFactory.create(provider_name="gemini", api_key="", model_name="")
        self.assertTrue(isinstance(provider, OfflineDeterministicLLMProvider))
        print("[PASS] LLMProviderFactory safely fell back to OfflineDeterministicLLMProvider when API key is missing.")

    def test_10_structured_response_serialization(self):
        ans = self.rag_pipeline.run("What are the MFA requirements in SR-402?", top_k=2)
        d = ans.to_dict()
        self.assertIn("answer", d)
        self.assertIn("confidence", d)
        self.assertIn("citations", d)
        self.assertIn("evidence", d)
        self.assertEqual(d["confidence"]["level"], ans.confidence.level)
        print(f"[PASS] RAGAnswer serialized cleanly to structured dictionary with keys: {list(d.keys())}")

if __name__ == "__main__":
    unittest.main()
