import unittest
import tempfile
import gc
import numpy as np
from pathlib import Path

from nexusrag.backend.storage.database import DatabaseManager
from nexusrag.backend.storage.repository import DocumentRepository, ChunkRepository
from nexusrag.backend.retrieval.embedding_service import EmbeddingService, LocalDenseEmbeddingProvider
from nexusrag.backend.retrieval.vector_store import LocalVectorStore
from nexusrag.backend.retrieval.retriever import VectorRetriever
from nexusrag.backend.retrieval.rag_service import RAGService
from nexusrag.backend.ingestion.pipeline import IngestionPipeline
from nexusrag.data.sample_data import generate_sample_documents

class TestPhase2RAG(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.temp_dir = tempfile.TemporaryDirectory()
        cls.docs_dir = Path(cls.temp_dir.name) / "documents"
        cls.docs_dir.mkdir(parents=True, exist_ok=True)
        cls.vec_dir = Path(cls.temp_dir.name) / "vector_store"
        cls.vec_dir.mkdir(parents=True, exist_ok=True)
        cls.db_path = Path(cls.temp_dir.name) / "test_nexus_phase2.db"

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
        cls.rag_service = RAGService(cls.retriever, llm_provider="offline")

    @classmethod
    def tearDownClass(cls):
        cls.db_manager.close()
        gc.collect()
        try:
            cls.temp_dir.cleanup()
        except Exception:
            pass

    def test_01_embedding_service(self):
        texts = [
            "Attendance policy requires 75% on-site presence in 2025.",
            "Remote work is authorized for up to 3 days per week in 2026."
        ]
        embeddings = self.embedding_provider.embed_texts(texts)
        self.assertEqual(len(embeddings), 2)
        norm = np.linalg.norm(embeddings[0])
        self.assertAlmostEqual(norm, 1.0, places=3)

        query_emb = self.embedding_provider.embed_query("What is the attendance rule?")
        self.assertEqual(query_emb.shape, (1, embeddings.shape[1]))
        print(f"\n[PASS] EmbeddingService created normalized dense embeddings of dimension {embeddings.shape[1]}.")

    def test_02_vector_insertion_and_indexing(self):
        pdf_path = self.docs_dir / "Employee_Operations_Policy_2026.pdf"
        res = self.pipeline.process_file(pdf_path)
        self.assertTrue(res.success)
        self.assertGreater(res.vectors_indexed, 0)
        self.assertEqual(self.vector_store.get_total_vectors(), res.vectors_indexed)
        print(f"[PASS] LocalVectorStore indexed {self.vector_store.get_total_vectors()} vectors for '{pdf_path.name}'.")

    def test_03_vector_store_persistence(self):
        total_before = self.vector_store.get_total_vectors()
        reloaded_store = LocalVectorStore(persist_dir=self.vec_dir)
        self.assertEqual(reloaded_store.get_total_vectors(), total_before)
        print(f"[PASS] LocalVectorStore successfully reloaded {reloaded_store.get_total_vectors()} vectors from disk.")

    def test_04_similarity_retrieval(self):
        query = "What is the mandatory on-site attendance requirement?"
        results = self.retriever.retrieve(query, top_k=3)
        self.assertGreater(len(results), 0)
        top_result = results[0]
        self.assertIn("attendance", top_result.text.lower() + (top_result.section_title or "").lower())
        self.assertGreater(top_result.similarity_score, 0.0)
        print(f"[PASS] VectorRetriever retrieved top chunk '{top_result.chunk_id}' with similarity score {top_result.similarity_score:.4f}.")

    def test_05_metadata_preservation(self):
        query = "Remote work authorization rules"
        results = self.retriever.retrieve(query, top_k=3)
        self.assertGreater(len(results), 0)
        r = results[0]
        self.assertEqual(r.document_name, "Employee_Operations_Policy_2026.pdf")
        self.assertTrue(r.version in ["2", "2.0"])
        self.assertGreater(r.page_number, 0)
        self.assertTrue(len(r.section_title) > 0)
        print(f"[PASS] Retrieved result preserved metadata: Doc={r.document_name}, Version={r.version}, Page={r.page_number}, Section='{r.section_title}'.")

    def test_06_empty_retrieval_handling(self):
        empty_res = self.retriever.retrieve("", top_k=3)
        self.assertEqual(len(empty_res), 0)
        print("[PASS] VectorRetriever handled empty query gracefully without error.")

    def test_07_rag_context_and_citations(self):
        query = "How many days of remote work are authorized in 2026?"
        rag_resp = self.rag_service.answer_question(query, top_k=3)
        self.assertFalse(rag_resp.is_abstention)
        self.assertGreater(len(rag_resp.citations), 0)
        self.assertIn("Employee_Operations_Policy_2026.pdf", rag_resp.citations[0])
        print(f"[PASS] RAGService generated answer with citations: {rag_resp.citations}")

    def test_08_grounded_answer_content(self):
        query = "What is the working hours and attendance percentage for 2026 policy?"
        rag_resp = self.rag_service.answer_question(query, top_k=3)
        self.assertIn("60%", rag_resp.answer)
        print(f"[PASS] Grounded answer generated accurately: {rag_resp.answer[:120]}...")

    def test_09_abstention_on_unrelated_query(self):
        query = "What is the speed of light in deep space vacuum?"
        rag_resp = self.rag_service.answer_question(query, top_k=3)
        self.assertTrue("not contain sufficient" in rag_resp.answer.lower() or rag_resp.is_abstention or "insufficient" in rag_resp.answer.lower())
        print(f"[PASS] RAGService correctly abstained on unrelated query without hallucinating.")

    def test_10_delete_synchronization(self):
        doc_path = self.docs_dir / "IT_Infrastructure_Manual_v3.docx"
        res = self.pipeline.process_file(doc_path)
        doc_id = res.document.document_id

        vectors_before = self.vector_store.get_total_vectors()
        self.assertIn(doc_id, self.vector_store.document_ids)

        deleted = self.pipeline.delete_document(doc_id)
        self.assertTrue(deleted)

        vectors_after = self.vector_store.get_total_vectors()
        self.assertNotIn(doc_id, self.vector_store.document_ids)
        self.assertLess(vectors_after, vectors_before)
        print(f"[PASS] Deleting document '{doc_id}' successfully removed its vectors ({vectors_before} -> {vectors_after}).")

if __name__ == "__main__":
    unittest.main()
