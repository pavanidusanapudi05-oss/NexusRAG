import unittest
import tempfile
import os
import gc
from pathlib import Path

from nexusrag.backend.storage.database import DatabaseManager
from nexusrag.backend.storage.repository import DocumentRepository, ChunkRepository
from nexusrag.backend.models.document import ProcessingStatus
from nexusrag.backend.ingestion.pdf_processor import PDFProcessor
from nexusrag.backend.ingestion.docx_processor import DOCXProcessor
from nexusrag.backend.ingestion.text_processor import TextProcessor
from nexusrag.backend.ingestion.spreadsheet_processor import SpreadsheetProcessor
from nexusrag.backend.ingestion.metadata import MetadataExtractor
from nexusrag.backend.ingestion.chunker import DocumentChunker
from nexusrag.backend.ingestion.pipeline import IngestionPipeline
from nexusrag.data.sample_data import generate_sample_documents

class TestPhase1Ingestion(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.temp_dir = tempfile.TemporaryDirectory()
        cls.docs_dir = Path(cls.temp_dir.name) / "documents"
        cls.docs_dir.mkdir(parents=True, exist_ok=True)
        cls.db_path = Path(cls.temp_dir.name) / "test_nexus.db"
        
        cls.sample_files = generate_sample_documents(str(cls.docs_dir))
        cls.db_manager = DatabaseManager(db_path=cls.db_path)
        cls.doc_repo = DocumentRepository(cls.db_manager)
        cls.chunk_repo = ChunkRepository(cls.db_manager)
        cls.pipeline = IngestionPipeline(db_manager=cls.db_manager, chunk_size=500, chunk_overlap=100)

    @classmethod
    def tearDownClass(cls):
        cls.db_manager.close()
        gc.collect()
        try:
            cls.temp_dir.cleanup()
        except Exception:
            pass

    def test_01_pdf_processing(self):
        pdf_path = self.docs_dir / "Employee_Operations_Policy_2025.pdf"
        self.assertTrue(pdf_path.exists())
        
        blocks = PDFProcessor.process(pdf_path)
        self.assertGreater(len(blocks), 0)
        self.assertEqual(blocks[0].page_number, 1)
        self.assertIn("NexusCorp Employee Operations Policy 2025", blocks[0].text)
        print(f"\n[PASS] PDFProcessor extracted {len(blocks)} blocks from PDF with page metadata.")

    def test_02_docx_processing(self):
        docx_path = self.docs_dir / "IT_Infrastructure_Manual_v3.docx"
        self.assertTrue(docx_path.exists())
        
        blocks = DOCXProcessor.process(docx_path)
        self.assertGreater(len(blocks), 0)
        self.assertTrue(any("Process X" in b.text for b in blocks))
        print(f"[PASS] DOCXProcessor extracted {len(blocks)} blocks including headings and paragraphs.")

    def test_03_spreadsheet_processing(self):
        xlsx_path = self.docs_dir / "Compliance_Audit_Guidelines_2026.xlsx"
        self.assertTrue(xlsx_path.exists())
        
        blocks = SpreadsheetProcessor.process(xlsx_path)
        self.assertGreater(len(blocks), 0)
        self.assertEqual(blocks[0].sheet_name, "Audit_Checklist")
        self.assertIn("AUD-101", blocks[0].text)
        print(f"[PASS] SpreadsheetProcessor extracted sheet '{blocks[0].sheet_name}' with structured rows.")

    def test_04_text_processing(self):
        txt_path = Path(self.temp_dir.name) / "sample_manual.txt"
        with open(txt_path, "w", encoding="utf-8") as f:
            f.write("# Server Provisioning Guide\n\nSection 1. Architecture\nAll nodes run Linux OS.\n\nSection 2. Backups\nNightly backups at 02:00 UTC.")

        blocks = TextProcessor.process(txt_path)
        self.assertGreater(len(blocks), 0)
        self.assertIn("Linux OS", blocks[0].text)
        print(f"[PASS] TextProcessor extracted {len(blocks)} sections from plain text file.")

    def test_05_metadata_extraction(self):
        meta = MetadataExtractor.extract_metadata(
            file_name="Enterprise_Security_Regulation_SR402.pdf",
            file_type="pdf",
            file_size=1024,
            file_hash="dummyhash",
            text_sample="Regulation SR-402 effective year 2026 governing cybersecurity access."
        )
        self.assertEqual(meta["year"], "2026")
        self.assertEqual(meta["department"], "Information Security")
        self.assertEqual(meta["category"], "Regulation")
        print(f"[PASS] MetadataExtractor successfully extracted metadata: {meta}")

    def test_06_chunking_and_provenance(self):
        pdf_path = self.docs_dir / "Employee_Operations_Policy_2026.pdf"
        blocks = PDFProcessor.process(pdf_path)
        chunker = DocumentChunker(chunk_size=400, chunk_overlap=80)
        chunks = chunker.chunk_blocks("doc_test_123", "Employee_Operations_Policy_2026.pdf", "pdf", blocks, {"version": "2.0", "year": "2026"})

        self.assertGreater(len(chunks), 0)
        for c in chunks:
            self.assertTrue(c.chunk_id.startswith("doc_test_123_p"))
            self.assertIn("document_name", c.metadata)
            self.assertIn("page_number", c.metadata)
            self.assertIn("section_title", c.metadata)
            self.assertGreater(c.char_count, 0)
        print(f"[PASS] DocumentChunker created {len(chunks)} traceable chunks with metadata.")

    def test_07_pipeline_ingestion_and_sqlite_registry(self):
        pdf_path = self.docs_dir / "Employee_Operations_Policy_2025.pdf"
        res = self.pipeline.process_file(pdf_path)
        self.assertTrue(res.success)
        self.assertFalse(res.is_duplicate)
        self.assertGreater(res.chunks_created, 0)

        # Check SQLite record
        doc_record = self.doc_repo.get_by_id(res.document.document_id)
        self.assertIsNotNone(doc_record)
        self.assertEqual(doc_record.processing_status, ProcessingStatus.PROCESSED)
        self.assertEqual(doc_record.chunk_count, res.chunks_created)

        # Check chunks in SQLite
        db_chunks = self.chunk_repo.get_by_document_id(res.document.document_id)
        self.assertEqual(len(db_chunks), res.chunks_created)
        print(f"[PASS] IngestionPipeline ingested document into SQLite with {len(db_chunks)} chunks.")

    def test_08_duplicate_detection(self):
        pdf_path = self.docs_dir / "Employee_Operations_Policy_2025.pdf"
        res_dup = self.pipeline.process_file(pdf_path)
        self.assertTrue(res_dup.success)
        self.assertTrue(res_dup.is_duplicate)
        self.assertIn("already been indexed", res_dup.message)
        print(f"[PASS] Duplicate upload correctly detected via SHA-256 hash.")

    def test_09_empty_file_handling(self):
        empty_path = Path(self.temp_dir.name) / "empty_doc.txt"
        with open(empty_path, "w", encoding="utf-8") as f:
            pass

        res = self.pipeline.process_file(empty_path)
        self.assertFalse(res.success)
        self.assertIn("empty", res.message.lower())
        print(f"[PASS] Empty document rejected gracefully with error message: '{res.message}'")

    def test_10_unsupported_file_handling(self):
        bad_path = Path(self.temp_dir.name) / "malicious_binary.exe"
        with open(bad_path, "wb") as f:
            f.write(b"MZ\x90\x00")

        res = self.pipeline.process_file(bad_path)
        self.assertFalse(res.success)
        self.assertIn("unsupported", res.message.lower())
        print(f"[PASS] Unsupported file format rejected gracefully: '{res.message}'")

    def test_11_delete_document_and_cascade_chunks(self):
        docx_path = self.docs_dir / "IT_Infrastructure_Manual_v3.docx"
        res = self.pipeline.process_file(docx_path)
        doc_id = res.document.document_id

        chunks_before = self.chunk_repo.get_by_document_id(doc_id)
        self.assertGreater(len(chunks_before), 0)

        deleted = self.pipeline.delete_document(doc_id)
        self.assertTrue(deleted)

        self.assertIsNone(self.doc_repo.get_by_id(doc_id))
        chunks_after = self.chunk_repo.get_by_document_id(doc_id)
        self.assertEqual(len(chunks_after), 0)
        print(f"[PASS] Deleting document '{doc_id}' successfully removed document record and all associated chunks.")

if __name__ == "__main__":
    unittest.main()
