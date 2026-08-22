import unittest
from pathlib import Path
from nexusrag.backend.ingestion.parser import DocumentParser
from nexusrag.backend.comparison.version_diff import DocumentVersionComparator
from nexusrag.backend.comparison.cross_doc import CrossDocumentComparator

class TestComparison(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        docs_dir = Path("nexusrag/data/documents")
        p2025_path = docs_dir / "Employee_Operations_Policy_2025.pdf"
        p2026_path = docs_dir / "Employee_Operations_Policy_2026.pdf"
        sr402_path = docs_dir / "Enterprise_Security_Regulation_SR402.pdf"

        cls.doc2025 = DocumentParser.parse_pdf(str(p2025_path), {"version": "1.0", "year": "2025"})
        cls.doc2026 = DocumentParser.parse_pdf(str(p2026_path), {"version": "2.0", "year": "2026"})
        cls.doc_sr402 = DocumentParser.parse_pdf(str(sr402_path), {"version": "3.1", "year": "2026"})

    def test_version_diff(self):
        report = DocumentVersionComparator.compare_documents(self.doc2025, self.doc2026)
        self.assertGreater(report.total_sections_compared, 0)
        self.assertGreater(report.modified_count, 0)
        self.assertIn("60%", report.executive_summary)
        print(f"\n[PASS] Version diff detected {report.modified_count} modified sections between 2025 and 2026.")

    def test_cross_doc_comparison(self):
        report = CrossDocumentComparator.compare_documents([self.doc2025, self.doc2026, self.doc_sr402])
        self.assertGreater(len(report.matrix), 0)
        self.assertGreater(len(report.overall_conflicts), 0)
        print(f"[PASS] Cross-doc comparison generated matrix with {len(report.matrix)} requirement topics and {len(report.overall_conflicts)} conflicts.")

if __name__ == "__main__":
    unittest.main()
