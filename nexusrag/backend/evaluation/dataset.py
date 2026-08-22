from dataclasses import dataclass
from typing import List, Dict, Any

@dataclass
class EvalTestCase:
    test_id: str
    query: str
    category: str
    expected_doc: str
    expected_keywords: List[str]
    expected_version: str
    ground_truth_fact: str

BENCHMARK_DATASET: List[EvalTestCase] = [
    EvalTestCase(
        test_id="TC-001",
        query="What is the mandatory on-site attendance requirement in the 2026 policy?",
        category="Policy Retrieval",
        expected_doc="Employee_Operations_Policy_2026.pdf",
        expected_keywords=["60%", "attendance"],
        expected_version="2.0",
        ground_truth_fact="Minimum of 60% on-site presence (24 hours per week)."
    ),
    EvalTestCase(
        test_id="TC-002",
        query="How many days per week can employees work remotely according to the 2026 policy?",
        category="Allowance Retrieval",
        expected_doc="Employee_Operations_Policy_2026.pdf",
        expected_keywords=["3 days", "remote"],
        expected_version="2.0",
        ground_truth_fact="Employees are authorized to work remotely up to 3 days per week."
    ),
    EvalTestCase(
        test_id="TC-003",
        query="What are the encryption standards and MFA requirements in Regulation SR-402?",
        category="Security Regulation",
        expected_doc="Enterprise_Security_Regulation_SR402.pdf",
        expected_keywords=["mfa", "aes-256", "encryption"],
        expected_version="1.0",
        ground_truth_fact="Multi-Factor Authentication is required for all access; sensitive data must use AES-256 encryption."
    ),
    EvalTestCase(
        test_id="TC-004",
        query="What was the mandatory on-site presence percentage in the 2025 policy?",
        category="Version Comparison",
        expected_doc="Employee_Operations_Policy_2025.pdf",
        expected_keywords=["75%", "attendance"],
        expected_version="1.0",
        ground_truth_fact="75% on-site presence required in the 2025 policy."
    ),
    EvalTestCase(
        test_id="TC-005",
        query="What are the audit checklist criteria for attendance and hybrid schedule review in 2026?",
        category="Spreadsheet Ingestion",
        expected_doc="Compliance_Audit_Guidelines_2026.xlsx",
        expected_keywords=["aud-101", "60%"],
        expected_version="1.0",
        ground_truth_fact="AUD-101 attendance and hybrid schedule review (60% on-site)."
    ),
    EvalTestCase(
        test_id="TC-006",
        query="What is the warp drive speed limit in quantum gravitational space?",
        category="Out-of-Domain Abstention",
        expected_doc="None",
        expected_keywords=["insufficient", "couldn't find"],
        expected_version="None",
        ground_truth_fact="System should abstain safely with insufficient evidence message."
    )
]
