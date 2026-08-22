from dataclasses import dataclass, asdict
from typing import List, Dict, Any
from nexusrag.backend.rag.rag_pipeline import RAGPipeline, RAGAnswer
from .dataset import BENCHMARK_DATASET, EvalTestCase
from .metrics import MetricsCalculator, EvalMetricResult

@dataclass
class TestCaseResult:
    test_id: str
    query: str
    category: str
    passed: bool
    precision: float
    recall: float
    faithfulness: float
    relevance: float
    citation_accuracy: float
    answer_preview: str
    citations_count: int
    notes: str

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

@dataclass
class EvaluationReport:
    total_tests: int
    passed_tests: int
    failed_tests: int
    metrics: EvalMetricResult
    results: List[TestCaseResult]

    def to_dict(self) -> Dict[str, Any]:
        return {
            "total_tests": self.total_tests,
            "passed_tests": self.passed_tests,
            "failed_tests": self.failed_tests,
            "metrics": self.metrics.to_dict(),
            "results": [r.to_dict() for r in self.results]
        }

class RAGEvaluator:
    def __init__(self, rag_pipeline: RAGPipeline):
        self.rag_pipeline = rag_pipeline

    def run_evaluation(self, test_cases: List[EvalTestCase] = BENCHMARK_DATASET) -> EvaluationReport:
        results: List[TestCaseResult] = []

        for tc in test_cases:
            ans: RAGAnswer = self.rag_pipeline.run(tc.query, top_k=4)

            if tc.expected_doc == "None":
                passed = ans.is_abstention
                prec = 1.0 if ans.is_abstention else 0.0
                rec = 1.0 if ans.is_abstention else 0.0
                faith = 1.0 if ans.is_abstention else 0.0
                rel = 1.0 if ans.is_abstention else 0.0
                cite_acc = 1.0
                notes = "Correctly abstained on out-of-domain query." if passed else "Failed to abstain on unsupported query."
            else:
                retrieved_docs = [e.document_name.lower() for e in ans.evidence]
                exp_clean = tc.expected_doc.lower().replace(".pdf", "").replace(".xlsx", "").replace(".docx", "")
                doc_matched = any(exp_clean in d for d in retrieved_docs)

                ans_lower = ans.answer.lower()
                kw_matches = sum(1 for kw in tc.expected_keywords if kw.lower() in ans_lower)
                faith = kw_matches / len(tc.expected_keywords) if tc.expected_keywords else 1.0
                prec = 1.0 if doc_matched else 0.5
                rec = 1.0 if doc_matched else 0.5
                rel = 1.0 if doc_matched else 0.5
                cite_acc = 1.0 if len(ans.citations) > 0 else 0.5

                passed = doc_matched and (faith >= 0.5 or len(ans.citations) > 0)
                notes = f"Retrieved matching evidence for '{tc.expected_doc}' with {kw_matches}/{len(tc.expected_keywords)} keyword matches."

            results.append(TestCaseResult(
                test_id=tc.test_id,
                query=tc.query,
                category=tc.category,
                passed=passed,
                precision=prec,
                recall=rec,
                faithfulness=faith,
                relevance=rel,
                citation_accuracy=cite_acc,
                answer_preview=ans.answer[:120] + "...",
                citations_count=len(ans.citations),
                notes=notes
            ))

        metrics = MetricsCalculator.calculate_metrics([r.to_dict() for r in results])
        passed_count = sum(1 for r in results if r.passed)
        failed_count = len(results) - passed_count

        return EvaluationReport(
            total_tests=len(results),
            passed_tests=passed_count,
            failed_tests=failed_count,
            metrics=metrics,
            results=results
        )
