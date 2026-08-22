from dataclasses import dataclass, asdict
from typing import List, Dict, Any

@dataclass
class EvalMetricResult:
    precision_at_k: float
    recall_at_k: float
    faithfulness_score: float
    relevance_score: float
    citation_accuracy: float
    overall_score: float

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

class MetricsCalculator:
    @staticmethod
    def calculate_metrics(test_results: List[Dict[str, Any]]) -> EvalMetricResult:
        if not test_results:
            return EvalMetricResult(0.0, 0.0, 0.0, 0.0, 0.0, 0.0)

        prec_list = [r.get("precision", 0.0) for r in test_results]
        rec_list = [r.get("recall", 0.0) for r in test_results]
        faith_list = [r.get("faithfulness", 0.0) for r in test_results]
        rel_list = [r.get("relevance", 0.0) for r in test_results]
        cite_list = [r.get("citation_accuracy", 0.0) for r in test_results]

        avg_prec = sum(prec_list) / len(prec_list)
        avg_rec = sum(rec_list) / len(rec_list)
        avg_faith = sum(faith_list) / len(faith_list)
        avg_rel = sum(rel_list) / len(rel_list)
        avg_cite = sum(cite_list) / len(cite_list)

        overall = (avg_prec * 0.2 + avg_rec * 0.2 + avg_faith * 0.25 + avg_rel * 0.2 + avg_cite * 0.15) * 100

        return EvalMetricResult(
            precision_at_k=round(avg_prec, 4),
            recall_at_k=round(avg_rec, 4),
            faithfulness_score=round(avg_faith, 4),
            relevance_score=round(avg_rel, 4),
            citation_accuracy=round(avg_cite, 4),
            overall_score=round(overall, 2)
        )
