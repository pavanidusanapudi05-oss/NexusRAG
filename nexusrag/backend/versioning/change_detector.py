import difflib
from dataclasses import dataclass, field, asdict
from typing import List, Dict, Any, Optional

@dataclass
class SectionChange:
    topic: str
    status: str
    doc_a_text: Optional[str] = None
    doc_b_text: Optional[str] = None
    doc_a_source: Optional[str] = None
    doc_b_source: Optional[str] = None
    summary_of_change: Optional[str] = None
    similarity_ratio: float = 1.0

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

class ChangeDetector:
    @staticmethod
    def compare_sections(
        sections_a: Dict[str, Dict[str, Any]],
        sections_b: Dict[str, Dict[str, Any]]
    ) -> List[SectionChange]:
        changes: List[SectionChange] = []
        all_topics = set(sections_a.keys()).union(set(sections_b.keys()))

        for topic in sorted(list(all_topics)):
            in_a = topic in sections_a
            in_b = topic in sections_b

            if in_a and not in_b:
                data_a = sections_a[topic]
                changes.append(SectionChange(
                    topic=topic,
                    status="Removed",
                    doc_a_text=data_a.get("text", ""),
                    doc_a_source=data_a.get("source", ""),
                    summary_of_change=f"Requirement '{topic}' present in Document A was removed in Document B.",
                    similarity_ratio=0.0
                ))
            elif in_b and not in_a:
                data_b = sections_b[topic]
                changes.append(SectionChange(
                    topic=topic,
                    status="Added",
                    doc_b_text=data_b.get("text", ""),
                    doc_b_source=data_b.get("source", ""),
                    summary_of_change=f"New requirement '{topic}' introduced in Document B.",
                    similarity_ratio=0.0
                ))
            else:
                data_a = sections_a[topic]
                data_b = sections_b[topic]
                text_a = data_a.get("text", "")
                text_b = data_b.get("text", "")

                matcher = difflib.SequenceMatcher(None, text_a.lower(), text_b.lower())
                sim = matcher.ratio()

                if sim >= 0.95:
                    status = "Unchanged"
                    summary = f"No substantive policy change in '{topic}'."
                else:
                    status = "Modified"
                    summary = f"Policy clause '{topic}' modified between versions ({int((1-sim)*100)}% content delta)."

                changes.append(SectionChange(
                    topic=topic,
                    status=status,
                    doc_a_text=text_a,
                    doc_b_text=text_b,
                    doc_a_source=data_a.get("source", ""),
                    doc_b_source=data_b.get("source", ""),
                    summary_of_change=summary,
                    similarity_ratio=round(sim, 4)
                ))

        return changes
