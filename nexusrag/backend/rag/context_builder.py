from typing import List, Dict, Any
from nexusrag.backend.retrieval.retriever import RetrievalResult

class ContextBuilder:
    @staticmethod
    def build_context(evidence_chunks: List[RetrievalResult]) -> str:
        if not evidence_chunks:
            return "NO RETRIEVED EVIDENCE AVAILABLE."

        blocks = []
        for i, ev in enumerate(evidence_chunks):
            source_num = i + 1
            loc = f"Page: {ev.page_number}" if ev.page_number else (f"Sheet: {ev.sheet_name}" if ev.sheet_name else "Page: 1")
            sec = f"Section: {ev.section_title}" if ev.section_title else "Section: General"
            ver = f"Version: {ev.version}" if ev.version else "Version: 1.0"
            yr = f"Year: {ev.year}" if ev.year else "Year: 2026"
            dept = f"Department: {ev.department}" if ev.department else "Department: General"
            score = f"Retrieval Score: {ev.similarity_score:.4f}"

            header = f"=== SOURCE [{source_num}] ===\n" \
                     f"Document: {ev.document_name}\n" \
                     f"{loc}\n" \
                     f"{sec}\n" \
                     f"{ver} | {yr} | {dept}\n" \
                     f"{score}\n" \
                     f"Content:\n{ev.text.strip()}"
            blocks.append(header)

        return "\n\n" + "\n\n----------------------------------------\n\n".join(blocks)
