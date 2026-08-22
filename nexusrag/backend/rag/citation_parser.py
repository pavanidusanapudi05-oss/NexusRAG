import re
from dataclasses import dataclass, asdict
from typing import List, Dict, Any, Optional
from nexusrag.backend.retrieval.retriever import RetrievalResult

@dataclass
class Citation:
    source_id: int
    chunk_id: str
    document_id: str
    document_name: str
    page_number: Optional[int]
    section_title: Optional[str]
    sheet_name: Optional[str]
    version: str
    year: str
    department: str
    similarity_score: float
    citation_label: str

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

class CitationParser:
    @staticmethod
    def extract_citations(answer_text: str, evidence_chunks: List[RetrievalResult]) -> List[Citation]:
        if not evidence_chunks:
            return []

        # Find all citation tokens like [1], [2], [1, 2] in the answer
        found_indices = set()
        matches = re.findall(r'\[([0-9,\s]+)\]', answer_text)
        for m in matches:
            for num_str in re.split(r'[,\s]+', m):
                if num_str.isdigit():
                    idx = int(num_str)
                    if 1 <= idx <= len(evidence_chunks):
                        found_indices.add(idx)

        # If answer mentions no bracketed citations or mentions all, include all retrieved sources that were provided
        if not found_indices:
            found_indices = set(range(1, len(evidence_chunks) + 1))

        citations = []
        for src_idx in sorted(list(found_indices)):
            ev = evidence_chunks[src_idx - 1]
            loc_parts = []
            if ev.page_number:
                loc_parts.append(f"Page {ev.page_number}")
            elif ev.sheet_name:
                loc_parts.append(f"Sheet: {ev.sheet_name}")
            if ev.section_title:
                loc_parts.append(f"Section: {ev.section_title}")
            
            loc_str = " — ".join(loc_parts) if loc_parts else "General"
            label = f"[{src_idx}] {ev.document_name} ({loc_str}) [v{ev.version} | {ev.year}]"

            citations.append(Citation(
                source_id=src_idx,
                chunk_id=ev.chunk_id,
                document_id=ev.document_id,
                document_name=ev.document_name,
                page_number=ev.page_number,
                section_title=ev.section_title,
                sheet_name=ev.sheet_name,
                version=ev.version or "1.0",
                year=ev.year or "2026",
                department=ev.department or "General",
                similarity_score=ev.similarity_score,
                citation_label=label
            ))

        return citations
