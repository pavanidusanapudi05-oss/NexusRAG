from dataclasses import dataclass, field, asdict
from typing import List, Dict, Any, Optional
from nexusrag.backend.storage.repository import DocumentRepository, ChunkRepository
from .change_detector import ChangeDetector, SectionChange

@dataclass
class DocumentComparisonResult:
    doc_a_name: str
    doc_b_name: str
    doc_a_version: str
    doc_b_version: str
    doc_a_year: str
    doc_b_year: str
    total_sections: int
    added_count: int
    removed_count: int
    modified_count: int
    unchanged_count: int
    changes: List[SectionChange]
    summary: str

    def to_dict(self) -> Dict[str, Any]:
        return {
            "doc_a_name": self.doc_a_name,
            "doc_b_name": self.doc_b_name,
            "doc_a_version": self.doc_a_version,
            "doc_b_version": self.doc_b_version,
            "doc_a_year": self.doc_a_year,
            "doc_b_year": self.doc_b_year,
            "total_sections": self.total_sections,
            "added_count": self.added_count,
            "removed_count": self.removed_count,
            "modified_count": self.modified_count,
            "unchanged_count": self.unchanged_count,
            "changes": [c.to_dict() for c in self.changes],
            "summary": self.summary
        }

class DocumentComparator:
    def __init__(self, doc_repo: DocumentRepository, chunk_repo: ChunkRepository):
        self.doc_repo = doc_repo
        self.chunk_repo = chunk_repo

    def compare_documents(self, doc_id_a: str, doc_id_b: str) -> DocumentComparisonResult:
        doc_a = self.doc_repo.get_by_id(doc_id_a)
        doc_b = self.doc_repo.get_by_id(doc_id_b)

        name_a = doc_a.file_name if doc_a else "Document A"
        name_b = doc_b.file_name if doc_b else "Document B"
        ver_a = doc_a.version if doc_a else "1.0"
        ver_b = doc_b.version if doc_b else "2.0"
        yr_a = doc_a.year if doc_a else "2025"
        yr_b = doc_b.year if doc_b else "2026"

        chunks_a = self.chunk_repo.get_by_document_id(doc_id_a)
        chunks_b = self.chunk_repo.get_by_document_id(doc_id_b)

        sections_a: Dict[str, Dict[str, Any]] = {}
        for c in chunks_a:
            meta = c.metadata
            sec = meta.get("section_title") or f"Section {c.chunk_index + 1}"
            norm_key = sec.lower().replace("section ", "").strip()
            loc = f"Page {meta.get('page_number', 1)}"
            if norm_key not in sections_a:
                sections_a[norm_key] = {"title": sec, "text": c.text, "source": f"{name_a} ({loc}, {sec})"}
            else:
                sections_a[norm_key]["text"] += "\n" + c.text

        sections_b: Dict[str, Dict[str, Any]] = {}
        for c in chunks_b:
            meta = c.metadata
            sec = meta.get("section_title") or f"Section {c.chunk_index + 1}"
            norm_key = sec.lower().replace("section ", "").strip()
            loc = f"Page {meta.get('page_number', 1)}"
            if norm_key not in sections_b:
                sections_b[norm_key] = {"title": sec, "text": c.text, "source": f"{name_b} ({loc}, {sec})"}
            else:
                sections_b[norm_key]["text"] += "\n" + c.text

        if not sections_a or not sections_b:
            for i, c in enumerate(chunks_a):
                k = f"Topic {i+1}"
                sections_a[k] = {"title": k, "text": c.text, "source": f"{name_a} (Chunk {i+1})"}
            for i, c in enumerate(chunks_b):
                k = f"Topic {i+1}"
                sections_b[k] = {"title": k, "text": c.text, "source": f"{name_b} (Chunk {i+1})"}

        changes = ChangeDetector.compare_sections(sections_a, sections_b)

        added = sum(1 for c in changes if c.status == "Added")
        removed = sum(1 for c in changes if c.status == "Removed")
        modified = sum(1 for c in changes if c.status == "Modified")
        unchanged = sum(1 for c in changes if c.status == "Unchanged")

        summary = (
            f"Compared {name_a} (v{ver_a}, {yr_a}) against {name_b} (v{ver_b}, {yr_b}). "
            f"Found {modified} modified section(s), {added} added requirement(s), {removed} removed clause(s), "
            f"and {unchanged} unchanged section(s)."
        )

        return DocumentComparisonResult(
            doc_a_name=name_a,
            doc_b_name=name_b,
            doc_a_version=ver_a,
            doc_b_version=ver_b,
            doc_a_year=yr_a,
            doc_b_year=yr_b,
            total_sections=len(changes),
            added_count=added,
            removed_count=removed,
            modified_count=modified,
            unchanged_count=unchanged,
            changes=changes,
            summary=summary
        )
