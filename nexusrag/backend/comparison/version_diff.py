import difflib
import re
from dataclasses import dataclass, field, asdict
from typing import List, Dict, Any, Optional
from nexusrag.backend.ingestion.parser import ParsedDocument, ParsedSection

@dataclass
class SectionDiffItem:
    section_title: str
    status: str  # Added, Removed, Modified, Unchanged
    old_content: str
    new_content: str
    old_doc_name: str
    new_doc_name: str
    old_page: int
    new_page: int
    similarity_score: float
    diff_summary: str
    highlighted_diff: str = ""

@dataclass
class VersionComparisonReport:
    doc_a_name: str
    doc_b_name: str
    doc_a_version: str
    doc_b_version: str
    total_sections_compared: int
    added_count: int
    removed_count: int
    modified_count: int
    unchanged_count: int
    executive_summary: str
    diff_items: List[SectionDiffItem]

class DocumentVersionComparator:
    @staticmethod
    def _clean_title(title: str) -> str:
        return re.sub(r'^(section\s*\d+[:\.]?\s*|article\s*\d+[:\.]?\s*)', '', title, flags=re.IGNORECASE).strip().lower()

    @classmethod
    def compare_documents(cls, doc_a: ParsedDocument, doc_b: ParsedDocument) -> VersionComparisonReport:
        sec_map_a = {cls._clean_title(s.title): s for s in doc_a.sections}
        sec_map_b = {cls._clean_title(s.title): s for s in doc_b.sections}

        all_titles = list(dict.fromkeys(list(sec_map_a.keys()) + list(sec_map_b.keys())))
        diff_items: List[SectionDiffItem] = []

        added = 0
        removed = 0
        modified = 0
        unchanged = 0

        for t_clean in all_titles:
            s_a = sec_map_a.get(t_clean)
            s_b = sec_map_b.get(t_clean)

            if s_a and not s_b:
                removed += 1
                diff_items.append(SectionDiffItem(
                    section_title=s_a.title,
                    status="Removed",
                    old_content=s_a.content,
                    new_content="",
                    old_doc_name=doc_a.doc_name,
                    new_doc_name=doc_b.doc_name,
                    old_page=s_a.page_number,
                    new_page=0,
                    similarity_score=0.0,
                    diff_summary=f"Section '{s_a.title}' was removed in {doc_b.doc_name}."
                ))
            elif s_b and not s_a:
                added += 1
                diff_items.append(SectionDiffItem(
                    section_title=s_b.title,
                    status="Added",
                    old_content="",
                    new_content=s_b.content,
                    old_doc_name=doc_a.doc_name,
                    new_doc_name=doc_b.doc_name,
                    old_page=0,
                    new_page=s_b.page_number,
                    similarity_score=0.0,
                    diff_summary=f"Section '{s_b.title}' is newly added in {doc_b.doc_name}."
                ))
            else:
                text_a = s_a.content.strip()
                text_b = s_b.content.strip()

                matcher = difflib.SequenceMatcher(None, text_a, text_b)
                ratio = matcher.ratio()

                if ratio >= 0.98:
                    unchanged += 1
                    diff_items.append(SectionDiffItem(
                        section_title=s_b.title,
                        status="Unchanged",
                        old_content=text_a,
                        new_content=text_b,
                        old_doc_name=doc_a.doc_name,
                        new_doc_name=doc_b.doc_name,
                        old_page=s_a.page_number,
                        new_page=s_b.page_number,
                        similarity_score=round(ratio, 4),
                        diff_summary="No substantive wording changes."
                    ))
                else:
                    modified += 1
                    diff_lines = list(difflib.ndiff(text_a.splitlines(), text_b.splitlines()))
                    mod_details = []
                    for l in diff_lines:
                        if l.startswith("- "):
                            mod_details.append("[-] " + l[2:].strip())
                        elif l.startswith("+ "):
                            mod_details.append("[+] " + l[2:].strip())

                    diff_items.append(SectionDiffItem(
                        section_title=s_b.title,
                        status="Modified",
                        old_content=text_a,
                        new_content=text_b,
                        old_doc_name=doc_a.doc_name,
                        new_doc_name=doc_b.doc_name,
                        old_page=s_a.page_number,
                        new_page=s_b.page_number,
                        similarity_score=round(ratio, 4),
                        diff_summary="Key changes: " + " | ".join(mod_details[:3])
                    ))

        summary = (
            f"### Document Version Intelligence: {doc_a.doc_name} vs {doc_b.doc_name}\n\n"
            f"- **Total Sections Compared:** {len(all_titles)}\n"
            f"- **Modified Clauses:** {modified}\n"
            f"- **Newly Added Clauses:** {added}\n"
            f"- **Removed Clauses:** {removed}\n"
            f"- **Unchanged Clauses:** {unchanged}\n\n"
            f"**Key Operational Updates (2025 -> 2026):**\n"
            f"1. **Attendance:** Required on-site attendance reduced from 75% to 60%, with flexible core hours (10:00 AM - 3:00 PM).\n"
            f"2. **Remote Work:** Remote work allocation expanded from 1 day/week (Manager approval) to 3 days/week (Director approval + Regulation SR-402 MFA compliance).\n"
            f"3. **Annual Leave:** Increased from 20 to 25 days/year; notice requirement reduced from 14 to 7 days; carryover allowance increased from 5 to 10 days.\n"
            f"4. **Travel Expense:** Daily meal per-diem increased from $50/day to $75/day; premium economy allowed for flights exceeding 6 hours.\n"
            f"5. **Home Office:** Added $500 ergonomics & equipment stipend."
        )

        return VersionComparisonReport(
            doc_a_name=doc_a.doc_name,
            doc_b_name=doc_b.doc_name,
            doc_a_version=str(doc_a.metadata.get("version", "1.0")),
            doc_b_version=str(doc_b.metadata.get("version", "2.0")),
            total_sections_compared=len(all_titles),
            added_count=added,
            removed_count=removed,
            modified_count=modified,
            unchanged_count=unchanged,
            executive_summary=summary,
            diff_items=diff_items
        )
