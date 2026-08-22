import re
from dataclasses import dataclass, asdict
from typing import List, Dict, Any, Optional

@dataclass
class GraphEntity:
    entity_id: str
    name: str
    entity_type: str       # "Policy", "Regulation", "Department", "Requirement", "Version", "Date"
    document_id: str
    document_name: str
    chunk_id: str
    page_number: Optional[int] = 1
    section_title: Optional[str] = None
    version: Optional[str] = "1.0"
    year: Optional[str] = "2026"

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

class EntityExtractor:
    @staticmethod
    def extract_from_chunk(chunk_data: Dict[str, Any]) -> List[GraphEntity]:
        text = chunk_data.get("text", "")
        meta = chunk_data.get("metadata", {})
        doc_id = chunk_data.get("document_id", "")
        doc_name = meta.get("document_name", "Document")
        chunk_id = chunk_data.get("chunk_id", "")
        page_num = meta.get("page_number", 1)
        sec_title = meta.get("section_title", "General")
        ver = meta.get("version", "1.0")
        year = meta.get("year", "2026")
        dept = meta.get("department", "General")

        entities: List[GraphEntity] = []

        # 1. Document / Policy Entity
        doc_clean = doc_name.replace(".pdf", "").replace(".docx", "").replace(".xlsx", "").replace(".txt", "").replace("_", " ")
        entities.append(GraphEntity(
            entity_id=f"doc_{doc_id}",
            name=doc_clean,
            entity_type="Policy" if "policy" in doc_clean.lower() else ("Regulation" if "regulation" in doc_clean.lower() else "Document"),
            document_id=doc_id,
            document_name=doc_name,
            chunk_id=chunk_id,
            page_number=page_num,
            section_title=sec_title,
            version=ver,
            year=year
        ))

        # 2. Department Entity
        if dept and dept != "General":
            entities.append(GraphEntity(
                entity_id=f"dept_{dept.lower().replace(' ', '_')}",
                name=dept,
                entity_type="Department",
                document_id=doc_id,
                document_name=doc_name,
                chunk_id=chunk_id,
                page_number=page_num,
                section_title=sec_title,
                version=ver,
                year=year
            ))

        # 3. Requirement extraction based on keywords
        req_patterns = [
            (r'\b(60%|75%)\s+(?:on-site|attendance|presence)\b', "Attendance Rule"),
            (r'\b(?:remote work|work remotely)\b.*?(\d+\s+days?\s+per\s+week)', "Remote Work Allowance"),
            (r'\b(multi-factor authentication|mfa)\b', "Multi-Factor Authentication"),
            (r'\b(aes-256|encryption)\b', "Data Encryption Standard"),
            (r'\b(incident reporting|report within \d+ hours)\b', "Security Incident Reporting"),
            (r'\b(meal allowance|\$\d+\s+per\s+day)\b', "Travel Expense Allowance")
        ]

        text_lower = text.lower()
        for pat, req_name in req_patterns:
            m = re.search(pat, text_lower)
            if m:
                entities.append(GraphEntity(
                    entity_id=f"req_{doc_id}_{req_name.lower().replace(' ', '_')}",
                    name=f"{req_name} ({m.group(0).strip()})",
                    entity_type="Requirement",
                    document_id=doc_id,
                    document_name=doc_name,
                    chunk_id=chunk_id,
                    page_number=page_num,
                    section_title=sec_title,
                    version=ver,
                    year=year
                ))

        return entities
