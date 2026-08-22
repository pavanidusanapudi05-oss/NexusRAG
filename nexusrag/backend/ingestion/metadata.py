import re
from pathlib import Path
from typing import Dict, Any, Optional

class MetadataExtractor:
    @staticmethod
    def extract_metadata(file_name: str, file_type: str, file_size: int, file_hash: str, text_sample: str = "") -> Dict[str, Any]:
        name_lower = file_name.lower()
        text_lower = text_sample[:2000].lower() if text_sample else ""
        combined = f"{name_lower} {text_lower}"

        # Default fallback values
        meta = {
            "version": "1.0",
            "year": "2026",
            "department": "General",
            "category": "Policy"
        }

        # 1. Year Detection
        year_match = re.search(r'\b(202[0-9]|201[5-9])\b', combined)
        if year_match:
            meta["year"] = year_match.group(1)

        # 2. Version Detection
        ver_match = re.search(r'\b(v\s*\d+(\.\d+)?|version\s*\d+(\.\d+)?)\b', combined)
        if ver_match:
            meta["version"] = re.sub(r'[^0-9\.]', '', ver_match.group(1)).strip('.')
        elif "2025" in name_lower:
            meta["version"] = "1.0"
        elif "2026" in name_lower:
            meta["version"] = "2.0"

        # 3. Department & Category Detection
        if "security" in combined or "cyber" in combined or "sr402" in combined or "sr-402" in combined:
            meta["department"] = "Information Security"
            meta["category"] = "Regulation"
        elif "infra" in combined or "technology" in combined or "manual" in combined or "it_" in name_lower:
            meta["department"] = "Information Technology"
            meta["category"] = "Technical Manual"
        elif "operations" in combined or "employee" in combined or "hr" in combined:
            meta["department"] = "Human Resources & Operations"
            meta["category"] = "Policy"
        elif "audit" in combined or "compliance" in combined:
            meta["department"] = "Regulatory Compliance"
            meta["category"] = "Audit Guidelines"

        return meta
