import re
from typing import Dict, Any, Optional

class VersionDetector:
    @staticmethod
    def detect_version_and_year(filename: str, sample_text: str = "") -> Dict[str, str]:
        combined = f"{filename} {sample_text[:1000]}".lower()

        year = "2026"
        year_match = re.search(r'\b(202[0-9]|201[5-9])\b', combined)
        if year_match:
            year = year_match.group(1)

        version = "1.0"
        ver_match = re.search(r'\b(v\s*\d+(\.\d+)?|version\s*\d+(\.\d+)?|pol-\d+-v(\d+))\b', combined)
        if ver_match:
            clean_ver = re.sub(r'[^0-9\.]', '', ver_match.group(0)).strip('.')
            version = clean_ver if clean_ver else "1.0"
        elif "2025" in combined:
            version = "1.0"
        elif "2026" in combined:
            version = "2.0"

        return {"version": version, "year": year}
