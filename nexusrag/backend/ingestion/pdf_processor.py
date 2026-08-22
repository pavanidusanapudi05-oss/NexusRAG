import os
import re
from pathlib import Path
from typing import List, Dict, Any
from dataclasses import dataclass, field

@dataclass
class ExtractedContentBlock:
    text: str
    page_number: int = 1
    section_title: Optional[str] = None
    sheet_name: Optional[str] = None
    metadata: Dict[str, Any] = field(default_factory=dict)

class PDFProcessor:
    @staticmethod
    def process(file_path: Path) -> List[ExtractedContentBlock]:
        import pymupdf
        blocks: List[ExtractedContentBlock] = []

        doc = pymupdf.open(str(file_path))
        sec_pattern = re.compile(r'^(Section\s+\d+[^\n]*|Article\s+\d+[^\n]*|[A-Z0-9\.\s]{4,40}:)', re.IGNORECASE)

        for page_idx, page in enumerate(doc):
            page_num = page_idx + 1
            page_text = page.get_text("text")
            if not page_text.strip():
                continue

            lines = page_text.splitlines()
            current_sec = f"Page {page_num} Overview"
            current_lines = []

            for line in lines:
                stripped = line.strip()
                if not stripped:
                    continue
                if sec_pattern.match(stripped) and len(stripped) < 80:
                    if current_lines:
                        blocks.append(ExtractedContentBlock(
                            text="\n".join(current_lines),
                            page_number=page_num,
                            section_title=current_sec,
                            metadata={"source_page": page_num}
                        ))
                        current_lines = []
                    current_sec = stripped
                else:
                    current_lines.append(stripped)

            if current_lines:
                blocks.append(ExtractedContentBlock(
                    text="\n".join(current_lines),
                    page_number=page_num,
                    section_title=current_sec,
                    metadata={"source_page": page_num}
                ))

        doc.close()
        return blocks
