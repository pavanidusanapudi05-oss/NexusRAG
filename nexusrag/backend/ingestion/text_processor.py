import os
from pathlib import Path
from typing import List
from .pdf_processor import ExtractedContentBlock

class TextProcessor:
    @staticmethod
    def process(file_path: Path) -> List[ExtractedContentBlock]:
        with open(file_path, "r", encoding="utf-8", errors="ignore") as f:
            raw_text = f.read()

        if not raw_text.strip():
            return []

        lines = raw_text.splitlines()
        blocks: List[ExtractedContentBlock] = []
        current_title = "Section 1"
        current_lines = []

        for line in lines:
            stripped = line.strip()
            if not stripped:
                continue
            if stripped.startswith("#") or stripped.lower().startswith("section"):
                if current_lines:
                    blocks.append(ExtractedContentBlock(
                        text="\n".join(current_lines),
                        page_number=1,
                        section_title=current_title
                    ))
                    current_lines = []
                current_title = stripped.lstrip("#").strip()
            else:
                current_lines.append(stripped)

        if current_lines:
            blocks.append(ExtractedContentBlock(
                text="\n".join(current_lines),
                page_number=1,
                section_title=current_title
            ))

        return blocks
