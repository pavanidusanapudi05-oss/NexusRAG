import os
from pathlib import Path
from typing import List
import pandas as pd
from .pdf_processor import ExtractedContentBlock

class SpreadsheetProcessor:
    @staticmethod
    def process(file_path: Path) -> List[ExtractedContentBlock]:
        ext = file_path.suffix.lower()
        if ext == ".csv":
            df_map = {"Sheet1": pd.read_csv(str(file_path))}
        else:
            df_map = pd.read_excel(str(file_path), sheet_name=None)

        blocks: List[ExtractedContentBlock] = []

        for page_idx, (sheet_name, df) in enumerate(df_map.items()):
            df = df.fillna("")
            if df.empty:
                continue

            rows_formatted = []
            for r_idx, row in df.iterrows():
                row_items = [f"{col}: {val}" for col, val in row.items() if str(val).strip()]
                if row_items:
                    rows_formatted.append(f"[Row {r_idx+1}] " + " | ".join(row_items))

            sheet_header = f"Sheet: {sheet_name} | Columns: {', '.join(df.columns)} | Records: {len(df)}"
            full_sheet_content = sheet_header + "\n" + "\n".join(rows_formatted)

            blocks.append(ExtractedContentBlock(
                text=full_sheet_content,
                page_number=page_idx + 1,
                section_title=f"Sheet: {sheet_name}",
                sheet_name=sheet_name,
                metadata={"total_rows": len(df), "columns": list(df.columns)}
            ))

        return blocks
