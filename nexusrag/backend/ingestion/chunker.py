import re
from typing import List, Dict, Any, Optional
from nexusrag.backend.models.chunk import DocumentChunk
from .pdf_processor import ExtractedContentBlock

class DocumentChunker:
    def __init__(self, chunk_size: int = 600, chunk_overlap: int = 120):
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap

    def chunk_blocks(self, document_id: str, doc_name: str, file_type: str, blocks: List[ExtractedContentBlock], doc_metadata: Dict[str, Any]) -> List[DocumentChunk]:
        chunks: List[DocumentChunk] = []
        global_chunk_idx = 0

        version = str(doc_metadata.get("version", "1.0"))
        year = str(doc_metadata.get("year", "2026"))
        department = str(doc_metadata.get("department", "General"))

        for b in blocks:
            b_text = b.text.strip()
            if not b_text:
                continue

            page_num = b.page_number
            sec_title = b.section_title or f"Page {page_num}"
            sheet_name = b.sheet_name

            # If block text fits in single chunk, keep atomic
            if len(b_text) <= self.chunk_size:
                cid = f"{document_id}_p{page_num}_c{global_chunk_idx}"
                meta = {
                    "document_id": document_id,
                    "document_name": doc_name,
                    "file_type": file_type,
                    "page_number": page_num,
                    "section_title": sec_title,
                    "sheet_name": sheet_name,
                    "version": version,
                    "year": year,
                    "department": department
                }
                chunks.append(DocumentChunk(
                    chunk_id=cid,
                    document_id=document_id,
                    chunk_index=global_chunk_idx,
                    text=b_text,
                    char_count=len(b_text),
                    token_count=max(1, len(b_text.split())),
                    metadata=meta
                ))
                global_chunk_idx += 1
            else:
                # Sliding window chunker over paragraphs / lines
                paragraphs = [p.strip() for p in b_text.split("\n") if p.strip()]
                buffer = ""

                for p in paragraphs:
                    if len(buffer) + len(p) + 1 <= self.chunk_size:
                        buffer = (buffer + "\n" + p).strip()
                    else:
                        if buffer:
                            cid = f"{document_id}_p{page_num}_c{global_chunk_idx}"
                            meta = {
                                "document_id": document_id,
                                "document_name": doc_name,
                                "file_type": file_type,
                                "page_number": page_num,
                                "section_title": sec_title,
                                "sheet_name": sheet_name,
                                "version": version,
                                "year": year,
                                "department": department
                            }
                            chunks.append(DocumentChunk(
                                chunk_id=cid,
                                document_id=document_id,
                                chunk_index=global_chunk_idx,
                                text=buffer,
                                char_count=len(buffer),
                                token_count=max(1, len(buffer.split())),
                                metadata=meta
                            ))
                            global_chunk_idx += 1

                        overlap_text = buffer[-self.chunk_overlap:] if len(buffer) > self.chunk_overlap else ""
                        buffer = (overlap_text + "\n" + p).strip()

                if buffer:
                    cid = f"{document_id}_p{page_num}_c{global_chunk_idx}"
                    meta = {
                        "document_id": document_id,
                        "document_name": doc_name,
                        "file_type": file_type,
                        "page_number": page_num,
                        "section_title": sec_title,
                        "sheet_name": sheet_name,
                        "version": version,
                        "year": year,
                        "department": department
                    }
                    chunks.append(DocumentChunk(
                        chunk_id=cid,
                        document_id=document_id,
                        chunk_index=global_chunk_idx,
                        text=buffer,
                        char_count=len(buffer),
                        token_count=max(1, len(buffer.split())),
                        metadata=meta
                    ))
                    global_chunk_idx += 1

        return chunks
