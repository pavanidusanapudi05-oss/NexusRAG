import json
from typing import List, Optional, Dict, Any
from .database import DatabaseManager
from nexusrag.backend.models.document import DocumentRecord, ProcessingStatus
from nexusrag.backend.models.chunk import DocumentChunk

class DocumentRepository:
    def __init__(self, db_manager: DatabaseManager):
        self.db_manager = db_manager

    def create(self, doc: DocumentRecord) -> DocumentRecord:
        with self.db_manager.get_connection() as conn:
            conn.execute("""
                INSERT INTO documents (
                    document_id, file_name, file_type, file_size_bytes, file_hash,
                    version, year, department, upload_timestamp, processing_status,
                    chunk_count, total_pages, error_message
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """, (
                doc.document_id, doc.file_name, doc.file_type, doc.file_size_bytes, doc.file_hash,
                doc.version, doc.year, doc.department, doc.upload_timestamp, doc.processing_status.value,
                doc.chunk_count, doc.total_pages, doc.error_message
            ))
            conn.commit()
        return doc

    def get_by_id(self, document_id: str) -> Optional[DocumentRecord]:
        with self.db_manager.get_connection() as conn:
            cur = conn.execute("SELECT * FROM documents WHERE document_id = ?", (document_id,))
            row = cur.fetchone()
            if not row:
                return None
            return self._row_to_record(row)

    def get_by_hash(self, file_hash: str) -> Optional[DocumentRecord]:
        with self.db_manager.get_connection() as conn:
            cur = conn.execute("SELECT * FROM documents WHERE file_hash = ?", (file_hash,))
            row = cur.fetchone()
            if not row:
                return None
            return self._row_to_record(row)

    def get_by_name(self, file_name: str) -> Optional[DocumentRecord]:
        docs = self.list_all()
        for d in docs:
            if d.file_name == file_name:
                return d
        return None

    def list_all(self) -> List[DocumentRecord]:
        with self.db_manager.get_connection() as conn:
            cur = conn.execute("SELECT * FROM documents ORDER BY upload_timestamp DESC")
            return [self._row_to_record(r) for r in cur.fetchall()]

    def update_status(
        self,
        document_id: str,
        status: ProcessingStatus,
        chunk_count: int = 0,
        total_pages: int = 1,
        version: Optional[str] = None,
        year: Optional[str] = None,
        department: Optional[str] = None,
        error_message: Optional[str] = None
    ):
        with self.db_manager.get_connection() as conn:
            updates = ["processing_status = ?", "chunk_count = ?", "total_pages = ?", "error_message = ?"]
            params = [status.value, chunk_count, total_pages, error_message]
            if version is not None:
                updates.append("version = ?")
                params.append(version)
            if year is not None:
                updates.append("year = ?")
                params.append(year)
            if department is not None:
                updates.append("department = ?")
                params.append(department)
            params.append(document_id)

            sql = f"UPDATE documents SET {', '.join(updates)} WHERE document_id = ?"
            conn.execute(sql, tuple(params))
            conn.commit()

    def delete(self, document_id: str) -> bool:
        with self.db_manager.get_connection() as conn:
            conn.execute("DELETE FROM chunks WHERE document_id = ?", (document_id,))
            cur = conn.execute("DELETE FROM documents WHERE document_id = ?", (document_id,))
            conn.commit()
            return cur.rowcount > 0

    def get_stats(self) -> Dict[str, int]:
        with self.db_manager.get_connection() as conn:
            total_docs = conn.execute("SELECT COUNT(*) FROM documents").fetchone()[0]
            processed_docs = conn.execute("SELECT COUNT(*) FROM documents WHERE processing_status = 'Processed'").fetchone()[0]
            failed_docs = conn.execute("SELECT COUNT(*) FROM documents WHERE processing_status = 'Failed'").fetchone()[0]
            total_chunks = conn.execute("SELECT COUNT(*) FROM chunks").fetchone()[0]

            return {
                "total_documents": total_docs,
                "processed_documents": processed_docs,
                "failed_documents": failed_docs,
                "total_chunks": total_chunks
            }

    @staticmethod
    def _row_to_record(row) -> DocumentRecord:
        return DocumentRecord(
            document_id=row["document_id"],
            file_name=row["file_name"],
            file_type=row["file_type"],
            file_size_bytes=row["file_size_bytes"],
            file_hash=row["file_hash"],
            version=row["version"],
            year=row["year"],
            department=row["department"],
            upload_timestamp=row["upload_timestamp"],
            processing_status=ProcessingStatus(row["processing_status"]),
            chunk_count=row["chunk_count"],
            total_pages=row["total_pages"],
            error_message=row["error_message"]
        )

class ChunkRepository:
    def __init__(self, db_manager: DatabaseManager):
        self.db_manager = db_manager

    def save_chunks(self, chunks: List[DocumentChunk]):
        if not chunks:
            return
        with self.db_manager.get_connection() as conn:
            for c in chunks:
                meta = c.metadata
                conn.execute("""
                    INSERT OR REPLACE INTO chunks (
                        chunk_id, document_id, chunk_index, text,
                        page_number, section_title, sheet_name,
                        version, year, department, char_count, token_count,
                        metadata_json
                    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """, (
                    c.chunk_id, c.document_id, c.chunk_index, c.text,
                    meta.get("page_number"), meta.get("section_title"), meta.get("sheet_name"),
                    meta.get("version"), meta.get("year"), meta.get("department"),
                    c.char_count, c.token_count, json.dumps(meta)
                ))
            conn.commit()

    def get_by_document_id(self, document_id: str) -> List[DocumentChunk]:
        with self.db_manager.get_connection() as conn:
            cur = conn.execute("SELECT * FROM chunks WHERE document_id = ? ORDER BY chunk_index ASC", (document_id,))
            results = []
            for row in cur.fetchall():
                meta = json.loads(row["metadata_json"]) if row["metadata_json"] else {}
                results.append(DocumentChunk(
                    chunk_id=row["chunk_id"],
                    document_id=row["document_id"],
                    chunk_index=row["chunk_index"],
                    text=row["text"],
                    char_count=row["char_count"],
                    token_count=row["token_count"],
                    metadata=meta
                ))
            return results

    def list_all(self, limit: int = 100) -> List[DocumentChunk]:
        with self.db_manager.get_connection() as conn:
            cur = conn.execute("SELECT * FROM chunks ORDER BY created_at DESC LIMIT ?", (limit,))
            results = []
            for row in cur.fetchall():
                meta = json.loads(row["metadata_json"]) if row["metadata_json"] else {}
                results.append(DocumentChunk(
                    chunk_id=row["chunk_id"],
                    document_id=row["document_id"],
                    chunk_index=row["chunk_index"],
                    text=row["text"],
                    char_count=row["char_count"],
                    token_count=row["token_count"],
                    metadata=meta
                ))
            return results
