import sqlite3
from pathlib import Path
from typing import Optional

DEFAULT_DB_PATH = Path("nexusrag/data/nexusrag.db")

class DatabaseManager:
    def __init__(self, db_path: Optional[Path] = None):
        self.db_path = db_path or DEFAULT_DB_PATH
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self._conn: Optional[sqlite3.Connection] = None
        self.init_database()

    def get_connection(self) -> sqlite3.Connection:
        if self._conn is None:
            self._conn = sqlite3.connect(str(self.db_path), check_same_thread=False)
            self._conn.row_factory = sqlite3.Row
            self._conn.execute("PRAGMA foreign_keys = ON")
        return self._conn

    def close(self):
        if self._conn is not None:
            try:
                self._conn.close()
            except Exception:
                pass
            self._conn = None

    def init_database(self):
        conn = self.get_connection()
        # 1. Documents Table
        conn.execute("""
        CREATE TABLE IF NOT EXISTS documents (
            document_id TEXT PRIMARY KEY,
            file_name TEXT NOT NULL,
            file_type TEXT NOT NULL,
            file_size_bytes INTEGER NOT NULL,
            file_hash TEXT NOT NULL UNIQUE,
            version TEXT DEFAULT '1.0',
            year TEXT DEFAULT '2026',
            department TEXT DEFAULT 'General',
            upload_timestamp TEXT NOT NULL,
            processing_status TEXT NOT NULL,
            chunk_count INTEGER DEFAULT 0,
            total_pages INTEGER DEFAULT 1,
            error_message TEXT
        )
        """)

        # 2. Chunks Table
        conn.execute("""
        CREATE TABLE IF NOT EXISTS chunks (
            chunk_id TEXT PRIMARY KEY,
            document_id TEXT NOT NULL,
            chunk_index INTEGER NOT NULL,
            text TEXT NOT NULL,
            page_number INTEGER,
            section_title TEXT,
            sheet_name TEXT,
            version TEXT,
            year TEXT,
            department TEXT,
            char_count INTEGER NOT NULL,
            token_count INTEGER NOT NULL,
            metadata_json TEXT NOT NULL,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            FOREIGN KEY (document_id) REFERENCES documents(document_id) ON DELETE CASCADE
        )
        """)

        conn.execute("CREATE INDEX IF NOT EXISTS idx_doc_hash ON documents(file_hash)")
        conn.execute("CREATE INDEX IF NOT EXISTS idx_chunk_doc ON chunks(document_id)")
        conn.commit()
