import hashlib
import logging
import datetime
import uuid
from pathlib import Path
from typing import Dict, Any, List, Optional

from nexusrag.backend.models.document import DocumentRecord, ProcessingStatus
from nexusrag.backend.models.chunk import DocumentChunk
from nexusrag.backend.storage.database import DatabaseManager
from nexusrag.backend.storage.repository import DocumentRepository, ChunkRepository
from nexusrag.backend.retrieval.vector_store import LocalVectorStore
from nexusrag.backend.retrieval.embedding_service import BaseEmbeddingProvider, EmbeddingService
from nexusrag.backend.graph.graph_store import LocalKnowledgeGraph

from .pdf_processor import PDFProcessor, ExtractedContentBlock
from .docx_processor import DOCXProcessor
from .text_processor import TextProcessor
from .spreadsheet_processor import SpreadsheetProcessor
from .metadata import MetadataExtractor
from .chunker import DocumentChunker


logger = logging.getLogger("nexusrag.ingestion")

ALLOWED_EXTENSIONS = {
    ".pdf",
    ".docx",
    ".txt",
    ".csv",
    ".xlsx",
    ".xls",
}


class IngestionResult:

    def __init__(
        self,
        success: bool,
        document: Optional[DocumentRecord] = None,
        chunks_created: int = 0,
        vectors_indexed: int = 0,
        is_duplicate: bool = False,
        message: str = "",
    ):
        self.success = success
        self.document = document
        self.chunks_created = chunks_created
        self.vectors_indexed = vectors_indexed
        self.is_duplicate = is_duplicate
        self.message = message


class IngestionPipeline:

    def __init__(
        self,
        db_manager: Optional[DatabaseManager] = None,
        vector_store: Optional[LocalVectorStore] = None,
        embedding_provider: Optional[BaseEmbeddingProvider] = None,
        knowledge_graph: Optional[LocalKnowledgeGraph] = None,
        chunk_size: int = 600,
        chunk_overlap: int = 120,
    ):

        self.db_manager = db_manager or DatabaseManager()

        self.doc_repo = DocumentRepository(
            self.db_manager
        )

        self.chunk_repo = ChunkRepository(
            self.db_manager
        )

        self.vector_store = (
            vector_store or LocalVectorStore()
        )

        self.embedding_provider = (
            embedding_provider
            or EmbeddingService.create_provider(
                "local_dense"
            )
        )

        self.knowledge_graph = (
            knowledge_graph
            or LocalKnowledgeGraph()
        )

        self.chunker = DocumentChunker(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
        )

    @staticmethod
    def compute_sha256(
        file_path: Path,
    ) -> str:

        sha256 = hashlib.sha256()

        with open(file_path, "rb") as f:
            for chunk in iter(
                lambda: f.read(65536),
                b"",
            ):
                sha256.update(chunk)

        return sha256.hexdigest()

    def _get_all_indexed_texts(self) -> List[str]:

        texts: List[str] = []

        for item in self.vector_store.chunks_data:

            text = str(
                item.get("text", "")
            ).strip()

            if text:
                texts.append(text)

        return texts

    def _fit_embedding_provider(
        self,
        additional_texts: Optional[List[str]] = None,
    ) -> None:

        if not hasattr(
            self.embedding_provider,
            "fit",
        ):
            return

        all_texts = self._get_all_indexed_texts()

        if additional_texts:

            all_texts.extend(
                [
                    str(text)
                    for text in additional_texts
                    if str(text).strip()
                ]
            )

        if not all_texts:
            return

        self.embedding_provider.fit(
            all_texts
        )

        logger.info(
            "Embedding provider fitted. "
            "Texts=%d Dimension=%s",
            len(all_texts),
            getattr(
                self.embedding_provider,
                "dimension",
                "unknown",
            ),
        )

    def process_file(
        self,
        file_path: Path,
        custom_meta: Optional[
            Dict[str, Any]
        ] = None,
    ) -> IngestionResult:

        path = Path(file_path)

        if not path.exists():

            return IngestionResult(
                success=False,
                message=f"File not found: {path}",
            )

        ext = path.suffix.lower()

        if ext not in ALLOWED_EXTENSIONS:

            return IngestionResult(
                success=False,
                message=(
                    f"Unsupported file format '{ext}'. "
                    f"Allowed: "
                    f"{sorted(ALLOWED_EXTENSIONS)}"
                ),
            )

        file_size = path.stat().st_size

        if file_size == 0:

            return IngestionResult(
                success=False,
                message=(
                    f"File '{path.name}' "
                    f"is empty (0 bytes)."
                ),
            )

        file_hash = self.compute_sha256(path)

        existing = self.doc_repo.get_by_hash(
            file_hash
        )

        if existing:

            return IngestionResult(
                success=True,
                document=existing,
                chunks_created=existing.chunk_count,
                vectors_indexed=existing.chunk_count,
                is_duplicate=True,
                message=(
                    f"Document '{path.name}' "
                    f"has already been indexed "
                    f"(Document ID: "
                    f"{existing.document_id[:8]}...)."
                ),
            )

        doc_id = str(uuid.uuid4())

        timestamp = (
            datetime.datetime.now()
            .strftime("%Y-%m-%d %H:%M:%S")
        )

        custom_meta = custom_meta or {}

        doc_record = DocumentRecord(
            document_id=doc_id,
            file_name=path.name,
            file_type=ext.lstrip("."),
            file_size_bytes=file_size,
            file_hash=file_hash,
            version=custom_meta.get(
                "version",
                "1.0",
            ),
            year=custom_meta.get(
                "year",
                "2026",
            ),
            department=custom_meta.get(
                "department",
                "General",
            ),
            upload_timestamp=timestamp,
            processing_status=ProcessingStatus.PROCESSING,
            chunk_count=0,
            total_pages=1,
        )

        self.doc_repo.create(
            doc_record
        )

        try:

            blocks: List[
                ExtractedContentBlock
            ] = []

            if ext == ".pdf":

                blocks = PDFProcessor.process(
                    path
                )

            elif ext in {".docx", ".doc"}:

                blocks = DOCXProcessor.process(
                    path
                )

            elif ext in {
                ".xlsx",
                ".xls",
                ".csv",
            }:

                blocks = SpreadsheetProcessor.process(
                    path
                )

            elif ext == ".txt":

                blocks = TextProcessor.process(
                    path
                )

            if not blocks:

                self.doc_repo.update_status(
                    doc_id,
                    ProcessingStatus.FAILED,
                    error_message=(
                        "No readable text or "
                        "content extracted."
                    ),
                )

                return IngestionResult(
                    success=False,
                    message=(
                        f"Document '{path.name}' "
                        f"contains no extractable text."
                    ),
                )

            total_pages = max(
                [
                    b.page_number
                    for b in blocks
                ] + [1]
            )

            sample_text = " ".join(
                b.text
                for b in blocks[:3]
            )

            inferred_meta = (
                MetadataExtractor.extract_metadata(
                    path.name,
                    ext.lstrip("."),
                    file_size,
                    file_hash,
                    sample_text,
                )
            )

            inferred_meta.update(
                custom_meta
            )

            chunks = self.chunker.chunk_blocks(
                document_id=doc_id,
                doc_name=path.name,
                file_type=ext.lstrip("."),
                blocks=blocks,
                doc_metadata=inferred_meta,
            )

            self.chunk_repo.save_chunks(
                chunks
            )

            chunk_texts = [
                c.text
                for c in chunks
            ]

            self._fit_embedding_provider(
                additional_texts=chunk_texts
            )

            embeddings = (
                self.embedding_provider.embed_texts(
                    chunk_texts
                )
            )

            self.vector_store.add_chunks(
                doc_id,
                chunks,
                embeddings,
            )

            chunks_data = [
                c.to_dict()
                for c in chunks
            ]

            self.knowledge_graph.add_document_chunks(
                doc_id,
                chunks_data,
            )

            self.doc_repo.update_status(
                document_id=doc_id,
                status=ProcessingStatus.PROCESSED,
                chunk_count=len(chunks),
                total_pages=total_pages,
                version=inferred_meta.get(
                    "version",
                    "1.0",
                ),
                year=inferred_meta.get(
                    "year",
                    "2026",
                ),
                department=inferred_meta.get(
                    "department",
                    "General",
                ),
            )

            doc_record.processing_status = (
                ProcessingStatus.PROCESSED
            )
            doc_record.chunk_count = len(
                chunks
            )
            doc_record.total_pages = (
                total_pages
            )
            doc_record.version = inferred_meta.get(
                "version",
                "1.0",
            )
            doc_record.year = inferred_meta.get(
                "year",
                "2026",
            )
            doc_record.department = inferred_meta.get(
                "department",
                "General",
            )

            return IngestionResult(
                success=True,
                document=doc_record,
                chunks_created=len(chunks),
                vectors_indexed=len(chunks),
                is_duplicate=False,
                message=(
                    f"Successfully ingested & indexed "
                    f"'{path.name}': {len(chunks)} chunks, "
                    f"vectors, and graph entities created."
                ),
            )

        except Exception as e:

            logger.exception(
                "Error processing %s: %s",
                path.name,
                e,
            )

            error_msg = (
                f"Processing error: {str(e)}"
            )

            self.doc_repo.update_status(
                doc_id,
                ProcessingStatus.FAILED,
                error_message=error_msg,
            )

            return IngestionResult(
                success=False,
                message=(
                    f"Failed to process "
                    f"'{path.name}': "
                    f"{error_msg}"
                ),
            )

    def delete_document(
        self,
        document_id: str,
    ) -> bool:

        db_deleted = (
            self.doc_repo.delete(
                document_id
            )
        )

        self.vector_store.delete_document_vectors(
            document_id
        )

        self.knowledge_graph.delete_document_graph(
            document_id
        )

        return db_deleted

    def reindex_document(
        self,
        document_id: str,
    ) -> int:

        chunks = (
            self.chunk_repo.get_by_document_id(
                document_id
            )
        )

        if not chunks:
            return 0

        chunk_texts = [
            c.text
            for c in chunks
        ]

        self._fit_embedding_provider(
            additional_texts=chunk_texts
        )

        embeddings = (
            self.embedding_provider.embed_texts(
                chunk_texts
            )
        )

        self.vector_store.add_chunks(
            document_id,
            chunks,
            embeddings,
        )

        self.knowledge_graph.add_document_chunks(
            document_id,
            [
                c.to_dict()
                for c in chunks
            ],
        )

        return len(chunks)