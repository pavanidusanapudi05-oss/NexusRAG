import enum
from dataclasses import dataclass, field, asdict
from typing import Dict, Any, Optional

class ProcessingStatus(str, enum.Enum):
    UPLOADED = "Uploaded"
    PROCESSING = "Processing"
    PROCESSED = "Processed"
    FAILED = "Failed"

@dataclass
class DocumentMetadata:
    document_id: str
    document_name: str
    file_type: str
    upload_time: str
    file_size_bytes: int
    file_hash: str
    version: Optional[str] = "1.0"
    year: Optional[str] = "2026"
    department: Optional[str] = "General"
    category: Optional[str] = "Policy"
    total_pages: int = 1

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

@dataclass
class DocumentRecord:
    document_id: str
    file_name: str
    file_type: str
    file_size_bytes: int
    file_hash: str
    version: str
    year: str
    department: str
    upload_timestamp: str
    processing_status: ProcessingStatus
    chunk_count: int = 0
    total_pages: int = 1
    error_message: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        d = asdict(self)
        d["processing_status"] = self.processing_status.value
        return d
