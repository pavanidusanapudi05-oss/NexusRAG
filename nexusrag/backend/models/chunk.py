from dataclasses import dataclass, field, asdict
from typing import Dict, Any, Optional

@dataclass
class ChunkMetadata:
    document_id: str
    document_name: str
    file_type: str
    page_number: Optional[int] = 1
    section_title: Optional[str] = None
    sheet_name: Optional[str] = None
    version: Optional[str] = "1.0"
    year: Optional[str] = "2026"
    department: Optional[str] = "General"
    char_count: int = 0
    token_count: int = 0

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

@dataclass
class DocumentChunk:
    chunk_id: str
    document_id: str
    chunk_index: int
    text: str
    metadata: Dict[str, Any] = field(default_factory=dict)
    char_count: int = 0
    token_count: int = 0

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "DocumentChunk":
        return cls(**data)
