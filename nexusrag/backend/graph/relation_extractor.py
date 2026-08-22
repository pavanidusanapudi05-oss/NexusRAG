from dataclasses import dataclass, asdict
from typing import List, Dict, Any, Optional
from .entity_extractor import GraphEntity

@dataclass
class GraphRelation:
    relation_id: str
    source_entity_id: str
    source_name: str
    relation_type: str     # "REQUIRES", "APPLIES_TO", "CONTAINS", "UPDATED_BY", "SUPERSEDES"
    target_entity_id: str
    target_name: str
    document_id: str
    document_name: str
    chunk_id: str
    page_number: Optional[int] = 1
    section_title: Optional[str] = None
    version: Optional[str] = "1.0"
    year: Optional[str] = "2026"

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

class RelationExtractor:
    @staticmethod
    def extract_relations(entities: List[GraphEntity]) -> List[GraphRelation]:
        relations: List[GraphRelation] = []
        if not entities:
            return relations

        # Group by document
        doc_entity = next((e for e in entities if e.entity_type in ["Policy", "Regulation", "Document"]), None)
        dept_entity = next((e for e in entities if e.entity_type == "Department"), None)
        req_entities = [e for e in entities if e.entity_type == "Requirement"]

        if doc_entity and dept_entity:
            relations.append(GraphRelation(
                relation_id=f"rel_{doc_entity.entity_id}_applies_{dept_entity.entity_id}",
                source_entity_id=doc_entity.entity_id,
                source_name=doc_entity.name,
                relation_type="APPLIES_TO",
                target_entity_id=dept_entity.entity_id,
                target_name=dept_entity.name,
                document_id=doc_entity.document_id,
                document_name=doc_entity.document_name,
                chunk_id=doc_entity.chunk_id,
                page_number=doc_entity.page_number,
                section_title=doc_entity.section_title,
                version=doc_entity.version,
                year=doc_entity.year
            ))

        if doc_entity:
            for req in req_entities:
                relations.append(GraphRelation(
                    relation_id=f"rel_{doc_entity.entity_id}_req_{req.entity_id}",
                    source_entity_id=doc_entity.entity_id,
                    source_name=doc_entity.name,
                    relation_type="REQUIRES",
                    target_entity_id=req.entity_id,
                    target_name=req.name,
                    document_id=doc_entity.document_id,
                    document_name=doc_entity.document_name,
                    chunk_id=req.chunk_id,
                    page_number=req.page_number,
                    section_title=req.section_title,
                    version=req.version,
                    year=req.year
                ))

        return relations
