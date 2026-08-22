import json
import networkx as nx
from pathlib import Path
from typing import List, Dict, Any, Optional
from .entity_extractor import GraphEntity, EntityExtractor
from .relation_extractor import GraphRelation, RelationExtractor

class LocalKnowledgeGraph:
    def __init__(self, persist_path: Optional[Path] = None):
        self.persist_path = persist_path or Path("nexusrag/data/graph/knowledge_graph.json")
        self.persist_path.parent.mkdir(parents=True, exist_ok=True)
        self.graph = nx.DiGraph()
        self.entities: Dict[str, GraphEntity] = {}
        self.relations: List[GraphRelation] = []
        self.load()

    def add_document_chunks(self, document_id: str, chunks_data: List[Dict[str, Any]]):
        self.delete_document_graph(document_id, auto_save=False)

        all_doc_entities: List[GraphEntity] = []
        for c in chunks_data:
            extracted_ents = EntityExtractor.extract_from_chunk(c)
            for ent in extracted_ents:
                self.entities[ent.entity_id] = ent
                self.graph.add_node(
                    ent.entity_id,
                    name=ent.name,
                    type=ent.entity_type,
                    document_id=ent.document_id,
                    document_name=ent.document_name,
                    version=ent.version,
                    year=ent.year
                )
                all_doc_entities.append(ent)

        extracted_rels = RelationExtractor.extract_relations(all_doc_entities)
        for rel in extracted_rels:
            self.relations.append(rel)
            self.graph.add_edge(
                rel.source_entity_id,
                rel.target_entity_id,
                relation=rel.relation_type,
                document_name=rel.document_name,
                version=rel.version
            )

        self.save()

    def delete_document_graph(self, document_id: str, auto_save: bool = True):
        # Remove relations
        self.relations = [r for r in self.relations if r.document_id != document_id]

        # Remove entities
        nodes_to_remove = [e_id for e_id, ent in self.entities.items() if ent.document_id == document_id]
        for n in nodes_to_remove:
            if n in self.entities:
                del self.entities[n]
            if self.graph.has_node(n):
                self.graph.remove_node(n)

        if auto_save:
            self.save()

    def get_stats(self) -> Dict[str, int]:
        return {
            "total_entities": len(self.entities),
            "total_relations": len(self.relations),
            "graph_nodes": self.graph.number_of_nodes(),
            "graph_edges": self.graph.number_of_edges()
        }

    def list_entities(self, entity_type: Optional[str] = None, doc_name: Optional[str] = None) -> List[GraphEntity]:
        res = list(self.entities.values())
        if entity_type:
            res = [e for e in res if e.entity_type.lower() == entity_type.lower()]
        if doc_name:
            res = [e for e in res if e.document_name == doc_name]
        return res

    def list_relations(self) -> List[GraphRelation]:
        return list(self.relations)

    def save(self):
        data = {
            "entities": [e.to_dict() for e in self.entities.values()],
            "relations": [r.to_dict() for r in self.relations]
        }
        with open(self.persist_path, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2)

    def load(self):
        if not self.persist_path.exists():
            return
        try:
            with open(self.persist_path, "r", encoding="utf-8") as f:
                data = json.load(f)
            
            self.graph.clear()
            self.entities = {}
            self.relations = []

            for ed in data.get("entities", []):
                ent = GraphEntity(**ed)
                self.entities[ent.entity_id] = ent
                self.graph.add_node(
                    ent.entity_id,
                    name=ent.name,
                    type=ent.entity_type,
                    document_id=ent.document_id,
                    document_name=ent.document_name,
                    version=ent.version,
                    year=ent.year
                )

            for rd in data.get("relations", []):
                rel = GraphRelation(**rd)
                self.relations.append(rel)
                self.graph.add_edge(
                    rel.source_entity_id,
                    rel.target_entity_id,
                    relation=rel.relation_type,
                    document_name=rel.document_name,
                    version=rel.version
                )
        except Exception as e:
            print(f"Error loading knowledge graph: {e}")
