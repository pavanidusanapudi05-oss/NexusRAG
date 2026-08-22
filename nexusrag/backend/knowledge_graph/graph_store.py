import json
import networkx as nx
from pathlib import Path
from typing import List, Dict, Any, Optional, Set, Tuple
from .extractor import Entity, Relation

ENTITY_COLORS = {
    'Policy': '#8B5CF6',       # Violet
    'Regulation': '#EF4444',   # Red
    'Requirement': '#10B981',  # Emerald Green
    'Process': '#3B82F6',      # Blue
    'Department': '#06B6D4',   # Cyan
    'Organization': '#F59E0B', # Amber
    'Technical Manual': '#6366F1', # Indigo
    'System': '#EC4899'        # Pink
}

class KnowledgeGraphStore:
    def __init__(self, persist_dir: Optional[str] = None):
        self.persist_dir = Path(persist_dir) if persist_dir else Path('nexusrag/data/processed')
        self.persist_dir.mkdir(parents=True, exist_ok=True)
        self.graph = nx.MultiDiGraph()
        self.entities: Dict[str, Entity] = {}
        self.relations: List[Relation] = []

    def build_from_extracted(self, entities: List[Entity], relations: List[Relation]):
        self.graph.clear()
        self.entities = {e.id: e for e in entities}
        self.relations = list(relations)

        for e in entities:
            color = ENTITY_COLORS.get(e.type, '#94A3B8')
            self.graph.add_node(
                e.id,
                name=e.name,
                type=e.type,
                color=color,
                properties=e.properties,
                title=f'{e.type}: {e.name}'
            )

        for r in relations:
            if self.graph.has_node(r.source_id) and self.graph.has_node(r.target_id):
                self.graph.add_edge(
                    r.source_id,
                    r.target_id,
                    type=r.type,
                    label=r.type.replace('_', ' '),
                    properties=r.properties
                )

        self.save()

    def query_impact(self, entity_query: str) -> Dict[str, Any]:
        matched_nodes = []
        q_lower = entity_query.lower()
        
        for n, data in self.graph.nodes(data=True):
            if q_lower in data.get('name', '').lower() or q_lower in n.lower():
                matched_nodes.append(n)

        impact_summary = []
        paths_found = []

        for node_id in matched_nodes:
            node_name = self.graph.nodes[node_id].get('name', node_id)
            # Outgoing edges (what this node affects or requires)
            for _, target_id, edge_data in self.graph.out_edges(node_id, data=True):
                target_name = self.graph.nodes[target_id].get('name', target_id)
                rel_type = edge_data.get('type', 'RELATES_TO')
                props = edge_data.get('properties', {})
                prop_str = f' ({props})' if props else ''
                impact_summary.append({
                    'source': node_name,
                    'relation': rel_type,
                    'target': target_name,
                    'direction': 'outgoing',
                    'detail': f'{node_name} --[{rel_type}]--> {target_name}{prop_str}'
                })

            # Incoming edges (what affects or points to this node)
            for source_id, _, edge_data in self.graph.in_edges(node_id, data=True):
                source_name = self.graph.nodes[source_id].get('name', source_id)
                rel_type = edge_data.get('type', 'RELATES_TO')
                props = edge_data.get('properties', {})
                prop_str = f' ({props})' if props else ''
                impact_summary.append({
                    'source': source_name,
                    'relation': rel_type,
                    'target': node_name,
                    'direction': 'incoming',
                    'detail': f'{source_name} --[{rel_type}]--> {node_name}{prop_str}'
                })

        return {
            'matched_entities': [self.graph.nodes[n].get('name', n) for n in matched_nodes],
            'impacts': impact_summary,
            'total_connections': len(impact_summary)
        }

    def find_paths(self, source_name: str, target_name: str) -> List[List[str]]:
        src_nodes = [n for n, d in self.graph.nodes(data=True) if source_name.lower() in d.get('name', '').lower()]
        tgt_nodes = [n for n, d in self.graph.nodes(data=True) if target_name.lower() in d.get('name', '').lower()]

        all_paths = []
        for s in src_nodes:
            for t in tgt_nodes:
                try:
                    for p in nx.all_simple_paths(self.graph.to_undirected(), source=s, target=t, cutoff=4):
                        named_path = [self.graph.nodes[node_id].get('name', node_id) for node_id in p]
                        all_paths.append(named_path)
                except Exception:
                    pass
        return all_paths

    def get_stats(self) -> Dict[str, Any]:
        type_counts = {}
        for _, data in self.graph.nodes(data=True):
            t = data.get('type', 'Unknown')
            type_counts[t] = type_counts.get(t, 0) + 1
            
        rel_counts = {}
        for _, _, data in self.graph.edges(data=True):
            r = data.get('type', 'UNKNOWN')
            rel_counts[r] = rel_counts.get(r, 0) + 1

        return {
            'total_nodes': self.graph.number_of_nodes(),
            'total_edges': self.graph.number_of_edges(),
            'entity_types': type_counts,
            'relation_types': rel_counts
        }

    def save(self):
        data = {
            'entities': [{'id': e.id, 'name': e.name, 'type': e.type, 'properties': e.properties} for e in self.entities.values()],
            'relations': [{'source_id': r.source_id, 'target_id': r.target_id, 'type': r.type, 'properties': r.properties} for r in self.relations]
        }
        kg_file = self.persist_dir / 'knowledge_graph.json'
        with open(kg_file, 'w', encoding='utf-8') as f:
            json.dump(data, f, indent=2)

    def load(self) -> bool:
        kg_file = self.persist_dir / 'knowledge_graph.json'
        if not kg_file.exists():
            return False
        try:
            with open(kg_file, 'r', encoding='utf-8') as f:
                data = json.load(f)
            entities = [Entity(**d) for d in data.get('entities', [])]
            relations = [Relation(**d) for d in data.get('relations', [])]
            self.build_from_extracted(entities, relations)
            return True
        except Exception as e:
            print(f'Error loading knowledge graph: {e}')
            return False
