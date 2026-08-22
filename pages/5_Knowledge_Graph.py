import streamlit as st
from nexusrag.app.ui.theme import get_custom_css
from nexusrag.app.ui.components import render_hero_banner, render_metric_card
from nexusrag.app.state import get_system_state
from nexusrag.backend.graph.visualizer import GraphVisualizer

st.markdown(get_custom_css(), unsafe_allow_html=True)
state = get_system_state()
kg = state.knowledge_graph
stats = kg.get_stats()

render_hero_banner("🕸️ Enterprise Knowledge Graph", "Interactive entity-relationship graph mapping policies, regulations, departments, and requirements with full provenance.")

m1, m2, m3, m4 = st.columns(4)
with m1:
    render_metric_card("Graph Entities", str(stats["total_entities"]), "Extracted Entity Nodes")
with m2:
    render_metric_card("Relationships", str(stats["total_relations"]), "Directed Dependency Edges")
with m3:
    render_metric_card("Network Nodes", str(stats["graph_nodes"]), "NetworkX Topology")
with m4:
    render_metric_card("Connected Edges", str(stats["graph_edges"]), "Cross-Entity Links")

st.markdown("---")

tab_graph, tab_entities, tab_relations = st.tabs(["🌐 Interactive Network Graph", "🏷️ Entities Directory", "🔗 Relationships Table"])

with tab_graph:
    st.markdown("### **Visual Knowledge Network**")
    st.caption("Hover over nodes to inspect entity types and source document provenance.")

    fig = GraphVisualizer.create_plotly_figure(kg)
    st.plotly_chart(fig, use_container_width=True)

with tab_entities:
    st.markdown("### **Extracted Entity Directory**")
    c_f1, c_f2 = st.columns(2)
    with c_f1:
        ent_types = ["All Types"] + sorted(list(set(e.entity_type for e in kg.entities.values())))
        sel_type = st.selectbox("Filter Entity Type:", ent_types)
    with c_f2:
        doc_names = ["All Documents"] + sorted(list(set(e.document_name for e in kg.entities.values())))
        sel_doc = st.selectbox("Filter by Document:", doc_names)

    filtered_ents = list(kg.entities.values())
    if sel_type != "All Types":
        filtered_ents = [e for e in filtered_ents if e.entity_type == sel_type]
    if sel_doc != "All Documents":
        filtered_ents = [e for e in filtered_ents if e.document_name == sel_doc]

    ent_rows = []
    for e in filtered_ents:
        ent_rows.append({
            "Entity Name": e.name,
            "Type": e.entity_type,
            "Document": e.document_name,
            "Page": e.page_number,
            "Section": e.section_title or "General",
            "Version": e.version,
            "Year": e.year
        })
    st.dataframe(ent_rows, use_container_width=True)

with tab_relations:
    st.markdown("### **Entity Relationship Triples**")
    rels = kg.list_relations()
    rel_rows = []
    for r in rels:
        rel_rows.append({
            "Source Entity": r.source_name,
            "Relationship": f"── [{r.relation_type}] ──▶",
            "Target Entity": r.target_name,
            "Document": r.document_name,
            "Section": r.section_title or "General",
            "Version": r.version
        })
    st.dataframe(rel_rows, use_container_width=True)
