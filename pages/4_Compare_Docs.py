import streamlit as st
from nexusrag.app.ui.theme import get_custom_css
from nexusrag.app.ui.components import render_hero_banner
from nexusrag.app.state import get_system_state

st.markdown(get_custom_css(), unsafe_allow_html=True)
state = get_system_state()
docs = state.list_documents()

render_hero_banner("⚖️ Document Version Comparison & Conflict Intelligence", "Compare document versions, detect policy modifications, additions, and removals with exact evidence traceability.")

if len(docs) < 2:
    st.info("At least two documents are required to perform a version comparison. Upload additional documents in the Documents page.")
else:
    doc_map = {d.document_id: f"{d.file_name} (v{d.version}, {d.year})" for d in docs}
    doc_ids = list(doc_map.keys())

    def_idx_a = 0
    def_idx_b = min(1, len(doc_ids) - 1)
    for idx, did in enumerate(doc_ids):
        if "2025" in doc_map[did]:
            def_idx_a = idx
        elif "2026" in doc_map[did]:
            def_idx_b = idx

    col_sel_a, col_sel_b = st.columns(2)
    with col_sel_a:
        doc_a_id = st.selectbox("Select Baseline Document (A):", doc_ids, index=def_idx_a, format_func=lambda x: doc_map[x], key="comp_doc_a")
    with col_sel_b:
        doc_b_id = st.selectbox("Select Updated Document (B):", doc_ids, index=def_idx_b, format_func=lambda x: doc_map[x], key="comp_doc_b")

    if doc_a_id == doc_b_id:
        st.warning("Please select two distinct documents to compare.")
    else:
        with st.spinner("Analyzing document versions and extracting clause deltas..."):
            comp_res = state.compare_documents(doc_a_id, doc_b_id)

        st.markdown("---")
        m1, m2, m3, m4, m5 = st.columns(5)
        with m1:
            st.metric("Total Sections", str(comp_res.total_sections))
        with m2:
            st.metric("Modified Clauses", str(comp_res.modified_count), delta=f"{comp_res.modified_count} changed", delta_color="inverse")
        with m3:
            st.metric("Added Requirements", str(comp_res.added_count), delta=f"+{comp_res.added_count} new", delta_color="normal")
        with m4:
            st.metric("Removed Clauses", str(comp_res.removed_count), delta=f"-{comp_res.removed_count} deleted" if comp_res.removed_count else "0", delta_color="inverse")
        with m5:
            st.metric("Unchanged Clauses", str(comp_res.unchanged_count))

        st.markdown(f"""
        <div class="chat-assistant" style="margin-top:12px; margin-bottom:20px; font-size:0.98rem; line-height:1.6;">
            <b>Comparison Overview:</b> {comp_res.summary}
        </div>
        """, unsafe_allow_html=True)

        st.markdown("### 📋 **Clause-by-Clause Policy Changes**")
        for change in comp_res.changes:
            status = change.status
            if status == "Modified":
                badge_html = "<span class='badge badge-medium'>Modified</span>"
            elif status == "Added":
                badge_html = "<span class='badge badge-high'>Added</span>"
            elif status == "Removed":
                badge_html = "<span class='badge badge-low'>Removed</span>"
            else:
                badge_html = "<span class='badge badge-intent'>Unchanged</span>"

            with st.expander(f"{status.upper()}: {change.topic.title()} (Similarity: {change.similarity_ratio:.2f})"):
                st.markdown(f"**Change Assessment:** {change.summary_of_change}")
                st.markdown(f"**Status:** {badge_html}", unsafe_allow_html=True)

                col_t1, col_t2 = st.columns(2)
                with col_t1:
                    st.markdown(f"**Document A ({comp_res.doc_a_name} - v{comp_res.doc_a_version}):**")
                    if change.doc_a_text:
                        st.code(change.doc_a_text, language="text")
                        st.caption(f"Source: {change.doc_a_source}")
                    else:
                        st.caption("*(Not present in Document A)*")
                with col_t2:
                    st.markdown(f"**Document B ({comp_res.doc_b_name} - v{comp_res.doc_b_version}):**")
                    if change.doc_b_text:
                        st.code(change.doc_b_text, language="text")
                        st.caption(f"Source: {change.doc_b_source}")
                    else:
                        st.caption("*(Removed in Document B)*")
