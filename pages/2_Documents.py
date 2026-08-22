import streamlit as st
from pathlib import Path
from nexusrag.app.ui.theme import get_custom_css
from nexusrag.app.ui.components import render_hero_banner
from nexusrag.app.state import get_system_state
from nexusrag.backend.ingestion.pipeline import ALLOWED_EXTENSIONS
from nexusrag.backend.models.document import ProcessingStatus

st.markdown(get_custom_css(), unsafe_allow_html=True)
state = get_system_state()

render_hero_banner("📁 Document Management & Search Hub", "Upload, inspect, chunk, index, search, and manage enterprise documents.")

tab1, tab2, tab3, tab4 = st.tabs([
    "📤 Upload Documents",
    "📋 Document Registry",
    "🔎 Dedicated Hybrid Search",
    "🧩 Chunk Explorer"
])

with tab1:
    st.markdown("### **Upload Supported Document**")
    st.caption(f"Supported Formats: {', '.join(sorted(list(ALLOWED_EXTENSIONS)))}")

    uploaded_files = st.file_uploader(
        "Choose file(s) to ingest & index",
        type=["pdf", "docx", "txt", "csv", "xlsx"],
        accept_multiple_files=True
    )

    with st.expander("⚙️ Optional Custom Metadata Overrides"):
        c1, c2, c3 = st.columns(3)
        with c1:
            custom_version = st.text_input("Document Version", value="1.0")
        with c2:
            custom_year = st.text_input("Effective Year", value="2026")
        with c3:
            custom_dept = st.text_input("Department / Authority", value="General")

    if uploaded_files and st.button("🚀 Process, Chunk & Index Files", type="primary"):
        for uf in uploaded_files:
            meta = {
                "version": custom_version,
                "year": custom_year,
                "department": custom_dept
            }
            with st.spinner(f"Ingesting, embedding & extracting graph for '{uf.name}'..."):
                res = state.process_uploaded_file(uf, custom_meta=meta)
                if res.success:
                    if res.is_duplicate:
                        st.warning(f"⚠️ **Duplicate Document:** {res.message}")
                    else:
                        st.success(f"✅ **Indexed:** {res.message}")
                else:
                    st.error(f"❌ **Error:** {res.message}")
        st.rerun()

with tab2:
    st.markdown("### **Document Registry & Storage**")
    docs = state.list_documents()

    if not docs:
        st.info("No documents found in the registry.")
    else:
        indexed_ids = state.vector_store.get_indexed_document_ids()
        for doc in docs:
            is_indexed = doc.document_id in indexed_ids
            status_color = "badge-high" if doc.processing_status == ProcessingStatus.PROCESSED else ("badge-low" if doc.processing_status == ProcessingStatus.FAILED else "badge-medium")
            
            with st.expander(f"📄 {doc.file_name} | {doc.file_type.upper()} | Status: {doc.processing_status.value} | Chunks: {doc.chunk_count} | Vectors: {'✅' if is_indexed else '⚠️'}"):
                col_d1, col_d2, col_d3 = st.columns([2, 2, 1])
                with col_d1:
                    st.markdown(f"**Document ID:** `{doc.document_id}`")
                    st.markdown(f"**File Size:** `{doc.file_size_bytes:,} bytes`")
                    st.markdown(f"**SHA-256 Hash:** `{doc.file_hash[:16]}...`")
                with col_d2:
                    st.markdown(f"**Version:** `{doc.version}` | **Year:** `{doc.year}`")
                    st.markdown(f"**Department:** `{doc.department}`")
                    st.markdown(f"**Vector Store:** {'`Indexed`' if is_indexed else '`Not Indexed`'}")
                with col_d3:
                    st.markdown(f"""<span class="badge {status_color}">{doc.processing_status.value}</span>""", unsafe_allow_html=True)
                    st.markdown(f"**Pages:** `{doc.total_pages}`")

                if doc.error_message:
                    st.error(f"**Error Details:** {doc.error_message}")

                st.markdown("---")
                col_btn1, col_btn2 = st.columns([1, 1])
                with col_btn1:
                    if st.button(f"🔄 Re-Index Document", key=f"reidx_{doc.document_id}"):
                        count = state.reindex_document(doc.document_id)
                        st.success(f"Re-indexed {count} vectors & graph entities for '{doc.file_name}'.")
                        st.rerun()
                with col_btn2:
                    if st.button(f"🗑️ Delete '{doc.file_name}'", key=f"del_{doc.document_id}", type="secondary"):
                        success = state.delete_document(doc.document_id)
                        if success:
                            st.success(f"Deleted document '{doc.file_name}', chunks, vectors, and graph records.")
                            st.rerun()
                        else:
                            st.error("Failed to delete document.")

with tab3:
    st.markdown("### 🔎 **Dedicated Hybrid Retrieval Experience**")
    st.caption("Search across documents using combined Dense Vector Cosine Similarity and Lexical BM25 Keyword Search with Precision Reranking.")

    c_s1, c_s2, c_s3, c_s4 = st.columns([3, 1, 1, 1])
    with c_s1:
        search_query = st.text_input("Search query (keywords, natural language, or codes):", placeholder="e.g. SR-402 MFA encryption or attendance 60%", key="hybrid_search_box")
    with c_s2:
        search_topk = st.selectbox("Top-K", [3, 5, 10, 20], index=1, key="search_topk_select")
    with c_s3:
        all_docs = ["All Documents"] + [d.file_name for d in state.list_documents()]
        filter_doc = st.selectbox("Filter Document", all_docs)
    with c_s4:
        filter_year = st.selectbox("Filter Year", ["All Years", "2026", "2025", "2024"])

    if search_query.strip():
        doc_f = None if filter_doc == "All Documents" else filter_doc
        yr_f = None if filter_year == "All Years" else filter_year

        with st.spinner("Executing hybrid retrieval and reranking..."):
            search_results = state.search_hybrid(
                query=search_query.strip(),
                top_k=search_topk,
                doc_filter=doc_f,
                year_filter=yr_f
            )

        st.markdown(f"**Found {len(search_results)} relevant result(s):**")
        for i, res in enumerate(search_results):
            loc = f"Page {res.page_number}" if res.page_number else (f"Sheet: {res.sheet_name}" if res.sheet_name else "")
            sec = f"Section: {res.section_title}" if res.section_title else ""
            
            with st.expander(f"[{i+1}] {res.document_name} | {loc} | {sec} | Score: {res.similarity_score:.4f}"):
                st.markdown(f"**Metadata:** Document: `{res.document_name}` | Version: `{res.version}` | Year: `{res.year}` | Dept: `{res.department}`")
                st.code(res.text, language="text")

with tab4:
    st.markdown("### **Traceable Chunk Explorer**")
    docs = state.list_documents()
    doc_options = {d.document_id: f"{d.file_name} ({d.chunk_count} chunks)" for d in docs if d.processing_status == ProcessingStatus.PROCESSED}

    if not doc_options:
        st.info("No processed documents available to inspect.")
    else:
        selected_id = st.selectbox("Select Document to Inspect Chunks:", list(doc_options.keys()), format_func=lambda x: doc_options[x])
        chunks = state.get_document_chunks(selected_id)
        
        st.caption(f"Showing {len(chunks)} chunks for selected document.")

        for c in chunks:
            meta = c.metadata
            page_info = f"Page {meta.get('page_number', 1)}" if meta.get('page_number') else ""
            sheet_info = f"Sheet: {meta.get('sheet_name')}" if meta.get('sheet_name') else ""
            loc = " | ".join(filter(None, [page_info, sheet_info, meta.get("section_title")]))

            with st.expander(f"🧩 Chunk Index {c.chunk_index} | ID: `{c.chunk_id}` | {loc}"):
                st.markdown(f"**Section Title:** `{meta.get('section_title', 'N/A')}`")
                st.markdown(f"**Metadata:** Version: `{meta.get('version')}` | Year: `{meta.get('year')}` | Dept: `{meta.get('department')}` | Tokens: `{c.token_count}` | Characters: `{c.char_count}`")
                st.code(c.text, language="text")
