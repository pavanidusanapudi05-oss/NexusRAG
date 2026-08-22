import streamlit as st
from nexusrag.app.ui.theme import get_custom_css
from nexusrag.app.ui.components import render_hero_banner
from nexusrag.app.state import get_system_state
from nexusrag.config.settings import settings

st.markdown(get_custom_css(), unsafe_allow_html=True)
state = get_system_state()

render_hero_banner("⚙️ System & Platform Settings", "Configure retrieval weights, precision reranking, chunk hyperparameters, and model configurations.")

with st.form("complete_settings_form"):
    st.markdown("### 🔎 **Hybrid Retrieval & Reranking Parameters**")
    c1, c2 = st.columns(2)
    with c1:
        sem_weight = st.slider("Dense Semantic Weight (alpha)", min_value=0.0, max_value=1.0, value=settings.semantic_weight, step=0.05)
        top_k = st.number_input("Default Top-K Chunks", min_value=1, max_value=20, value=settings.retrieval_top_k)
    with c2:
        kw_weight = st.slider("Lexical BM25 Keyword Weight (1 - alpha)", min_value=0.0, max_value=1.0, value=settings.keyword_weight, step=0.05)
        rerank_enable = st.checkbox("Enable Precision Reranker", value=settings.reranker_enabled)

    st.markdown("---")
    st.markdown("### 🤖 **LLM & Embedding Configuration**")
    c3, c4 = st.columns(2)
    with c3:
        llm_provider = st.selectbox(
            "LLM Provider",
            ["offline", "gemini", "openai"],
            index=0 if settings.llm_provider == "offline" else (1 if settings.llm_provider == "gemini" else 2)
        )
        gemini_model = st.text_input("Gemini Model", value=settings.gemini_model)
    with c4:
        emb_provider = st.selectbox("Embedding Provider", ["local_dense", "gemini", "openai"], index=0)
        openai_model = st.text_input("OpenAI Model", value=settings.openai_model)

    st.markdown("---")
    st.markdown("### 🧩 **Chunking Hyperparameters**")
    c5, c6 = st.columns(2)
    with c5:
        chunk_size = st.number_input("Chunk Size (Characters)", min_value=200, max_value=2000, value=settings.chunk_size, step=50)
    with c6:
        chunk_overlap = st.number_input("Chunk Overlap (Characters)", min_value=0, max_value=500, value=settings.chunk_overlap, step=20)

    st.markdown("---")
    st.markdown("### 🗄️ **Storage & Registry Paths**")
    st.text_input("SQLite Database Location", value=str(settings.data_dir / "nexusrag.db"), disabled=True)
    st.text_input("Vector Database Location", value=str(settings.data_dir / "vector_store"), disabled=True)
    st.text_input("Knowledge Graph Location", value=str(settings.data_dir / "graph" / "knowledge_graph.json"), disabled=True)

    submitted = st.form_submit_button("💾 Save Platform Settings", type="primary")
    if submitted:
        settings.semantic_weight = sem_weight
        settings.keyword_weight = kw_weight
        settings.reranker_enabled = rerank_enable
        settings.retrieval_top_k = top_k
        settings.llm_provider = llm_provider
        settings.chunk_size = chunk_size
        settings.chunk_overlap = chunk_overlap

        state.hybrid_searcher.semantic_weight = sem_weight
        state.hybrid_searcher.keyword_weight = kw_weight
        state.hybrid_searcher.reranker.enabled = rerank_enable
        state.retriever.top_k = top_k
        state.rag_pipeline.retriever.top_k = top_k
        st.success("Platform settings updated successfully!")
