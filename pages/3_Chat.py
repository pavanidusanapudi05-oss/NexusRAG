import streamlit as st
from nexusrag.app.ui.theme import get_custom_css
from nexusrag.app.ui.components import render_hero_banner
from nexusrag.app.state import get_system_state
from nexusrag.config.settings import settings

st.markdown(get_custom_css(), unsafe_allow_html=True)
state = get_system_state()

render_hero_banner("💬 Evidence-First Knowledge Intelligence Chat", "Ask questions over your ingested documents with strict evidence grounding, verified citations, and confidence transparency.")

# Controls & Quick Action Header
col_hdr_l, col_hdr_r = st.columns([3, 1])
with col_hdr_l:
    st.caption("NexusRAG Phase 3: Zero-Hallucination Grounded Answering Engine")
with col_hdr_r:
    if st.button("🗑️ Clear Conversation", use_container_width=True):
        state.clear_chat_history()
        st.rerun()

# Sample Prompts Bar
st.markdown("##### **💡 Try a sample prompt:**")
c_p1, c_p2, c_p3 = st.columns(3)
sample_prompt = None
with c_p1:
    if st.button("🏢 Remote Work Policy 2026", use_container_width=True):
        sample_prompt = "What are the remote work policy rules and weekly allowances for 2026?"
with c_p2:
    if st.button("⚖️ 2025 vs 2026 Policy Changes", use_container_width=True):
        sample_prompt = "Compare the changes and differences between 2025 and 2026 attendance policy rules."
with c_p3:
    if st.button("🔒 Cybersecurity & MFA Rules", use_container_width=True):
        sample_prompt = "What are the multi-factor authentication (MFA) and encryption requirements in Regulation SR-402?"

# Query Input Box
with st.form("chat_query_form", clear_on_submit=False):
    query_text = st.text_input(
        "Ask a question about your uploaded documents:",
        value=sample_prompt if sample_prompt else "",
        placeholder="e.g. What is the mandatory attendance requirement in the 2026 policy?",
        key="phase3_user_query"
    )
    c_btn1, c_btn2 = st.columns([4, 1])
    with c_btn1:
        submit_btn = st.form_submit_button("🔍 Ask Question", type="primary", use_container_width=True)
    with c_btn2:
        top_k_select = st.selectbox("Top-K Chunks", [3, 5, 8], index=1)

if (submit_btn or sample_prompt) and query_text.strip():
    with st.spinner("Searching vector index, retrieving evidence, and generating grounded answer..."):
        ans = state.query_rag(query_text.strip(), top_k=top_k_select)
    st.rerun()

# Render Session Chat History
if not state.chat_history:
    st.info("No queries asked yet in this session. Choose a sample prompt above or enter a question to begin.")
else:
    st.markdown("---")
    st.markdown("### 📜 **Conversation History**")

    for turn_idx, item in enumerate(reversed(state.chat_history)):
        turn_num = len(state.chat_history) - turn_idx

        # 1. User Query Bubble
        st.markdown(f"""
        <div class="chat-user" style="margin-top:20px;">
            <div style="font-size:0.78rem; color:#94a3b8; margin-bottom:4px;"><b>User Query #{turn_num}</b></div>
            <div style="font-size:1.05rem; font-weight:500;">{item.query}</div>
        </div>
        """, unsafe_allow_html=True)

        # 2. Assistant Grounded Answer Container
        with st.container():
            # Confidence & Conflict Badges Header
            conf = item.confidence
            conf_color = "#10b981" if conf.level == "High" else ("#f59e0b" if conf.level == "Medium" else "#ef4444")
            
            c_meta1, c_meta2, c_meta3 = st.columns([2, 1, 1])
            with c_meta1:
                st.markdown(f"**Confidence:** <span style='color:{conf_color}; font-weight:700;'>{conf.level} ({conf.score_percentage}%)</span> — *{conf.explanation}*", unsafe_allow_html=True)
            with c_meta2:
                st.caption(f"Engine: `{item.llm_provider}`")
            with c_meta3:
                if item.has_conflict:
                    st.markdown("<span class='badge badge-medium'>⚠️ Version Differences</span>", unsafe_allow_html=True)

            # Grounded Answer Content
            if item.is_abstention:
                st.warning(f"⚠️ **Insufficient Evidence:** {item.answer}")
            else:
                st.markdown(f"""
                <div class="chat-assistant" style="margin-top:8px; line-height:1.65; font-size:1.02rem;">
                    {item.answer}
                </div>
                """, unsafe_allow_html=True)

            # Citations Section
            if item.citations:
                st.markdown("##### 📚 **Source Citations**")
                for c in item.citations:
                    st.markdown(f"- `{c.citation_label}` *(Similarity: {c.similarity_score:.4f})*")

            # Evidence Cards Drawer
            if item.evidence:
                with st.expander(f"🔍 Inspect Retrieved Evidence ({len(item.evidence)} Sources)", expanded=(turn_idx == 0 and not item.is_abstention)):
                    for ev_idx, ev in enumerate(item.evidence):
                        src_num = ev_idx + 1
                        loc = f"Page {ev.page_number}" if ev.page_number else (f"Sheet: {ev.sheet_name}" if ev.sheet_name else "")
                        sec = f"Section: {ev.section_title}" if ev.section_title else ""
                        
                        st.markdown(f"**[Source {src_num}] {ev.document_name}** | {loc} | {sec} | `Score: {ev.similarity_score:.4f}` | `v{ev.version}` ({ev.year})")
                        st.code(ev.text, language="text")
                        st.markdown("---")
