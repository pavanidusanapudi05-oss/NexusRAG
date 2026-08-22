import streamlit as st
import plotly.express as px
from nexusrag.app.ui.theme import get_custom_css
from nexusrag.app.ui.components import render_hero_banner, render_metric_card
from nexusrag.app.state import get_system_state

st.markdown(get_custom_css(), unsafe_allow_html=True)
state = get_system_state()

render_hero_banner("📊 RAG Evaluation & Quality Monitoring Suite", "Automated benchmarking for Retrieval Precision@K, Recall@K, Faithfulness, Relevance, and Citation Correctness.")

col_btn, col_info = st.columns([1, 3])
with col_btn:
    run_eval_btn = st.button("🚀 Run Evaluation Benchmark", type="primary", use_container_width=True)
with col_info:
    st.caption("Executes standard enterprise QA test cases across ingested policy, regulation, and spreadsheet documents.")

if run_eval_btn or state.latest_eval_report is None:
    with st.spinner("Running automated RAG benchmark test suite..."):
        report = state.run_evaluation()
else:
    report = state.latest_eval_report

metrics = report.metrics
m1, m2, m3, m4, m5, m6 = st.columns(6)
with m1:
    render_metric_card("Overall Score", f"{metrics.overall_score:.1f}%", f"{report.passed_tests}/{report.total_tests} Tests Passed")
with m2:
    render_metric_card("Precision@K", f"{metrics.precision_at_k * 100:.1f}%", "Retrieval Precision")
with m3:
    render_metric_card("Recall@K", f"{metrics.recall_at_k * 100:.1f}%", "Evidence Recall")
with m4:
    render_metric_card("Faithfulness", f"{metrics.faithfulness_score * 100:.1f}%", "Grounded Accuracy")
with m5:
    render_metric_card("Relevance", f"{metrics.relevance_score * 100:.1f}%", "Answer Relevance")
with m6:
    render_metric_card("Citation Acc", f"{metrics.citation_accuracy * 100:.1f}%", "Verified Sources")

st.markdown("---")

st.markdown("### 📋 **Benchmark Test Case Results**")
test_rows = []
for res in report.results:
    test_rows.append({
        "Test ID": res.test_id,
        "Query": res.query,
        "Category": res.category,
        "Status": "✅ PASS" if res.passed else "❌ FAIL",
        "Precision": f"{res.precision:.2f}",
        "Faithfulness": f"{res.faithfulness:.2f}",
        "Citations": res.citations_count,
        "Notes": res.notes
    })

st.dataframe(test_rows, use_container_width=True)

st.markdown("### 🔍 **Detailed Test Case Inspection**")
for res in report.results:
    status_badge = "<span class='badge badge-high'>PASS</span>" if res.passed else "<span class='badge badge-low'>FAIL</span>"
    with st.expander(f"{res.test_id}: {res.query} ({'PASS' if res.passed else 'FAIL'})"):
        st.markdown(f"**Category:** `{res.category}` | **Status:** {status_badge}", unsafe_allow_html=True)
        st.markdown(f"**Answer Generated:**\n> {res.answer_preview}")
        st.markdown(f"**Evaluation Notes:** {res.notes}")
