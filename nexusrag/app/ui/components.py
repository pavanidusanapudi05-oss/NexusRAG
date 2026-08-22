import streamlit as st
from typing import List, Dict, Any, Optional

def render_hero_banner(title: str, subtitle: str, badge_text: str = "Enterprise Knowledge Intelligence"):
    st.markdown(f"""
    <div class="nexus-hero">
        <div style="display:flex; justify-content:space-between; align-items:center;">
            <div>
                <h1>{title}</h1>
                <p>{subtitle}</p>
            </div>
            <div>
                <span class="badge badge-intent">{badge_text}</span>
            </div>
        </div>
    </div>
    """, unsafe_allow_html=True)

def render_evidence_cards(evidence_cards: List[Dict[str, Any]]):
    if not evidence_cards:
        return

    st.markdown("#### 📑 Verified Supporting Evidence")
    for card in evidence_cards:
        rank = card.get("rank", 1)
        doc = card.get("doc_name", "Unknown Document")
        ver = card.get("version", "1.0")
        year = card.get("year", "2026")
        page = card.get("page_number", 1)
        sec = card.get("section_title", "Section")
        score = card.get("relevance_score", 0.0)
        excerpt = card.get("excerpt", "")

        score_pct = int(score * 100)
        conf_class = "badge-high" if score >= 0.7 else ("badge-medium" if score >= 0.4 else "badge-low")

        st.markdown(f"""
        <div class="evidence-card">
            <div class="evidence-header">
                <div>
                    <span class="evidence-title">#{rank} {doc}</span>
                    <span class="badge badge-intent" style="margin-left:8px;">v{ver} ({year})</span>
                    <span class="badge" style="background:#334155; color:#cbd5e1; margin-left:4px;">Page {page}</span>
                </div>
                <div>
                    <span class="badge {conf_class}">Relevance: {score_pct}%</span>
                </div>
            </div>
            <div class="evidence-meta">
                <b>Section:</b> {sec}
            </div>
            <div class="evidence-snippet">
                "{excerpt}"
            </div>
        </div>
        """, unsafe_allow_html=True)

def render_metric_card(label: str, value: str, subtext: str = ""):
    st.markdown(f"""
    <div class="metric-box">
        <div class="metric-value">{value}</div>
        <div class="metric-label">{label}</div>
        <div style="font-size:0.75rem; color:#64748b; margin-top:4px;">{subtext}</div>
    </div>
    """, unsafe_allow_html=True)
