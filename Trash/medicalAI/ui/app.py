"""
MedDecide — Streamlit UI
Step 8: Beautiful AI-powered medical decision support interface
"""

import sys
import json
import time
import logging
from pathlib import Path

import streamlit as st
import plotly.graph_objects as go
import plotly.express as px
import numpy as np
import pandas as pd

# Add project root to path
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

logging.basicConfig(level=logging.WARNING)
logger = logging.getLogger(__name__)

# ── Page Config ───────────────────────────────────────────────────────────────

st.set_page_config(
    page_title="MedDecide — AI Medical Decision Support",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ── Custom CSS ────────────────────────────────────────────────────────────────

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Space+Grotesk:wght@300;400;500;600;700&family=JetBrains+Mono:wght@400;600&display=swap');

/* Base */
html, body, [class*="css"] {
    font-family: 'Space Grotesk', sans-serif;
}

/* Background */
.stApp {
    background: linear-gradient(135deg, #0a0e1a 0%, #0d1424 50%, #0a1020 100%);
    color: #e8eaf6;
}

/* Header */
.main-header {
    background: linear-gradient(135deg, rgba(99,102,241,0.15), rgba(168,85,247,0.1));
    border: 1px solid rgba(99,102,241,0.3);
    border-radius: 20px;
    padding: 2rem 2.5rem;
    margin-bottom: 2rem;
    backdrop-filter: blur(10px);
}
.main-title {
    font-size: 2.8rem;
    font-weight: 700;
    background: linear-gradient(135deg, #818cf8, #c084fc, #38bdf8);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    background-clip: text;
    margin: 0;
    letter-spacing: -1px;
}
.main-subtitle {
    color: #94a3b8;
    font-size: 1rem;
    font-weight: 400;
    margin-top: 0.4rem;
}

/* Cards */
.card {
    background: rgba(255,255,255,0.04);
    border: 1px solid rgba(255,255,255,0.08);
    border-radius: 16px;
    padding: 1.5rem;
    margin-bottom: 1rem;
    backdrop-filter: blur(5px);
    transition: all 0.3s ease;
}
.card:hover {
    border-color: rgba(99,102,241,0.4);
    background: rgba(255,255,255,0.06);
}

/* Action cards */
.action-rest {
    border-left: 4px solid #34d399;
    background: rgba(52,211,153,0.05);
}
.action-otc {
    border-left: 4px solid #60a5fa;
    background: rgba(96,165,250,0.05);
}
.action-gp {
    border-left: 4px solid #fbbf24;
    background: rgba(251,191,36,0.05);
}
.action-er {
    border-left: 4px solid #f87171;
    background: rgba(248,113,113,0.08);
    animation: pulse-er 2s infinite;
}
@keyframes pulse-er {
    0%, 100% { border-left-color: #f87171; }
    50% { border-left-color: #fca5a5; }
}

/* Symptom tags */
.symptom-tag {
    display: inline-block;
    background: rgba(99,102,241,0.2);
    border: 1px solid rgba(99,102,241,0.4);
    border-radius: 20px;
    padding: 3px 12px;
    margin: 3px;
    font-size: 0.82rem;
    color: #a5b4fc;
    font-family: 'JetBrains Mono', monospace;
}
.symptom-tag-soft {
    background: rgba(168,85,247,0.15);
    border-color: rgba(168,85,247,0.35);
    color: #d8b4fe;
}

/* Step badge */
.step-badge {
    display: inline-flex;
    align-items: center;
    justify-content: center;
    width: 28px;
    height: 28px;
    background: linear-gradient(135deg, #6366f1, #8b5cf6);
    border-radius: 50%;
    font-size: 0.75rem;
    font-weight: 700;
    color: white;
    margin-right: 8px;
}

/* Metric cards */
.metric-box {
    background: rgba(255,255,255,0.03);
    border: 1px solid rgba(255,255,255,0.07);
    border-radius: 12px;
    padding: 1rem;
    text-align: center;
}
.metric-value {
    font-size: 1.8rem;
    font-weight: 700;
    color: #818cf8;
    font-family: 'JetBrains Mono', monospace;
}
.metric-label {
    font-size: 0.75rem;
    color: #64748b;
    text-transform: uppercase;
    letter-spacing: 1px;
    margin-top: 4px;
}

/* Confidence badges */
.conf-high   { color: #34d399; font-weight: 600; }
.conf-mod    { color: #fbbf24; font-weight: 600; }
.conf-low    { color: #f97316; font-weight: 600; }
.conf-vlow   { color: #f87171; font-weight: 600; }

/* Warning box */
.warning-box {
    background: rgba(251,191,36,0.08);
    border: 1px solid rgba(251,191,36,0.3);
    border-radius: 12px;
    padding: 1rem 1.2rem;
    margin: 0.5rem 0;
    color: #fde68a;
}
.danger-box {
    background: rgba(248,113,113,0.08);
    border: 1px solid rgba(248,113,113,0.35);
    border-radius: 12px;
    padding: 1rem 1.2rem;
    margin: 0.5rem 0;
    color: #fca5a5;
}

/* Log panel */
.log-panel {
    background: #070d1a;
    border: 1px solid rgba(99,102,241,0.2);
    border-radius: 12px;
    padding: 1rem 1.2rem;
    font-family: 'JetBrains Mono', monospace;
    font-size: 0.78rem;
    color: #64748b;
    max-height: 400px;
    overflow-y: auto;
}
.log-step {
    color: #6366f1;
    font-weight: 600;
}
.log-detail {
    color: #475569;
    margin-left: 1rem;
}

/* Input area */
.stTextArea textarea {
    background: rgba(255,255,255,0.04) !important;
    border: 1px solid rgba(99,102,241,0.3) !important;
    border-radius: 12px !important;
    color: #e2e8f0 !important;
    font-family: 'Space Grotesk', sans-serif !important;
    font-size: 1rem !important;
}
.stTextArea textarea:focus {
    border-color: rgba(99,102,241,0.7) !important;
    box-shadow: 0 0 0 2px rgba(99,102,241,0.15) !important;
}

/* Button */
.stButton > button {
    background: linear-gradient(135deg, #6366f1, #8b5cf6) !important;
    color: white !important;
    border: none !important;
    border-radius: 12px !important;
    padding: 0.7rem 2.5rem !important;
    font-family: 'Space Grotesk', sans-serif !important;
    font-weight: 600 !important;
    font-size: 1rem !important;
    letter-spacing: 0.3px !important;
    transition: all 0.3s ease !important;
    width: 100%;
}
.stButton > button:hover {
    transform: translateY(-2px) !important;
    box-shadow: 0 8px 20px rgba(99,102,241,0.4) !important;
}

/* Sidebar */
[data-testid="stSidebar"] {
    background: rgba(7,13,26,0.95) !important;
    border-right: 1px solid rgba(99,102,241,0.15) !important;
}

/* Divider */
hr {
    border-color: rgba(255,255,255,0.06) !important;
}

/* Select box */
.stSelectbox > div > div {
    background: rgba(255,255,255,0.04) !important;
    border: 1px solid rgba(99,102,241,0.25) !important;
    border-radius: 10px !important;
    color: #e2e8f0 !important;
}

/* Expander */
.streamlit-expanderHeader {
    background: rgba(255,255,255,0.03) !important;
    border-radius: 10px !important;
    color: #94a3b8 !important;
}

/* Scrollbar */
::-webkit-scrollbar { width: 6px; height: 6px; }
::-webkit-scrollbar-track { background: rgba(255,255,255,0.02); }
::-webkit-scrollbar-thumb { background: rgba(99,102,241,0.4); border-radius: 3px; }
</style>
""", unsafe_allow_html=True)


# ── Helper Functions ──────────────────────────────────────────────────────────

@st.cache_resource(show_spinner="Initializing AI models...")
def load_pipeline():
    """Load and cache the pipeline."""
    from src.pipeline import MedDecidePipeline
    return MedDecidePipeline(use_llm=False)  # Set use_llm=True if ANTHROPIC_API_KEY set


def confidence_color_class(label: str) -> str:
    return {"High": "conf-high", "Moderate": "conf-mod",
            "Low": "conf-low", "Very Low": "conf-vlow"}.get(label, "conf-low")


def action_card_class(action: str) -> str:
    return {"rest": "action-rest", "otc": "action-otc",
            "see_gp": "action-gp", "visit_er": "action-er"}.get(action, "action-gp")


def create_disease_chart(top_diseases):
    """Create a horizontal bar chart of disease probabilities."""
    diseases = [d for d, _ in top_diseases[:8]]
    probs = [p * 100 for _, p in top_diseases[:8]]
    colors = px.colors.sequential.Plasma_r[:len(diseases)]

    fig = go.Figure(go.Bar(
        x=probs,
        y=diseases,
        orientation='h',
        marker=dict(
            color=probs,
            colorscale=[[0, '#4338ca'], [0.5, '#7c3aed'], [1, '#c084fc']],
            showscale=False,
        ),
        text=[f"{p:.1f}%" for p in probs],
        textposition='outside',
        textfont=dict(color='#94a3b8', size=11)
    ))
    fig.update_layout(
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='rgba(0,0,0,0)',
        font=dict(color='#94a3b8', family='Space Grotesk'),
        height=350,
        margin=dict(l=0, r=60, t=10, b=10),
        xaxis=dict(
            showgrid=True, gridcolor='rgba(255,255,255,0.05)',
            showline=False, ticksuffix='%', color='#475569',
            range=[0, max(probs) * 1.2]
        ),
        yaxis=dict(showgrid=False, color='#94a3b8', autorange='reversed'),
    )
    return fig


def create_eu_chart(decision_result):
    """Create EU comparison radar/bar chart."""
    from src.decision.utility_decision import ACTION_LABELS
    actions = list(decision_result.expected_utilities.keys())
    eu_values = [decision_result.expected_utilities[a] for a in actions]
    labels = [ACTION_LABELS[a].split(' ', 1)[1] for a in actions]  # Remove emoji

    colors = ['#34d399' if a == decision_result.recommended_action else '#4338ca' for a in actions]

    fig = go.Figure(go.Bar(
        x=labels,
        y=eu_values,
        marker_color=colors,
        text=[f"{v:.1f}" for v in eu_values],
        textposition='outside',
        textfont=dict(color='#94a3b8', size=11)
    ))
    fig.update_layout(
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='rgba(0,0,0,0)',
        font=dict(color='#94a3b8', family='Space Grotesk'),
        height=260,
        margin=dict(l=0, r=20, t=10, b=10),
        xaxis=dict(showgrid=False, color='#94a3b8'),
        yaxis=dict(showgrid=True, gridcolor='rgba(255,255,255,0.05)',
                   color='#475569', title='Expected Utility'),
    )
    return fig


def create_entropy_gauge(entropy, max_entropy=5.0):
    """Create entropy gauge."""
    pct = min(entropy / max_entropy * 100, 100)
    color = "#34d399" if pct < 40 else "#fbbf24" if pct < 70 else "#f87171"

    fig = go.Figure(go.Indicator(
        mode="gauge+number",
        value=entropy,
        number=dict(font=dict(color=color, size=32, family='JetBrains Mono'),
                    suffix=" bits"),
        gauge=dict(
            axis=dict(range=[0, max_entropy], tickcolor='#475569',
                      tickfont=dict(color='#475569')),
            bar=dict(color=color, thickness=0.25),
            bgcolor='rgba(0,0,0,0)',
            bordercolor='rgba(0,0,0,0)',
            steps=[
                dict(range=[0, 2], color='rgba(52,211,153,0.1)'),
                dict(range=[2, 3.5], color='rgba(251,191,36,0.1)'),
                dict(range=[3.5, 5], color='rgba(248,113,113,0.1)'),
            ]
        ),
        title=dict(text="Entropy (H)", font=dict(color='#64748b', size=12))
    ))
    fig.update_layout(
        paper_bgcolor='rgba(0,0,0,0)',
        height=200,
        margin=dict(l=20, r=20, t=30, b=10)
    )
    return fig


SAMPLE_QUERIES = [
    "I have had a high fever, severe headache, chills, and my whole body aches for the past 2 days.",
    "I feel chest pain, short of breath, and my left arm feels numb.",
    "I have nausea, diarrhea, stomach cramps, and lost my appetite since yesterday.",
    "Sudden terrible headache, stiff neck, and I'm sensitive to light.",
    "I've been very tired, have a dry cough, and lost my sense of smell and taste.",
    "My joints are very painful and swollen, especially in the morning.",
]


# ── Sidebar ───────────────────────────────────────────────────────────────────

with st.sidebar:
    st.markdown("""
    <div style="text-align:center; padding: 1rem 0;">
        <div style="font-size:3rem">🧠</div>
        <div style="font-size:1.3rem; font-weight:700; color:#818cf8;">MedDecide</div>
        <div style="font-size:0.75rem; color:#475569; margin-top:4px;">AI Medical Decision Support</div>
    </div>
    """, unsafe_allow_html=True)

    st.divider()

    st.markdown("**⚡ Quick Examples**")
    for i, sample in enumerate(SAMPLE_QUERIES, 1):
        if st.button(f"📋 Example {i}", key=f"sample_{i}", use_container_width=True):
            st.session_state["input_text"] = sample

    st.divider()

    st.markdown("**⚙️ Settings**")
    use_soft_evidence = st.toggle("Soft Evidence Mode", value=True,
                                   help="Use probabilistic soft evidence vs binary hard evidence")
    show_log = st.toggle("Show Inference Log", value=True,
                          help="Display step-by-step reasoning trace")
    top_k = st.slider("Top-K Diseases", min_value=3, max_value=10, value=6)

    st.divider()

    st.markdown("""
    <div style="font-size:0.7rem; color:#334155; line-height:1.6;">
    <b style="color:#475569">Pipeline Steps</b><br>
    1. BioBERT NER Extraction<br>
    2. Canonical Normalization<br>
    3. Soft Evidence Creation<br>
    4. Bayesian Network (pgmpy)<br>
    5. Entropy / Novelty Detector<br>
    6. Utility Decision (EU)<br>
    7. Explanation Generator<br>
    8. Streamlit UI (you are here)
    </div>
    """, unsafe_allow_html=True)

    st.divider()
    st.markdown("""
    <div style="font-size:0.7rem; color:#334155; text-align:center;">
    ⚠️ For educational purposes only.<br>
    Not a substitute for medical advice.
    </div>
    """, unsafe_allow_html=True)


# ── Main Header ───────────────────────────────────────────────────────────────

st.markdown("""
<div class="main-header">
    <h1 class="main-title">🧠 MedDecide</h1>
    <p class="main-subtitle">
        Probabilistic AI Medical Decision Support &nbsp;·&nbsp;
        BioBERT + Bayesian Network + Expected Utility Theory
    </p>
</div>
""", unsafe_allow_html=True)


# ── Input Section ─────────────────────────────────────────────────────────────

st.markdown("### 💬 Describe Your Symptoms")

default_text = st.session_state.get("input_text", "")
user_input = st.text_area(
    label="",
    value=default_text,
    height=120,
    placeholder="Describe your symptoms in natural language...\n\nExample: I have had a high fever, severe headache, chills, and body aches for 2 days.",
    key="symptom_input"
)

col_btn, col_clear = st.columns([4, 1])
with col_btn:
    analyze = st.button("🔍 Analyze Symptoms", use_container_width=True)
with col_clear:
    if st.button("✕ Clear", use_container_width=True):
        st.session_state["input_text"] = ""
        st.rerun()


# ── Analysis ──────────────────────────────────────────────────────────────────

if analyze and user_input.strip():
    pipeline = load_pipeline()

    with st.spinner("🔬 Running probabilistic analysis pipeline..."):
        progress_bar = st.progress(0)
        status_text = st.empty()

        steps = [
            (15, "Step 1: Extracting symptoms with BioBERT NER..."),
            (30, "Step 2: Normalizing to canonical vocabulary..."),
            (45, "Step 3: Creating soft evidence..."),
            (60, "Step 4: Running Bayesian Network inference..."),
            (75, "Step 5: Analyzing uncertainty & novelty..."),
            (88, "Step 6: Computing expected utility..."),
            (95, "Step 7: Generating explanation..."),
        ]

        for pct, msg in steps:
            progress_bar.progress(pct)
            status_text.markdown(f"<div style='color:#6366f1;font-size:0.85rem'>{msg}</div>",
                                  unsafe_allow_html=True)
            time.sleep(0.1)

        result = pipeline.run(user_input)
        progress_bar.progress(100)
        status_text.empty()
        progress_bar.empty()

    if not result.success:
        st.error(f"Pipeline error: {result.error}")
        st.stop()

    st.markdown("---")

    # ── Results Layout ────────────────────────────────────────────────────────

    # WARNINGS
    if result.warnings:
        for w in result.warnings:
            if "⚠️" in w and "High-risk" in w:
                st.markdown(f'<div class="danger-box">🚨 {w}</div>', unsafe_allow_html=True)
            elif w:
                st.markdown(f'<div class="warning-box">{w}</div>', unsafe_allow_html=True)

    # TOP ROW: Extracted Symptoms + Recommended Action
    col_left, col_right = st.columns([1, 1])

    with col_left:
        st.markdown("#### <span class='step-badge'>2</span> Extracted & Normalized Symptoms",
                    unsafe_allow_html=True)
        st.markdown('<div class="card">', unsafe_allow_html=True)

        if result.normalization and result.normalization.normalized:
            tags_html = ""
            for sym in result.normalization.normalized:
                method = sym.match_method
                cls = "symptom-tag-soft" if method in ("semantic",) else "symptom-tag"
                display = sym.canonical.replace("_", " ").title()
                conf = f"{sym.confidence:.0%}"
                tags_html += f'<span class="{cls}" title="Match: {method} | Conf: {conf}">{display}</span>'
            st.markdown(tags_html, unsafe_allow_html=True)

            if result.normalization.unmatched:
                st.markdown(
                    f"<small style='color:#475569'>Unmatched: "
                    f"{', '.join(result.normalization.unmatched)}</small>",
                    unsafe_allow_html=True
                )
        else:
            st.markdown("<span style='color:#475569'>No canonical symptoms extracted.</span>",
                        unsafe_allow_html=True)

        # Extraction method indicator
        if result.extraction:
            method_txt = "✓ BioBERT NER" if not result.extraction.fallback_used else "⚡ Rule-based (fast)"
            st.markdown(f"<small style='color:#334155'>Extractor: {method_txt}</small>",
                        unsafe_allow_html=True)

        st.markdown('</div>', unsafe_allow_html=True)

    with col_right:
        st.markdown("#### <span class='step-badge'>6</span> Recommended Action",
                    unsafe_allow_html=True)
        action_class = action_card_class(result.recommended_action)
        conf_class = confidence_color_class(result.confidence_label)

        st.markdown(f"""
        <div class="card {action_class}">
            <div style="font-size:1.6rem; font-weight:700; color:#e2e8f0; margin-bottom:0.4rem">
                {result.recommended_label}
            </div>
            <div style="color:#94a3b8; font-size:0.9rem; margin-bottom:0.8rem">
                {result.decision_result.recommended_description}
            </div>
            <div style="display:flex; gap:1.5rem; margin-top:0.5rem">
                <div>
                    <div class="metric-label">Top Diagnosis</div>
                    <div style="color:#c084fc; font-weight:600; font-size:0.95rem">{result.top_disease}</div>
                </div>
                <div>
                    <div class="metric-label">Probability</div>
                    <div style="color:#818cf8; font-weight:600; font-family:'JetBrains Mono',monospace">{result.top_disease_prob:.1%}</div>
                </div>
                <div>
                    <div class="metric-label">Confidence</div>
                    <div class="{conf_class}">{result.confidence_label}</div>
                </div>
            </div>
        </div>
        """, unsafe_allow_html=True)

    # MIDDLE ROW: Disease probabilities + EU chart
    col_chart, col_eu = st.columns([3, 2])

    with col_chart:
        st.markdown("#### <span class='step-badge'>4</span> Disease Probability Distribution",
                    unsafe_allow_html=True)
        if result.inference_result and result.inference_result.top_diseases:
            top_diseases = result.inference_result.top_diseases[:top_k]
            fig = create_disease_chart(top_diseases)
            st.plotly_chart(fig, use_container_width=True, config={'displayModeBar': False})
        else:
            st.info("No inference results available.")

    with col_eu:
        st.markdown("#### <span class='step-badge'>6</span> Expected Utility Analysis",
                    unsafe_allow_html=True)
        if result.decision_result:
            fig_eu = create_eu_chart(result.decision_result)
            st.plotly_chart(fig_eu, use_container_width=True, config={'displayModeBar': False})

    # BOTTOM ROW: Uncertainty + Explanation + Red Flags
    col_unc, col_exp = st.columns([1, 2])

    with col_unc:
        st.markdown("#### <span class='step-badge'>5</span> Uncertainty Analysis",
                    unsafe_allow_html=True)

        unc = result.uncertainty_report
        if unc:
            fig_gauge = create_entropy_gauge(unc.entropy)
            st.plotly_chart(fig_gauge, use_container_width=True, config={'displayModeBar': False})

            m1, m2 = st.columns(2)
            with m1:
                st.markdown(f"""
                <div class="metric-box">
                    <div class="metric-value">{unc.max_prob:.1%}</div>
                    <div class="metric-label">Max P(Disease)</div>
                </div>
                """, unsafe_allow_html=True)
            with m2:
                novel_color = "#f87171" if unc.is_novel else "#34d399"
                novel_txt = "Novel" if unc.is_novel else "Known"
                st.markdown(f"""
                <div class="metric-box">
                    <div class="metric-value" style="color:{novel_color}">{novel_txt}</div>
                    <div class="metric-label">Pattern Type</div>
                </div>
                """, unsafe_allow_html=True)

    with col_exp:
        st.markdown("#### <span class='step-badge'>7</span> Clinical Explanation",
                    unsafe_allow_html=True)

        exp = result.explanation
        if exp:
            # Summary
            st.markdown(f'<div class="card">{exp.summary}</div>', unsafe_allow_html=True)

            # LLM response if available
            if exp.used_llm and exp.llm_response:
                st.markdown("**🤖 AI Summary:**")
                st.info(exp.llm_response)

            # Tabs for details
            tab1, tab2 = st.tabs(["🔴 Red Flags", "📋 Monitoring"])
            with tab1:
                st.markdown(
                    f'<div style="font-size:0.87rem; color:#fca5a5; line-height:1.8">{exp.red_flags}</div>',
                    unsafe_allow_html=True
                )
            with tab2:
                st.markdown(
                    f'<div style="font-size:0.87rem; color:#94a3b8; line-height:1.8">{exp.monitoring_advice}</div>',
                    unsafe_allow_html=True
                )

    # INFERENCE LOG
    if show_log and result.inference_log:
        st.markdown("---")
        st.markdown("#### 🔬 Inference Step Log")
        log_html = '<div class="log-panel">'
        for step in result.inference_log:
            lines = step.split('\n')
            for i, line in enumerate(lines):
                if line.startswith('[Step') or line.startswith('[Error'):
                    log_html += f'<div class="log-step">{line}</div>'
                else:
                    log_html += f'<div class="log-detail">{line}</div>'
        log_html += '</div>'
        st.markdown(log_html, unsafe_allow_html=True)

    # EU Decision table
    if result.decision_result:
        with st.expander("📊 Full Expected Utility Table"):
            eu_data = []
            for au in result.decision_result.action_utilities:
                selected = "✅" if au.action == result.recommended_action else ""
                eu_data.append({
                    "Action": au.label,
                    "Expected Utility": f"{au.expected_utility:.2f}",
                    "Recommended": selected
                })
            eu_df = pd.DataFrame(eu_data)
            st.dataframe(eu_df, use_container_width=True, hide_index=True)

elif analyze and not user_input.strip():
    st.warning("Please describe your symptoms first.")


# ── Empty State ───────────────────────────────────────────────────────────────

else:
    st.markdown("""
    <div style="text-align:center; padding:3rem 1rem; opacity:0.5">
        <div style="font-size:4rem; margin-bottom:1rem">🩺</div>
        <div style="font-size:1.1rem; color:#475569">
            Enter your symptoms above to begin the probabilistic analysis pipeline.<br>
            <small style="font-size:0.8rem">Or choose a quick example from the sidebar.</small>
        </div>
    </div>
    """, unsafe_allow_html=True)

    # Architecture diagram
    with st.expander("🔧 Pipeline Architecture"):
        steps_data = [
            ("1", "Symptom Extractor", "BioBERT NER via HuggingFace Transformers", "#6366f1"),
            ("2", "Symptom Normalizer", "Dictionary + SBERT Semantic Fallback", "#8b5cf6"),
            ("3", "Soft Evidence Creator", "Confidence → P(symptom=present) likelihood arrays", "#a855f7"),
            ("4", "Bayesian Network", "pgmpy Naive Bayes BN, trained on Mendeley dataset", "#c084fc"),
            ("5", "Uncertainty Detector", "Shannon Entropy + Open-Set Novelty Score", "#7c3aed"),
            ("6", "Utility Decision", "Expected Utility EU(a) = Σ P(d)·U(a,d)", "#6d28d9"),
            ("7", "Explanation Generator", "Templates + Claude LLM for conversational phrasing", "#5b21b6"),
            ("8", "Streamlit UI", "Interactive demo with inference log", "#4c1d95"),
        ]
        for num, name, desc, color in steps_data:
            st.markdown(
                f'<div class="card" style="border-left:4px solid {color}; margin-bottom:0.5rem">'
                f'<span class="step-badge">{num}</span>'
                f'<strong style="color:#e2e8f0">{name}</strong>'
                f'<span style="color:#64748b; font-size:0.85rem; margin-left:0.5rem">— {desc}</span>'
                f'</div>',
                unsafe_allow_html=True
            )


# ── Footer ─────────────────────────────────────────────────────────────────────

st.markdown("---")
st.markdown("""
<div style="text-align:center; font-size:0.75rem; color:#334155; padding:0.5rem 0">
    ⚠️ <strong>Medical Disclaimer</strong> — MedDecide is an educational AI tool only. 
    It is NOT a substitute for professional medical advice, diagnosis, or treatment. 
    Always consult a qualified healthcare provider.
</div>
""", unsafe_allow_html=True)
