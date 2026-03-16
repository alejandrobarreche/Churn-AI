import streamlit as st
from .config import (
    UAX_BG, UAX_CARD, UAX_CARD2, UAX_GOLD, UAX_GREY, UAX_NAVY,
    UAX_ACCENT, UAX_TEXT,
)


def inject_css():
    st.markdown(f"""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Orbitron:wght@400;500;600;700;800;900&family=Rajdhani:wght@300;400;500;600;700&family=Inter:wght@300;400;500;600&display=swap');

    /* Global */
    .stApp {{
        background: linear-gradient(135deg, {UAX_BG} 0%, #0D1321 40%, #121A2B 100%);
        color: {UAX_TEXT};
    }}

    /* Hide default streamlit chrome */
    #MainMenu {{visibility: hidden;}}
    footer {{visibility: hidden;}}

    /* Style the header toolbar to match dark theme */
    header[data-testid="stHeader"] {{
        background: {UAX_BG} !important;
        border-bottom: 1px solid rgba(200,169,81,0.10);
    }}

    /* Sidebar */
    section[data-testid="stSidebar"] {{
        background: linear-gradient(180deg, {UAX_CARD} 0%, {UAX_BG} 100%);
        border-right: 1px solid rgba(200, 169, 81, 0.15);
    }}
    section[data-testid="stSidebar"] .stMarkdown p,
    section[data-testid="stSidebar"] .stMarkdown li,
    section[data-testid="stSidebar"] label {{
        color: {UAX_TEXT} !important;
        font-family: 'Rajdhani', sans-serif !important;
    }}

    /* Typography */
    h1, h2, h3 {{
        font-family: 'Orbitron', monospace !important;
        color: {UAX_GOLD} !important;
        letter-spacing: 2px;
    }}
    h1 {{
        font-size: 2.2rem !important;
        text-transform: uppercase;
        background: linear-gradient(90deg, {UAX_GOLD}, #E8D5A0, {UAX_GOLD});
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        text-shadow: none;
    }}
    h2 {{
        font-size: 1.3rem !important;
        border-bottom: 1px solid rgba(200, 169, 81, 0.2);
        padding-bottom: 8px;
    }}
    h3 {{
        font-size: 1.1rem !important;
        color: {UAX_TEXT} !important;
    }}
    p, li {{
        font-family: 'Rajdhani', sans-serif !important;
    }}
    /* Solo divs de contenido, no los que contienen iconos de Streamlit */
    .stMarkdown p, .stMarkdown li, .stMarkdown div,
    .stAlert p, .stDataFrame, .element-container {{
        font-family: 'Rajdhani', sans-serif;
    }}

    /* Metric cards */
    .metric-card {{
        background: linear-gradient(145deg, {UAX_CARD} 0%, {UAX_CARD2} 100%);
        border: 1px solid rgba(200, 169, 81, 0.12);
        border-radius: 12px;
        padding: 20px 24px;
        text-align: center;
        position: relative;
        overflow: hidden;
        transition: all 0.3s ease;
    }}
    .metric-card:hover {{
        border-color: rgba(200, 169, 81, 0.35);
        box-shadow: 0 0 30px rgba(200, 169, 81, 0.08);
    }}
    .metric-card::before {{
        content: '';
        position: absolute;
        top: 0;
        left: 0;
        right: 0;
        height: 2px;
        background: linear-gradient(90deg, transparent, {UAX_GOLD}, transparent);
    }}
    .metric-label {{
        font-family: 'Rajdhani', sans-serif;
        font-size: 0.85rem;
        color: {UAX_GREY};
        text-transform: uppercase;
        letter-spacing: 2px;
        margin-bottom: 6px;
    }}
    .metric-value {{
        font-family: 'Orbitron', monospace;
        font-size: 1.8rem;
        font-weight: 700;
        color: {UAX_GOLD};
        line-height: 1.2;
    }}
    .metric-delta {{
        font-family: 'Rajdhani', sans-serif;
        font-size: 0.9rem;
        color: {UAX_GREY};
        margin-top: 4px;
    }}

    /* Section separator */
    .section-line {{
        height: 1px;
        background: linear-gradient(90deg, transparent, rgba(200,169,81,0.3), transparent);
        margin: 30px 0;
    }}

    /* Dataframe styling */
    .stDataFrame {{
        border: 1px solid rgba(200,169,81,0.1) !important;
        border-radius: 8px;
    }}

    /* Plotly charts background */
    .js-plotly-plot .plotly .main-svg {{
        border-radius: 12px;
    }}

    /* Tabs */
    .stTabs [data-baseweb="tab-list"] {{
        gap: 2px;
        background: {UAX_CARD};
        border-radius: 8px;
        padding: 4px;
    }}
    .stTabs [data-baseweb="tab"] {{
        font-family: 'Rajdhani', sans-serif !important;
        font-weight: 600;
        color: {UAX_GREY};
        border-radius: 6px;
        padding: 8px 20px;
    }}
    .stTabs [aria-selected="true"] {{
        background: {UAX_NAVY} !important;
        color: {UAX_GOLD} !important;
    }}

    /* Selectbox, slider */
    .stSelectbox label, .stSlider label, .stMultiSelect label {{
        font-family: 'Rajdhani', sans-serif !important;
        color: {UAX_GREY} !important;
        text-transform: uppercase;
        letter-spacing: 1px;
        font-size: 0.85rem;
    }}

    /* Info box */
    .info-box {{
        background: {UAX_CARD};
        border: 1px solid rgba(200,169,81,0.2);
        border-radius: 10px;
        padding: 20px 24px;
        margin: 12px 0;
    }}
    .info-box-title {{
        font-family: 'Orbitron', monospace;
        color: {UAX_GOLD};
        font-size: 0.9rem;
        letter-spacing: 2px;
        text-transform: uppercase;
        margin-bottom: 10px;
    }}
    .info-box-text {{
        font-family: 'Rajdhani', sans-serif;
        color: {UAX_TEXT};
        font-size: 1rem;
        line-height: 1.6;
    }}

    /* Gauge result */
    .result-badge {{
        border-radius: 10px;
        padding: 14px 20px;
        text-align: center;
        font-family: 'Orbitron', monospace;
        font-size: 1.1rem;
        font-weight: 700;
        letter-spacing: 2px;
    }}
</style>
""", unsafe_allow_html=True)
