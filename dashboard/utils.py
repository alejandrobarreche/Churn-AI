import streamlit as st
import plotly.graph_objects as go
from .config import UAX_CARD, UAX_GOLD, UAX_GREY, UAX_TEXT


def metric_card(label, value, delta=None):
    delta_html = f'<div class="metric-delta">{delta}</div>' if delta else ''
    return f"""
    <div class="metric-card">
        <div class="metric-label">{label}</div>
        <div class="metric-value">{value}</div>
        {delta_html}
    </div>
    """


def section_line():
    st.markdown('<div class="section-line"></div>', unsafe_allow_html=True)


def plotly_layout(fig, height=450):
    fig.update_layout(
        plot_bgcolor='rgba(0,0,0,0)',
        paper_bgcolor=UAX_CARD,
        font=dict(family='Rajdhani, sans-serif', color=UAX_TEXT, size=13),
        title_font=dict(family='Orbitron, monospace', color=UAX_GOLD, size=16),
        legend=dict(bgcolor='rgba(0,0,0,0)', bordercolor='rgba(200,169,81,0.15)', borderwidth=1),
        margin=dict(l=50, r=30, t=60, b=50),
        height=height,
        xaxis=dict(gridcolor='rgba(200,169,81,0.06)', zerolinecolor='rgba(200,169,81,0.1)'),
        yaxis=dict(gridcolor='rgba(200,169,81,0.06)', zerolinecolor='rgba(200,169,81,0.1)'),
    )
    return fig
