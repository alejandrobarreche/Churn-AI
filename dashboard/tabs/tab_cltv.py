import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from ..config import UAX_NAVY, UAX_GOLD, UAX_ACCENT, UAX_GREY, UAX_TEXT, UAX_CARD, UAX_CARD2, UAX_GREEN, UAX_RED
from ..utils import section_line, plotly_layout


def _dff_hash(df):
    """Hash rápido por índice — identifica unívocamente el conjunto filtrado."""
    return int(pd.util.hash_pandas_object(df.index).sum())


@st.cache_data(hash_funcs={pd.DataFrame: _dff_hash})
def _fig_hist_cltv(dff):
    fig = go.Figure()
    fig.add_trace(go.Histogram(x=dff['CLTV'], nbinsx=40,
                               marker_color=UAX_NAVY, marker_line_color=UAX_GOLD,
                               marker_line_width=0.5, opacity=0.85))
    fig.add_vline(x=dff['CLTV'].mean(), line_dash="dash", line_color=UAX_GOLD, line_width=2,
                  annotation_text=f"Media: {dff['CLTV'].mean():,.0f}€",
                  annotation_font_color=UAX_GOLD)
    fig.update_layout(title="Distribución del CLTV",
                      xaxis_title="CLTV (€)", yaxis_title="Nº Clientes")
    plotly_layout(fig)
    return fig


@st.cache_data(hash_funcs={pd.DataFrame: _dff_hash})
def _fig_bar_modelo(dff):
    cltv_modelo = dff.groupby('Modelo_letra')['CLTV'].mean().sort_values()
    fig = go.Figure(go.Bar(x=cltv_modelo.values, y=cltv_modelo.index,
                           orientation='h', marker_color=UAX_GOLD,
                           marker_line_color=UAX_NAVY, marker_line_width=1))
    fig.update_layout(title="CLTV medio por modelo", xaxis_title="CLTV (€)")
    plotly_layout(fig)
    return fig


@st.cache_data(hash_funcs={pd.DataFrame: _dff_hash})
def _fig_scatter_pvp(dff):
    fig = px.scatter(dff, x='p_churn', y='CLTV', color='PVP_original',
                     color_continuous_scale=[[0, UAX_NAVY], [0.5, UAX_ACCENT], [1, UAX_GOLD]],
                     opacity=0.5, hover_data=['CODE', 'Modelo_letra', 'accion'])
    fig.update_layout(title="P(Churn) vs CLTV — coloreado por PVP",
                      xaxis_title="P(Churn)", yaxis_title="CLTV (€)",
                      coloraxis_colorbar_title="PVP (€)")
    plotly_layout(fig, height=500)
    return fig


@st.cache_data(hash_funcs={pd.DataFrame: _dff_hash})
def _fig_scatter_cliente(dff, cliente_sel):
    RIESGO_COLORS = {
        'MUY_BAJO': UAX_GREEN, 'BAJO': '#80D080',
        'MEDIO': UAX_GOLD, 'ALTO': UAX_RED, 'MUY_ALTO': '#FF4444',
    }
    cliente_row = dff[dff['CODE'] == cliente_sel].iloc[0]
    fig = go.Figure()
    for seg in ['MUY_BAJO', 'BAJO', 'MEDIO', 'ALTO', 'MUY_ALTO']:
        sub = dff[dff['riesgo'] == seg]
        fig.add_trace(go.Scatter(
            x=sub['p_churn'], y=sub['CLTV'], mode='markers', name=seg,
            marker=dict(size=6, color=RIESGO_COLORS[seg], opacity=0.55),
            hovertemplate='<b>%{text}</b><br>P(Churn): %{x:.2%}<br>CLTV: %{y:,.0f}€<extra></extra>',
            text=sub['CODE'].astype(str),
        ))
    fig.add_trace(go.Scatter(
        x=[float(cliente_row['p_churn'])], y=[float(cliente_row['CLTV'])],
        mode='markers', name='Cliente seleccionado',
        marker=dict(size=18, color=UAX_GOLD, symbol='star',
                    line=dict(color='#FFFFFF', width=2)),
        hovertemplate=f'<b>{cliente_sel}</b><br>P(Churn): {float(cliente_row["p_churn"]):.2%}'
                      f'<br>CLTV: {float(cliente_row["CLTV"]):,.0f}€<extra></extra>',
    ))
    fig.update_layout(
        title="Posición del cliente en la cartera (P(Churn) vs CLTV)",
        xaxis_title="P(Churn)", yaxis_title="CLTV (€)",
        xaxis=dict(tickformat='.0%'), yaxis=dict(tickformat=",.0f"),
    )
    plotly_layout(fig, height=450)
    return fig, cliente_row


def render(dff):
    st.markdown("## Análisis del Customer Lifetime Value")

    col1, col2 = st.columns(2)
    with col1:
        st.plotly_chart(_fig_hist_cltv(dff), width='stretch')
    with col2:
        st.plotly_chart(_fig_bar_modelo(dff), width='stretch')

    st.plotly_chart(_fig_scatter_pvp(dff), width='stretch')

    section_line()
    st.markdown("## Análisis Individual de Cliente")

    if len(dff) == 0:
        return

    RIESGO_COLORS_CLTV = {
        'MUY_BAJO': UAX_GREEN, 'BAJO': '#80D080',
        'MEDIO': UAX_GOLD, 'ALTO': UAX_RED, 'MUY_ALTO': '#FF4444',
    }

    cliente_cltv_sel = st.selectbox(
        "Selecciona cliente (CODE)", sorted(dff['CODE'].unique()), key="cltv_cliente_sel"
    )
    cliente_row = dff[dff['CODE'] == cliente_cltv_sel].iloc[0]

    cl_col1, cl_col2 = st.columns(2)
    with cl_col1:
        pchurn_pct = float(cliente_row['p_churn']) * 100
        riesgo_color = RIESGO_COLORS_CLTV.get(str(cliente_row['riesgo']), UAX_GOLD)
        fig_g = go.Figure(go.Indicator(
            mode="gauge+number",
            value=pchurn_pct,
            number={'suffix': '%', 'font': {'family': 'Orbitron', 'color': UAX_GOLD, 'size': 32}},
            title={'text': "P(Churn)", 'font': {'family': 'Orbitron', 'color': UAX_TEXT, 'size': 13}},
            gauge={
                'axis': {'range': [0, 100], 'tickcolor': UAX_GREY,
                         'tickfont': {'family': 'Rajdhani', 'color': UAX_GREY}},
                'bar': {'color': riesgo_color, 'thickness': 0.25},
                'bgcolor': UAX_CARD2, 'borderwidth': 1, 'bordercolor': UAX_GREY,
                'steps': [
                    {'range': [0, 20],  'color': 'rgba(39,174,96,0.15)'},
                    {'range': [20, 40], 'color': 'rgba(200,169,81,0.12)'},
                    {'range': [40, 60], 'color': 'rgba(200,169,81,0.20)'},
                    {'range': [60, 80], 'color': 'rgba(192,57,43,0.15)'},
                    {'range': [80, 100],'color': 'rgba(192,57,43,0.25)'},
                ],
            }
        ))
        fig_g.update_layout(paper_bgcolor=UAX_CARD, font=dict(family='Rajdhani', color=UAX_TEXT),
                            height=300, margin=dict(l=20, r=20, t=50, b=10))
        st.plotly_chart(fig_g, width='stretch')

    with cl_col2:
        st.markdown(f"""
        <div style="background:linear-gradient(145deg,{UAX_CARD},{UAX_CARD2}); border:1px solid rgba(200,169,81,0.25);
                    border-radius:12px; padding:22px 20px; margin-top:10px;">
            <div style="font-family:'Orbitron',monospace; color:{UAX_GOLD}; font-size:0.85rem; letter-spacing:2px; margin-bottom:14px;">
                FICHA DE CLIENTE
            </div>
            <table style="width:100%; font-family:'Rajdhani',sans-serif; font-size:0.95rem; border-collapse:collapse;">
                <tr><td style="color:{UAX_GREY}; padding:4px 0;">CODE</td>
                    <td style="color:{UAX_TEXT}; font-weight:600; text-align:right;">{cliente_row['CODE']}</td></tr>
                <tr><td style="color:{UAX_GREY}; padding:4px 0;">Modelo</td>
                    <td style="color:{UAX_TEXT}; font-weight:600; text-align:right;">{cliente_row['Modelo_letra']}</td></tr>
                <tr><td style="color:{UAX_GREY}; padding:4px 0;">CLTV</td>
                    <td style="color:{UAX_GOLD}; font-weight:600; text-align:right;">{float(cliente_row['CLTV']):,.0f}€</td></tr>
                <tr><td style="color:{UAX_GREY}; padding:4px 0;">P(Churn)</td>
                    <td style="color:{UAX_TEXT}; font-weight:600; text-align:right;">{float(cliente_row['p_churn']):.2%}</td></tr>
                <tr><td style="color:{UAX_GREY}; padding:4px 0;">Riesgo</td>
                    <td style="color:{RIESGO_COLORS_CLTV.get(str(cliente_row['riesgo']), UAX_GOLD)}; font-weight:700; text-align:right;">{cliente_row['riesgo']}</td></tr>
                <tr><td style="color:{UAX_GREY}; padding:4px 0;">Acción</td>
                    <td style="color:{UAX_TEXT}; font-weight:600; text-align:right;">{cliente_row['accion']}</td></tr>
                <tr><td style="color:{UAX_GREY}; padding:4px 0;">PVP original</td>
                    <td style="color:{UAX_TEXT}; font-weight:600; text-align:right;">{float(cliente_row['PVP_original']):,.0f}€</td></tr>
                <tr><td style="color:{UAX_GREY}; padding:4px 0;">Zona</td>
                    <td style="color:{UAX_TEXT}; font-weight:600; text-align:right;">{cliente_row['ZONA_original']}</td></tr>
            </table>
        </div>
        """, unsafe_allow_html=True)

    fig_sel, _ = _fig_scatter_cliente(dff, cliente_cltv_sel)
    st.plotly_chart(fig_sel, width='stretch')
