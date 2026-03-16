import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from ..config import (UAX_NAVY, UAX_GOLD, UAX_ACCENT, UAX_GREY, UAX_TEXT,
                      UAX_CARD, UAX_CARD2, UAX_BG, UAX_GREEN, UAX_RED)
from ..utils import metric_card, section_line, plotly_layout


def _dff_hash(df):
    return int(pd.util.hash_pandas_object(df.index).sum())


_RIESGO_ORDER = ['MUY_ALTO', 'ALTO', 'MEDIO', 'BAJO', 'MUY_BAJO']
_VALOR_ORDER  = ['Bajo', 'Medio', 'Alto']


# ── Cached figure builders ───────────────────────────────────────────────────

@st.cache_data(hash_funcs={pd.DataFrame: _dff_hash})
def _fig_heatmap_count(dff):
    pivot = dff.groupby(['riesgo', 'valor'], observed=False).size().unstack(fill_value=0)
    pivot = pivot.reindex(index=_RIESGO_ORDER, columns=_VALOR_ORDER, fill_value=0)
    fig = go.Figure(go.Heatmap(
        z=pivot.values, x=_VALOR_ORDER, y=_RIESGO_ORDER,
        text=[[f"{v:,}" for v in row] for row in pivot.values],
        texttemplate="%{text}", textfont=dict(size=18, family='Orbitron'),
        colorscale=[[0, UAX_BG], [1, UAX_NAVY]],
        showscale=False, hoverinfo='skip',
    ))
    fig.update_layout(title="Nº de clientes por segmento",
                      xaxis_title="Valor (CLTV)", yaxis_title="Riesgo (Churn)")
    plotly_layout(fig, height=380)
    return fig


@st.cache_data(hash_funcs={pd.DataFrame: _dff_hash})
def _fig_heatmap_cltv(dff):
    pivot = dff.groupby(['riesgo', 'valor'], observed=False)['CLTV'].mean().unstack(fill_value=0)
    pivot = pivot.reindex(index=_RIESGO_ORDER, columns=_VALOR_ORDER, fill_value=0)
    fig = go.Figure(go.Heatmap(
        z=pivot.values, x=_VALOR_ORDER, y=_RIESGO_ORDER,
        text=[[f"{v:,.0f}€" for v in row] for row in pivot.values],
        texttemplate="%{text}", textfont=dict(size=16, family='Orbitron'),
        colorscale=[[0, UAX_BG], [0.5, UAX_ACCENT], [1, UAX_GOLD]],
        showscale=False, hoverinfo='skip',
    ))
    fig.update_layout(title="CLTV medio (€) por segmento",
                      xaxis_title="Valor (CLTV)", yaxis_title="Riesgo (Churn)")
    plotly_layout(fig, height=380)
    return fig


@st.cache_data(hash_funcs={pd.DataFrame: _dff_hash})
def _fig_treemap(dff):
    seg_counts = dff.groupby('segmento').agg(
        count=('CODE', 'count'), cltv_mean=('CLTV', 'mean'), pchurn_mean=('p_churn', 'mean')
    ).reset_index()
    fig = px.treemap(seg_counts, path=['segmento'], values='count',
                     color='cltv_mean',
                     color_continuous_scale=[[0, UAX_NAVY], [0.5, UAX_ACCENT], [1, UAX_GOLD]],
                     hover_data={'cltv_mean': ':.0f', 'pchurn_mean': ':.2%'})
    fig.update_layout(title="Distribución de clientes por segmento",
                      coloraxis_colorbar_title="CLTV medio")
    plotly_layout(fig, height=450)
    return fig


@st.cache_data(hash_funcs={pd.DataFrame: _dff_hash})
def _accion_resumen(dff):
    return dff.groupby('accion').agg(
        clientes=('CODE', 'count'),
        coste_total=('coste_accion', 'sum'),
        cltv_medio=('CLTV', 'mean'),
        p_churn_medio=('p_churn', 'mean'),
    ).sort_values('coste_total', ascending=False).reset_index()


@st.cache_data(hash_funcs={pd.DataFrame: _dff_hash})
def _fig_bar_inversion(dff):
    resumen = _accion_resumen(dff)
    fig = go.Figure(go.Bar(
        y=resumen['accion'], x=resumen['coste_total'],
        orientation='h', marker_color=UAX_GOLD,
        marker_line_color=UAX_NAVY, marker_line_width=1,
        text=[f"{v:,.0f}€" for v in resumen['coste_total']],
        textposition='outside', textfont=dict(color=UAX_GOLD, size=12),
    ))
    fig.update_layout(title="Inversión total por acción", xaxis_title="Coste total (€)")
    plotly_layout(fig, height=400)
    return fig


@st.cache_data(hash_funcs={pd.DataFrame: _dff_hash})
def _fig_bar_clientes(dff):
    resumen = _accion_resumen(dff)
    fig = go.Figure(go.Bar(
        y=resumen['accion'], x=resumen['clientes'],
        orientation='h', marker_color=UAX_NAVY,
        marker_line_color=UAX_GOLD, marker_line_width=1,
        text=[f"{v:,}" for v in resumen['clientes']],
        textposition='outside', textfont=dict(color=UAX_TEXT, size=12),
    ))
    fig.update_layout(title="Clientes por acción", xaxis_title="Nº clientes")
    plotly_layout(fig, height=400)
    return fig


@st.cache_data(hash_funcs={pd.DataFrame: _dff_hash})
def _sim_data(dff):
    sim = dff.groupby('accion').agg(
        clientes=('CODE', 'count'),
        inversion=('coste_accion', 'sum'),
        ganancia_cltv=('ganancia_cltv', 'sum'),
    ).reset_index()
    sim['beneficio_neto'] = sim['ganancia_cltv'] - sim['inversion']
    return sim.sort_values('beneficio_neto', ascending=False)


@st.cache_data(hash_funcs={pd.DataFrame: _dff_hash})
def _fig_beneficio_neto(dff):
    sim = _sim_data(dff)
    colors = [UAX_GREEN if v > 0 else UAX_RED for v in sim['beneficio_neto']]
    fig = go.Figure(go.Bar(
        y=sim['accion'], x=sim['beneficio_neto'],
        orientation='h', marker_color=colors,
        text=[f"{v:,.0f}€" for v in sim['beneficio_neto']],
        textposition='outside', textfont=dict(size=11),
    ))
    fig.add_vline(x=0, line_color=UAX_GREY, line_width=1)
    fig.update_layout(title="Beneficio neto por acción", xaxis_title="Beneficio neto (€)")
    plotly_layout(fig, height=400)
    return fig


@st.cache_data(hash_funcs={pd.DataFrame: _dff_hash})
def _fig_roi(dff):
    roi_data = dff[dff['coste_accion'] > 0].groupby('accion')['ROI'].mean().sort_values()
    fig = go.Figure(go.Bar(
        y=roi_data.index, x=roi_data.values,
        orientation='h', marker_color=UAX_ACCENT,
        text=[f"{v:.1f}x" for v in roi_data.values],
        textposition='outside', textfont=dict(color=UAX_GOLD, size=12),
    ))
    fig.add_vline(x=0, line_color=UAX_GREY, line_width=1)
    fig.update_layout(title="ROI medio por acción (solo con inversión)", xaxis_title="ROI (x)")
    plotly_layout(fig, height=400)
    return fig


@st.cache_data(hash_funcs={pd.DataFrame: _dff_hash})
def _economic_totals(dff):
    total_cltv_sin   = dff['CLTV'].sum()
    total_ganancia   = dff['ganancia_cltv'].sum()
    total_inversion  = dff['coste_accion'].sum()
    total_neto       = total_ganancia - total_inversion
    roi_global       = total_neto / total_inversion if total_inversion > 0 else 0
    pct_mejora       = total_ganancia / total_cltv_sin * 100 if total_cltv_sin > 0 else 0
    n_con_accion     = int((dff['coste_accion'] > 0).sum())
    return total_cltv_sin, total_ganancia, total_inversion, total_neto, roi_global, pct_mejora, n_con_accion


@st.cache_data(hash_funcs={pd.DataFrame: _dff_hash})
def _fig_waterfall(dff):
    _, total_ganancia, total_inversion, total_neto, *_ = _economic_totals(dff)
    fig = go.Figure(go.Waterfall(
        orientation="v",
        x=["Ganancia CLTV esperada", "Inversión en acciones", "Beneficio neto"],
        y=[total_ganancia, -total_inversion, total_neto],
        measure=["absolute", "relative", "total"],
        text=[f"+{total_ganancia:,.0f}€", f"−{total_inversion:,.0f}€", f"{total_neto:,.0f}€"],
        textposition="outside",
        textfont=dict(family="Orbitron", size=13, color=UAX_TEXT),
        connector=dict(line=dict(color=UAX_GREY, width=1, dash="dot")),
        increasing=dict(marker=dict(color=UAX_GREEN)),
        decreasing=dict(marker=dict(color=UAX_RED)),
        totals=dict(marker=dict(color=UAX_GOLD)),
    ))
    fig.update_layout(title="Plan de acciones — flujo incremental",
                      yaxis_title="€", yaxis=dict(tickformat=",.0f"))
    plotly_layout(fig, height=420)
    return fig


@st.cache_data(hash_funcs={pd.DataFrame: _dff_hash})
def _fig_grouped_bar(dff):
    sim = _sim_data(dff)
    sim_inv = sim[sim['inversion'] > 0].copy()
    fig = go.Figure()
    fig.add_trace(go.Bar(
        name="Inversión", x=sim_inv['accion'], y=sim_inv['inversion'],
        marker_color=UAX_RED, opacity=0.75,
        text=[f"{v:,.0f}€" for v in sim_inv['inversion']],
        textposition='inside', textfont=dict(size=10, color=UAX_TEXT),
    ))
    fig.add_trace(go.Bar(
        name="Ganancia CLTV", x=sim_inv['accion'], y=sim_inv['ganancia_cltv'],
        marker_color=UAX_GREEN, opacity=0.75,
        text=[f"{v:,.0f}€" for v in sim_inv['ganancia_cltv']],
        textposition='inside', textfont=dict(size=10, color=UAX_TEXT),
    ))
    fig.update_layout(title="Inversión vs Ganancia por acción",
                      barmode='group', yaxis_title="€",
                      yaxis=dict(tickformat=",.0f"), xaxis_tickangle=-25)
    plotly_layout(fig, height=420)
    return fig


# ── Render ───────────────────────────────────────────────────────────────────

def render(dff):
    st.markdown("## Matriz de Segmentación: Riesgo × Valor")

    col1, col2 = st.columns(2)
    with col1:
        st.plotly_chart(_fig_heatmap_count(dff), width='stretch')
    with col2:
        st.plotly_chart(_fig_heatmap_cltv(dff), width='stretch')

    st.plotly_chart(_fig_treemap(dff), width='stretch')

    # ── ACCIONES COMERCIALES ─────────────────────────────────────
    section_line()
    st.markdown("## Acciones Comerciales")

    col1, col2 = st.columns(2)
    with col1:
        st.plotly_chart(_fig_bar_inversion(dff), width='stretch')
    with col2:
        st.plotly_chart(_fig_bar_clientes(dff), width='stretch')

    # ── Reglas de negocio (static HTML, no caching needed) ───────
    st.markdown("### Reglas de negocio")

    RIESGO_CARD_COLORS = {
        'MUY_ALTO': '#ef4444', 'ALTO': '#f97316',
        'MEDIO':    '#f59e0b', 'BAJO': '#22c55e', 'MUY_BAJO': '#10b981',
    }
    RIESGO_ICONS = {
        'MUY_ALTO': '🔴', 'ALTO': '🟠', 'MEDIO': '🟡',
        'BAJO': '🟢', 'MUY_BAJO': '✅',
    }
    RIESGO_RULES = {
        'MUY_ALTO': {
            'tr': '10%', 'delta_churn': '−15% / −5%', 'coste_resp': '20€ / 0€',
            'acciones': [
                ('Alto/Medio CLTV', 'Contacto prioritario', 'Bono 20€'),
                ('Bajo CLTV',       'Contacto mínimo',      'Email de seguimiento'),
            ],
        },
        'ALTO': {
            'tr': '80%', 'delta_churn': '−40% / −5%', 'coste_resp': '150€ / 0€',
            'acciones': [
                ('Alto/Medio CLTV', 'Pack Premium VIP', 'Recogida + lavado + neumáticos + bono 50€'),
                ('Bajo CLTV',       'Contacto mínimo',  'Email de seguimiento'),
            ],
        },
        'MEDIO': {
            'tr': '55%', 'delta_churn': '−20% / −5%', 'coste_resp': '66€ / 0€',
            'acciones': [
                ('Alto/Medio CLTV', 'Pack Intermedio',      'Regalo + lavado + bono 30€'),
                ('Bajo CLTV',       'Seguimiento estándar', 'Recordatorio de revisión'),
            ],
        },
        'BAJO': {
            'tr': '45%', 'delta_churn': '−10% / −5% / 0%', 'coste_resp': '0€',
            'acciones': [
                ('Alto CLTV',  'Upselling',    'Ext. garantía / seguro batería'),
                ('Medio CLTV', 'Mantenimiento','Comunicación periódica'),
                ('Bajo CLTV',  'Sin acción',   '—'),
            ],
        },
        'MUY_BAJO': {
            'tr': '40%', 'delta_churn': '−10% / −5% / 0%', 'coste_resp': '0€',
            'acciones': [
                ('Alto CLTV',  'Upselling',    'Ext. garantía / seguro batería'),
                ('Medio CLTV', 'Mantenimiento','Comunicación periódica'),
                ('Bajo CLTV',  'Sin acción',   '—'),
            ],
        },
    }

    for riesgo_key in _RIESGO_ORDER:
        rcolor = RIESGO_CARD_COLORS[riesgo_key]
        ricon  = RIESGO_ICONS[riesgo_key]
        rdata  = RIESGO_RULES[riesgo_key]
        with st.expander(f"{ricon} {riesgo_key}", expanded=False):
            exp_col1, exp_col2 = st.columns(2)
            with exp_col1:
                rows_html = ""
                for cltv_lvl, accion_name, servicios in rdata['acciones']:
                    rows_html += f"""
                    <tr>
                        <td style="color:{UAX_GREY}; padding:5px 8px; font-size:0.88rem;">{cltv_lvl}</td>
                        <td style="color:{UAX_TEXT}; padding:5px 8px; font-weight:600; font-size:0.88rem;">{accion_name}</td>
                        <td style="color:{UAX_GREY}; padding:5px 8px; font-size:0.82rem;">{servicios}</td>
                    </tr>"""
                st.markdown(f"""
                <div style="margin-bottom:8px;">
                    <span style="font-family:'Rajdhani',sans-serif; font-size:0.8rem; color:{rcolor};
                                 text-transform:uppercase; letter-spacing:1px; font-weight:600;">
                        Acciones por CLTV
                    </span>
                </div>
                <table style="width:100%; font-family:'Rajdhani',sans-serif; border-collapse:collapse;">
                    <thead>
                        <tr>
                            <th style="color:{UAX_GREY}; font-size:0.75rem; text-align:left; padding:4px 8px; border-bottom:1px solid rgba(200,169,81,0.15);">CLTV</th>
                            <th style="color:{UAX_GREY}; font-size:0.75rem; text-align:left; padding:4px 8px; border-bottom:1px solid rgba(200,169,81,0.15);">Acción</th>
                            <th style="color:{UAX_GREY}; font-size:0.75rem; text-align:left; padding:4px 8px; border-bottom:1px solid rgba(200,169,81,0.15);">Servicios</th>
                        </tr>
                    </thead>
                    <tbody>{rows_html}</tbody>
                </table>
                """, unsafe_allow_html=True)
            with exp_col2:
                r, g, b = int(rcolor[1:3], 16), int(rcolor[3:5], 16), int(rcolor[5:7], 16)
                st.markdown(f"""
                <div style="background:rgba({r},{g},{b},0.08);
                            border:1px solid {rcolor}40; border-radius:10px; padding:16px 18px;">
                    <div style="font-family:'Rajdhani',sans-serif; font-size:0.8rem; color:{rcolor};
                                text-transform:uppercase; letter-spacing:1px; font-weight:600; margin-bottom:12px;">
                        Parámetros económicos
                    </div>
                    <div style="font-family:'Rajdhani',sans-serif; font-size:0.95rem; color:{UAX_TEXT}; line-height:2.2;">
                        <span style="color:{UAX_GREY};">Tasa de respuesta:</span>
                        <span style="color:{UAX_GOLD}; font-weight:600;"> {rdata['tr']}</span><br>
                        <span style="color:{UAX_GREY};">Δ churn estimado:</span>
                        <span style="color:{UAX_GOLD}; font-weight:600;"> {rdata['delta_churn']}</span><br>
                        <span style="color:{UAX_GREY};">Coste/respondedor:</span>
                        <span style="color:{UAX_GOLD}; font-weight:600;"> {rdata['coste_resp']}</span>
                    </div>
                </div>
                """, unsafe_allow_html=True)

    # ── SIMULACIÓN ECONÓMICA ROI ─────────────────────────────────
    section_line()
    st.markdown("## Simulación Económica — ROI")

    total_cltv_sin, total_ganancia, total_inversion, total_neto, roi_global, pct_mejora, n_con_accion = _economic_totals(dff)

    kc1, kc2, kc3, kc4 = st.columns(4)
    with kc1:
        st.markdown(metric_card("CLTV Cartera", f"{total_cltv_sin/1e6:.2f}M€", "sin acciones"), unsafe_allow_html=True)
    with kc2:
        st.markdown(metric_card("Ganancia CLTV", f"+{total_ganancia:,.0f}€", f"+{pct_mejora:.2f}% sobre cartera"), unsafe_allow_html=True)
    with kc3:
        st.markdown(metric_card("Inversión total", f"{total_inversion:,.0f}€", f"{n_con_accion} clientes con acción"), unsafe_allow_html=True)
    with kc4:
        st.markdown(metric_card("Beneficio neto", f"{total_neto:,.0f}€", f"ROI global: {roi_global:.1f}x"), unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)

    col1, col2 = st.columns(2)
    with col1:
        st.plotly_chart(_fig_beneficio_neto(dff), width='stretch')
    with col2:
        st.plotly_chart(_fig_roi(dff), width='stretch')

    section_line()
    st.markdown("### Desglose económico global")

    col_wf, col_bar = st.columns([3, 2])
    with col_wf:
        st.plotly_chart(_fig_waterfall(dff), width='stretch')
    with col_bar:
        st.plotly_chart(_fig_grouped_bar(dff), width='stretch')
