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

    # ── Reglas de negocio + Personas ─────────────────────────────
    st.markdown("### Reglas de negocio por segmento")

    SEGMENTOS = {
        'MUY_ALTO': {
            'color': '#ef4444', 'icon': '🔴',
            'label': 'RIESGO MUY ALTO',
            'persona_nombre': 'El Desencantado',
            'persona_emoji':  '😤',
            'persona_desc':   'Cliente con historial de quejas o incidencias no resueltas. Lleva más de 6 meses sin visitar el taller. Compara activamente precios en internet y ya ha solicitado presupuesto en la competencia.',
            'persona_tags':   ['Alta sensibilidad al precio', 'Bajo engagement', 'Experiencia negativa reciente'],
            'accion_ppal':    'Contacto prioritario + Bono 20€',
            'accion_desc':    'Llamada personalizada en menos de 48h ofreciendo un bono de 20€ en la próxima revisión. Para clientes de bajo CLTV, email de seguimiento sin coste.',
            'tr':             '10%',
            'delta_churn':    '−15% (alto CLTV) / −5% (bajo CLTV)',
            'coste_resp':     '20€ / 0€',
            'justificacion':  'El coste de adquisición de un nuevo cliente es 5–7× mayor que el de retención. Con un CLTV medio de varios miles de euros, incluso una tasa de respuesta del 10% genera ROI positivo. La intervención rápida es crítica: cada semana sin contacto reduce la probabilidad de recuperación.',
        },
        'ALTO': {
            'color': '#f97316', 'icon': '🟠',
            'label': 'RIESGO ALTO',
            'persona_nombre': 'El Dudoso',
            'persona_emoji':  '🤔',
            'persona_desc':   'Cliente con visitas cada vez más espaciadas. Tiene un vehículo de gama media-alta y valora el servicio, pero siente que no recibe trato diferencial. Está abierto a quedarse si se le ofrece algo tangible.',
            'persona_tags':   ['Visitas irregulares', 'Sensible a la experiencia', 'Recuperable con incentivo'],
            'accion_ppal':    'Pack Premium VIP (alto/medio CLTV)',
            'accion_desc':    'Recogida del vehículo en domicilio + lavado completo + revisión de neumáticos + bono de 50€. Para bajo CLTV: email de seguimiento con oferta básica.',
            'tr':             '80%',
            'delta_churn':    '−40% (alto/medio CLTV) / −5% (bajo CLTV)',
            'coste_resp':     '150€ / 0€',
            'justificacion':  'Alta probabilidad de churn pero alta tasa de respuesta a incentivos premium. El Pack VIP (150€) se amortiza recuperando una sola revisión adicional (~300–500€ de ingreso). La recogida a domicilio elimina la fricción principal de este perfil: la falta de tiempo.',
        },
        'MEDIO': {
            'color': '#f59e0b', 'icon': '🟡',
            'label': 'RIESGO MEDIO',
            'persona_nombre': 'El Indeciso',
            'persona_emoji':  '😐',
            'persona_desc':   'Cliente regular pero sin fidelización consolidada. Cumple con las revisiones obligatorias pero no contrata servicios adicionales. Podría irse si la competencia le ofrece una promoción atractiva.',
            'persona_tags':   ['Revisiones puntuales', 'Sin servicios adicionales', 'Sensible a promociones'],
            'accion_ppal':    'Pack Intermedio (alto/medio CLTV)',
            'accion_desc':    'Regalo de bienvenida + lavado gratuito + bono de 30€ en próxima visita. Para bajo CLTV: recordatorio de revisión con tono amigable.',
            'tr':             '55%',
            'delta_churn':    '−20% (alto/medio CLTV) / −5% (bajo CLTV)',
            'coste_resp':     '66€ / 0€',
            'justificacion':  'Intervenir en riesgo medio evita que escale a riesgo alto, donde el coste de recuperación se multiplica. El Pack Intermedio (66€) crea una experiencia de valor percibido que refuerza el vínculo antes de que el cliente tome una decisión activa de cambio.',
        },
        'BAJO': {
            'color': '#22c55e', 'icon': '🟢',
            'label': 'RIESGO BAJO',
            'persona_nombre': 'El Fiel en Proceso',
            'persona_emoji':  '🙂',
            'persona_desc':   'Cliente satisfecho con visitas regulares. Conoce al personal del taller y confía en la marca. No busca activamente alternativas, pero responde bien a ofertas de valor añadido.',
            'persona_tags':   ['Visitas regulares', 'Confianza en la marca', 'Potencial de upselling'],
            'accion_ppal':    'Upselling (alto CLTV) / Mantenimiento relacional (medio)',
            'accion_desc':    'Propuesta de extensión de garantía o seguro de batería para CLTV alto. Comunicación periódica de valor (consejos, novedades) para CLTV medio. Sin acción para CLTV bajo.',
            'tr':             '45%',
            'delta_churn':    '−10% (alto) / −5% (medio) / 0% (bajo)',
            'coste_resp':     '0€',
            'justificacion':  'No requiere inversión en retención. El objetivo es maximizar CLTV ofreciendo servicios complementarios de alto margen. La extensión de garantía tiene coste casi nulo para el concesionario pero genera percepción de seguridad en el cliente, aumentando la probabilidad de renovación del vehículo.',
        },
        'MUY_BAJO': {
            'color': '#10b981', 'icon': '🟣',
            'label': 'RIESGO MUY BAJO',
            'persona_nombre': 'El Embajador',
            'persona_emoji':  '😊',
            'persona_desc':   'Cliente altamente fidelizado, con largo historial en la marca. Recomienda el concesionario a su entorno y renueva su vehículo con regularidad. Representa el perfil ideal de cliente y el benchmark de fidelización.',
            'persona_tags':   ['Largo historial', 'Promotor activo', 'Alta probabilidad de renovación'],
            'accion_ppal':    'Upselling premium (alto CLTV) / Comunicación de valor (medio)',
            'accion_desc':    'Programa de fidelización con acceso anticipado a novedades, seguro de batería o extensión de garantía. Para CLTV medio: newsletter y comunicación de valor. Para CLTV bajo: sin acción comercial activa.',
            'tr':             '40%',
            'delta_churn':    '−10% (alto) / −5% (medio) / 0% (bajo)',
            'coste_resp':     '0€',
            'justificacion':  'Inversión en retención mínima o nula — el cliente ya está fidelizado. El foco es maximizar el CLTV residual con servicios de alto margen y reforzar la relación para asegurar la próxima renovación de vehículo, que representa el mayor ingreso del ciclo de vida.',
        },
    }

    for seg_key in _RIESGO_ORDER:
        seg = SEGMENTOS[seg_key]
        rcolor = seg['color']
        r, g, b = int(rcolor[1:3], 16), int(rcolor[3:5], 16), int(rcolor[5:7], 16)

        with st.expander(f"{seg['icon']} {seg['label']} — {seg['persona_nombre']}", expanded=False):

            # ── Fila 1: Persona + Acción principal ───────────────
            col_persona, col_accion = st.columns([1, 1])

            with col_persona:
                tags_html = "".join([
                    f'<span style="background:rgba({r},{g},{b},0.15); color:{rcolor}; '
                    f'font-family:Rajdhani,sans-serif; font-size:0.75rem; font-weight:600; '
                    f'padding:3px 10px; border-radius:20px; margin-right:6px; '
                    f'border:1px solid {rcolor}50;">{t}</span>'
                    for t in seg['persona_tags']
                ])
                st.markdown(f"""
                <div style="background:rgba({r},{g},{b},0.06); border:1px solid {rcolor}40;
                            border-radius:12px; padding:18px 20px; height:100%;">
                    <div style="font-family:'Orbitron',monospace; font-size:0.6rem; color:{rcolor};
                                letter-spacing:3px; text-transform:uppercase; margin-bottom:10px;">
                        Persona
                    </div>
                    <div style="font-size:2rem; margin-bottom:6px;">{seg['persona_emoji']}</div>
                    <div style="font-family:'Orbitron',monospace; font-size:0.95rem; color:{UAX_TEXT};
                                font-weight:700; margin-bottom:10px;">
                        {seg['persona_nombre']}
                    </div>
                    <div style="font-family:'Rajdhani',sans-serif; font-size:0.9rem; color:{UAX_GREY};
                                line-height:1.6; margin-bottom:14px;">
                        {seg['persona_desc']}
                    </div>
                    <div style="display:flex; flex-wrap:wrap; gap:6px;">
                        {tags_html}
                    </div>
                </div>
                """, unsafe_allow_html=True)

            with col_accion:
                st.markdown(f"""
                <div style="background:linear-gradient(145deg,{UAX_CARD},{UAX_CARD2});
                            border:1px solid rgba(200,169,81,0.25); border-radius:12px;
                            padding:18px 20px; height:100%;">
                    <div style="font-family:'Orbitron',monospace; font-size:0.6rem; color:{UAX_GOLD};
                                letter-spacing:3px; text-transform:uppercase; margin-bottom:10px;">
                        Acción recomendada
                    </div>
                    <div style="font-family:'Rajdhani',sans-serif; font-size:1.05rem; color:{UAX_TEXT};
                                font-weight:700; margin-bottom:8px;">
                        {seg['accion_ppal']}
                    </div>
                    <div style="font-family:'Rajdhani',sans-serif; font-size:0.88rem; color:{UAX_GREY};
                                line-height:1.6; margin-bottom:16px;">
                        {seg['accion_desc']}
                    </div>
                    <div style="display:grid; grid-template-columns:1fr 1fr 1fr; gap:8px; margin-top:auto;">
                        <div style="background:rgba(200,169,81,0.07); border-radius:8px; padding:10px 8px; text-align:center;">
                            <div style="font-family:'Rajdhani'; font-size:0.68rem; color:{UAX_GREY};
                                        letter-spacing:1px; text-transform:uppercase; margin-bottom:4px;">
                                Tasa respuesta
                            </div>
                            <div style="font-family:'Orbitron'; font-size:1.1rem; color:{rcolor}; font-weight:700;">
                                {seg['tr']}
                            </div>
                        </div>
                        <div style="background:rgba(200,169,81,0.07); border-radius:8px; padding:10px 8px; text-align:center;">
                            <div style="font-family:'Rajdhani'; font-size:0.68rem; color:{UAX_GREY};
                                        letter-spacing:1px; text-transform:uppercase; margin-bottom:4px;">
                                Δ Churn
                            </div>
                            <div style="font-family:'Orbitron'; font-size:0.85rem; color:{UAX_GOLD}; font-weight:700;">
                                {seg['delta_churn'].split('(')[0].strip()}
                            </div>
                        </div>
                        <div style="background:rgba(200,169,81,0.07); border-radius:8px; padding:10px 8px; text-align:center;">
                            <div style="font-family:'Rajdhani'; font-size:0.68rem; color:{UAX_GREY};
                                        letter-spacing:1px; text-transform:uppercase; margin-bottom:4px;">
                                Coste/resp.
                            </div>
                            <div style="font-family:'Orbitron'; font-size:1.1rem; color:{UAX_TEXT}; font-weight:700;">
                                {seg['coste_resp']}
                            </div>
                        </div>
                    </div>
                </div>
                """, unsafe_allow_html=True)

            # ── Fila 2: Justificación de negocio ─────────────────
            st.markdown(f"""
            <div style="background:rgba({r},{g},{b},0.04); border-left:3px solid {rcolor};
                        border-radius:0 8px 8px 0; padding:14px 18px; margin-top:12px;">
                <div style="font-family:'Orbitron',monospace; font-size:0.6rem; color:{rcolor};
                            letter-spacing:2px; text-transform:uppercase; margin-bottom:8px;">
                    💡 Justificación de negocio
                </div>
                <div style="font-family:'Rajdhani',sans-serif; font-size:0.92rem; color:{UAX_GREY}; line-height:1.7;">
                    {seg['justificacion']}
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
