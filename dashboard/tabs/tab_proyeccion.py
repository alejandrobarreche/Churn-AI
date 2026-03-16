import streamlit as st
import pandas as pd
import plotly.graph_objects as go
from ..config import UAX_GOLD, UAX_GREY, UAX_TEXT, UAX_CARD, UAX_CARD2, UAX_RED, UAX_GREEN
from ..utils import section_line, plotly_layout


# ── Cached builders (df_rev y costes nunca cambian → id() es constante) ────────

@st.cache_data(hash_funcs={pd.DataFrame: id})
def _fig_revision(df_rev):
    modelos = sorted(df_rev['Modelo'].unique())
    fig = go.Figure()
    for modelo in modelos:
        sub = df_rev[df_rev['Modelo'] == modelo]
        fig.add_trace(go.Scatter(
            x=sub['Revisión'], y=sub['Beneficio neto'],
            mode='lines+markers', name=f'Modelo {modelo}',
            line=dict(width=2.5), marker=dict(size=7),
        ))
    fig.add_vline(x=5, line_dash="dash", line_color=UAX_RED, line_width=2,
                  annotation_text="n≥5: elegible dto 2º vehículo",
                  annotation_font_color=UAX_RED)
    fig.update_layout(title="Evolución del beneficio neto por revisión",
                      xaxis_title="Nº de revisión", yaxis_title="Beneficio neto (€)",
                      xaxis=dict(dtick=1))
    plotly_layout(fig, height=500)
    return fig


@st.cache_data(hash_funcs={pd.DataFrame: id})
def _pivot_revision(df_rev):
    modelos = sorted(df_rev['Modelo'].unique())
    df_f = df_rev[df_rev['Modelo'].isin(modelos)]
    return df_f.pivot_table(index='Modelo', columns='Revisión', values='Beneficio neto').round(0)


@st.cache_data(hash_funcs={pd.DataFrame: id})
def _fig_sensitivity(costes):
    costes_dict = costes.set_index('Modelo')['Mantenimiento_medio'].to_dict()
    base  = costes_dict.get('A', 300)
    alpha = 0.07
    MARGEN = 0.62
    p_values = [0.1, 0.3, 0.5, 0.7, 0.9]
    p_labels = ['p=0.1 (muy bajo)', 'p=0.3 (bajo)', 'p=0.5 (medio)',
                'p=0.7 (alto)', 'p=0.9 (muy alto)']
    p_colors = [UAX_GREEN, '#80D080', UAX_GOLD, UAX_RED, '#FF4444']
    revisiones = list(range(1, 11))

    fig = go.Figure()
    for p_val, p_label, p_color in zip(p_values, p_labels, p_colors):
        running, acum = 0, []
        for n in revisiones:
            running += base * (1 + alpha) ** n * MARGEN * (1 - p_val) ** n
            acum.append(running)
        fig.add_trace(go.Scatter(
            x=revisiones, y=acum, mode='lines+markers',
            name=p_label, line=dict(color=p_color, width=2.5),
            marker=dict(size=7, color=p_color),
        ))
    fig.update_layout(
        title="Sensibilidad del CLTV según riesgo de churn (Modelo A, α=7%)",
        xaxis_title="Nº de revisiones", yaxis_title="CLTV acumulado (€)",
        xaxis=dict(dtick=1), yaxis=dict(tickformat=",.0f"),
    )
    plotly_layout(fig, height=450)
    return fig


# ── Render ──────────────────────────────────────────────────────────────────────

def render(df_rev, costes):
    st.markdown("## Proyección de Revisiones Futuras")

    st.plotly_chart(_fig_revision(df_rev), width='stretch')

    st.markdown("### Tabla de beneficios por revisión y modelo")
    pivot = _pivot_revision(df_rev)
    st.dataframe(pivot.style.format("{:,.0f}€").background_gradient(
        cmap='YlOrRd', axis=1), width='stretch')

    st.markdown(f"""
    <div style="background:{UAX_CARD}; border:1px solid rgba(200,169,81,0.2); border-radius:10px;
                padding:20px; margin-top:20px;">
        <span style="font-family:'Orbitron'; color:{UAX_GOLD}; font-size:1rem;">
            DESCUENTO 2 VEHICULO
        </span>
        <p style="color:{UAX_TEXT}; margin-top:10px; font-size:0.95rem;">
            A partir de la <b>revisión 5</b>, los clientes con CLTV medio-alto son elegibles para un
            <b>descuento fijo de 1.000€</b> en la compra de un segundo vehículo.
            Los clientes actuales (n=0) llegarán a esta revisión en aproximadamente 2-3 años.
        </p>
    </div>
    """, unsafe_allow_html=True)

    section_line()
    st.markdown("### ¿Cómo funciona la proyección?")
    st.markdown(f"""
    <div style="display:flex; gap:16px; flex-wrap:wrap; margin: 16px 0;">
        <div style="flex:1; min-width:260px; background:linear-gradient(145deg,{UAX_CARD},{UAX_CARD2});
                    border:1px solid rgba(200,169,81,0.25); border-radius:12px; padding:20px 22px;">
            <div style="font-family:'Orbitron',monospace; color:{UAX_GOLD}; font-size:0.75rem;
                        letter-spacing:2px; text-transform:uppercase; margin-bottom:12px;">
                Ingresos por revisión
            </div>
            <div style="font-family:'Rajdhani',sans-serif; font-size:1.3rem; color:{UAX_TEXT}; margin-bottom:8px;">
                <span style="color:{UAX_GOLD};">C(n)</span> = BASE × (1 + α)<sup>n</sup>
            </div>
            <div style="font-family:'Rajdhani',sans-serif; font-size:0.88rem; color:{UAX_GREY}; line-height:1.7;">
                BASE = coste medio de mantenimiento por modelo<br>
                α = tasa de crecimiento (7% modelos A/B, 10% resto)<br>
                n = número de revisión (1 → 10)
            </div>
        </div>
        <div style="flex:1; min-width:260px; background:linear-gradient(145deg,{UAX_CARD},{UAX_CARD2});
                    border:1px solid rgba(200,169,81,0.25); border-radius:12px; padding:20px 22px;">
            <div style="font-family:'Orbitron',monospace; color:{UAX_GOLD}; font-size:0.75rem;
                        letter-spacing:2px; text-transform:uppercase; margin-bottom:12px;">
                CLTV acumulado
            </div>
            <div style="font-family:'Rajdhani',sans-serif; font-size:1.3rem; color:{UAX_TEXT}; margin-bottom:8px;">
                <span style="color:{UAX_GOLD};">CLTV</span> = Σ C(n) × 0.62 × (1−p)<sup>n</sup>
            </div>
            <div style="font-family:'Rajdhani',sans-serif; font-size:0.88rem; color:{UAX_GREY}; line-height:1.7;">
                0.62 = margen neto sobre ingresos<br>
                p = probabilidad de churn individual<br>
                (1−p)<sup>n</sup> = probabilidad de supervivencia
            </div>
        </div>
    </div>
    """, unsafe_allow_html=True)

    section_line()
    st.markdown("### Sensibilidad del CLTV según riesgo de churn")
    st.plotly_chart(_fig_sensitivity(costes), width='stretch')
