import streamlit as st
import pandas as pd
import plotly.graph_objects as go
from ..config import UAX_GOLD, UAX_GREY, UAX_TEXT, UAX_CARD, UAX_CARD2, UAX_GREEN
from ..utils import metric_card, section_line, plotly_layout


# ── Cached builders (model_metrics nunca cambia → id() es constante) ────────────

@st.cache_data(hash_funcs={dict: id})
def _fig_roc(model_metrics):
    """Curvas ROC — no depende de la selección de modelo."""
    MODEL_COLORS = {'XGBoost': UAX_GOLD, 'Random Forest': UAX_GREEN, 'LightGBM': '#6EC6FF'}
    fig = go.Figure()
    fig.add_shape(type='line', x0=0, y0=0, x1=1, y1=1,
                  line=dict(dash='dash', color=UAX_GREY, width=1))
    for mn, mm in model_metrics.items():
        fig.add_trace(go.Scatter(
            x=mm['fpr'], y=mm['tpr'],
            name=f"{mn} (AUC={mm['auc']:.3f})",
            mode='lines',
            line=dict(width=2.5, color=MODEL_COLORS.get(mn, UAX_GOLD)),
        ))
    fig.update_layout(
        title="Curvas ROC — Comparación de modelos",
        xaxis_title="Tasa de Falsos Positivos (1 − Especificidad)",
        yaxis_title="Tasa de Verdaderos Positivos (Sensibilidad)",
    )
    plotly_layout(fig, height=420)
    return fig


@st.cache_data(hash_funcs={dict: id})
def _comparison_rows(model_metrics):
    """Tabla comparativa — no cambia."""
    return [
        {'Modelo': mn, 'AUC': f"{mm['auc']:.4f}", 'Precision': f"{mm['precision']:.4f}",
         'Recall': f"{mm['recall']:.4f}", 'F1': f"{mm['f1']:.4f}"}
        for mn, mm in model_metrics.items()
    ]


@st.cache_data(hash_funcs={dict: id})
def _fig_confusion(model_metrics, sel_model):
    """Matriz de confusión — cambia con el modelo seleccionado.

    sklearn almacena cm[0]=No Churn (clase 0), cm[1]=Churn (clase 1).
    Mostramos Churn arriba (clase positiva) invirtiendo el orden de filas.
    """
    cm = model_metrics[sel_model]['cm']
    # Churn (clase positiva) arriba → invertir filas
    z = [list(cm[1]), list(cm[0])]
    cell_labels = [
        [f"VN\n{cm[0][0]:,}", f"FP\n{cm[0][1]:,}"],   # fila Real No Churn
        [f"FN\n{cm[1][0]:,}", f"VP\n{cm[1][1]:,}"],   # fila Real Churn
    ]
    fig = go.Figure(go.Heatmap(
        z=z,
        x=['Predicho: No Churn', 'Predicho: Churn'],
        y=['Real: No Churn', 'Real: Churn'],
        text=cell_labels,
        texttemplate="%{text}",
        textfont=dict(size=16, family='Orbitron', color=UAX_TEXT),
        colorscale=[[0, UAX_CARD], [1, UAX_GOLD]],
        showscale=False,
    ))
    plotly_layout(fig, height=340)
    fig.update_layout(
        title="Matriz de Confusión",
        xaxis=dict(side='bottom', title='Predicción del modelo'),
        yaxis=dict(title='Valor real', autorange='reversed'),
        margin=dict(l=40, r=20, t=60, b=40),
    )
    return fig


@st.cache_data(hash_funcs={dict: id})
def _fig_feat_importance(model_metrics, sel_model):
    """Feature importance — cambia con el modelo seleccionado."""
    fi = model_metrics[sel_model].get('feat_imp', {})
    if not fi:
        return None
    fi_series = pd.Series(fi).sort_values(ascending=True).tail(10)
    fig = go.Figure(go.Bar(
        x=fi_series.values, y=fi_series.index,
        orientation='h', marker_color=UAX_GOLD, opacity=0.85,
    ))
    fig.update_layout(
        title="Variables más influyentes (Top 10)",
        xaxis_title="Importancia relativa",
        yaxis_title="Variable",
    )
    plotly_layout(fig, height=380)
    return fig


@st.cache_data(hash_funcs={dict: id})
def _fig_proba_dist(model_metrics, sel_model):
    """Distribución de probabilidades — cambia con el modelo seleccionado."""
    fig = go.Figure()
    fig.add_trace(go.Histogram(
        x=model_metrics[sel_model]['proba'],
        nbinsx=40, marker_color=UAX_GOLD, opacity=0.75, name=sel_model,
    ))
    fig.update_layout(
        title="Distribución de probabilidades de churn",
        xaxis_title="P(Churn) — probabilidad predicha",
        yaxis_title="Nº de clientes",
        bargap=0.05,
    )
    plotly_layout(fig, height=380)
    return fig


# ── Render ──────────────────────────────────────────────────────────────────────

def render(model_metrics):
    st.markdown("## Comparación de Modelos")

    if not model_metrics:
        st.warning("No hay métricas de modelos disponibles. Verifica que los archivos .pkl estén en data/warehouse/.")
        return

    sel_model = st.selectbox("Modelo a inspeccionar", list(model_metrics.keys()), key="mod_sel")
    m = model_metrics[sel_model]

    mk1, mk2, mk3, mk4 = st.columns(4)
    with mk1:
        st.markdown(metric_card("AUC-ROC", f"{m['auc']:.4f}"), unsafe_allow_html=True)
    with mk2:
        st.markdown(metric_card("Precision", f"{m['precision']:.4f}"), unsafe_allow_html=True)
    with mk3:
        st.markdown(metric_card("Recall", f"{m['recall']:.4f}"), unsafe_allow_html=True)
    with mk4:
        st.markdown(metric_card("F1-Score", f"{m['f1']:.4f}"), unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)
    section_line()

    col_cm, col_tbl = st.columns([1, 1])
    with col_cm:
        st.markdown(f"### Matriz de Confusión — {sel_model}")
        st.plotly_chart(_fig_confusion(model_metrics, sel_model), width='stretch')
    with col_tbl:
        st.markdown("### Tabla Comparativa")
        st.dataframe(pd.DataFrame(_comparison_rows(model_metrics)).set_index('Modelo'),
                     width='stretch')

    section_line()
    st.markdown("### Curvas ROC")
    st.plotly_chart(_fig_roc(model_metrics), width='stretch')

    section_line()
    col_fi, col_dist = st.columns([1, 1])
    with col_fi:
        st.markdown(f"### Feature Importance — {sel_model}")
        fig_fi = _fig_feat_importance(model_metrics, sel_model)
        if fig_fi:
            st.plotly_chart(fig_fi, width='stretch')
    with col_dist:
        st.markdown(f"### Distribución de Probabilidades — {sel_model}")
        st.plotly_chart(_fig_proba_dist(model_metrics, sel_model), width='stretch')
