"""
CLTV & Acciones Comerciales — Churn Prediction Dashboard
Universidad Alfonso X el Sabio (UAX)
"""
import traceback
import pandas as pd
import streamlit as st

pd.options.future.infer_string = False

# ── Page config (must be first Streamlit call) ───────────────────
st.set_page_config(
    page_title="CLTV Dashboard — UAX Churn",
    page_icon="🔷",
    layout="wide",
    initial_sidebar_state="expanded",
)

from dashboard.styles import inject_css
from dashboard.config import UAX_GREY
from dashboard.data_loader import load_and_compute
from dashboard.utils import section_line
from dashboard import sidebar
from dashboard.tabs import (tab_resumen, tab_cltv, tab_segmentacion,
                             tab_proyeccion, tab_modelos, tab_predictor)

# ── CSS ──────────────────────────────────────────────────────────
inject_css()

# ── Load data ────────────────────────────────────────────────────
try:
    df, costes, df_rev, BEST_THRESHOLD, full_pipeline, xgboost_model, train_medianas, model_metrics = load_and_compute()
    data_loaded = True
except Exception as e:
    data_loaded = False
    error_msg = f"{type(e).__name__}: {e}\n\n{traceback.format_exc()}"

# ── Sidebar ──────────────────────────────────────────────────────
dff = sidebar.render(df if data_loaded else None, data_loaded)

# ── Error state ──────────────────────────────────────────────────
if not data_loaded:
    st.error("Error al cargar datos")
    st.code(error_msg, language="python")
    st.info("Asegúrate de ejecutar el notebook 5-cltv_acciones.ipynb primero, o de que las rutas a los archivos sean correctas.")
    st.stop()

# ── Header ───────────────────────────────────────────────────────
st.markdown("""
<div style="text-align:center; padding: 15px 0 5px 0;">
    <h1>CLTV & Acciones Comerciales</h1>
    <p style="color:#6B7B8D; font-size:1.05rem; letter-spacing:1px;">
        Predicción de Churn · Segmentación · Estrategia de Retención
    </p>
</div>
""", unsafe_allow_html=True)

section_line()

# ── Tabs ─────────────────────────────────────────────────────────
tab1, tab2, tab3, tab4, tab5, tab6 = st.tabs([
    "🏠 Resumen",
    "📈 CLTV Analysis",
    "🎯 Segmentación & Acciones",
    "🔮 Proyección Revisiones",
    "🔬 Modelos",
    "🧮 Predictor de Churn",
])

with tab1:
    tab_resumen.render(dff, df, BEST_THRESHOLD, model_metrics)

with tab2:
    tab_cltv.render(dff)

with tab3:
    tab_segmentacion.render(dff)

with tab4:
    tab_proyeccion.render(df_rev, costes)

with tab5:
    tab_modelos.render(model_metrics)

with tab6:
    tab_predictor.render(df, BEST_THRESHOLD, full_pipeline, xgboost_model, train_medianas, costes)

# ── Footer ───────────────────────────────────────────────────────
section_line()
st.markdown(f"""
<div style="text-align:center; padding:20px 0;">
    <span style="font-family:'Orbitron'; font-size:0.7rem; color:{UAX_GREY}; letter-spacing:3px;">
        UAX CHURN INTELLIGENCE · UNIVERSIDAD ALFONSO X EL SABIO · 2025
    </span>
</div>
""", unsafe_allow_html=True)
