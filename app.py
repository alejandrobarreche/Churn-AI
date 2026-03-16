"""
CLTV & Acciones Comerciales — Churn Prediction Dashboard
Universidad Alfonso X el Sabio (UAX)
"""
import sys, importlib
sys.path.insert(0, '.')
from transformer import (BinaryEncoder, FrequencyEncoder, OrdinalExtensionEncoder,
                         OrdinalEquipamientoEncoder, NominalOneHotEncoder,
                         GastoRelativoEncoder, PriceStandard, InstanceDropper, ColumnDropper)
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split
import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import joblib
import traceback
import os
from sklearn.metrics import (roc_curve, auc, precision_score, recall_score,
                              f1_score, confusion_matrix)
pd.options.future.infer_string = False

# ═══════════════════════════════════════════════════════════════
# CONFIG
# ═══════════════════════════════════════════════════════════════
st.set_page_config(
    page_title="CLTV Dashboard — UAX Churn",
    page_icon="🔷",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ═══════════════════════════════════════════════════════════════
# THEME & CSS
# ═══════════════════════════════════════════════════════════════
UAX_NAVY   = "#1B2A4A"
UAX_GOLD   = "#C8A951"
UAX_ACCENT = "#3A5BA0"
UAX_GREY   = "#6B7B8D"
UAX_RED    = "#C0392B"
UAX_GREEN  = "#27AE60"
UAX_BG     = "#0B0F1A"
UAX_CARD   = "#111827"
UAX_CARD2  = "#1A2332"
UAX_TEXT   = "#E2E8F0"

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


# ═══════════════════════════════════════════════════════════════
# HELPER FUNCTIONS
# ═══════════════════════════════════════════════════════════════
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


# ═══════════════════════════════════════════════════════════════
# DATA LOADING & COMPUTATION (same logic as notebook 5)
# ═══════════════════════════════════════════════════════════════
@st.cache_resource
def load_and_compute():
    """Load raw data and compute CLTV + segmentation. No invented data."""

    # --- Load ---
    costes = pd.read_csv('data/lake/costes.csv')
    new_customers = pd.read_csv('data/lake/nuevos_clientes.csv')
    for col in ['Sales_Date', 'FIN_GARANTIA', 'BASE_DATE']:
        if col in new_customers.columns:
            new_customers[col] = pd.to_datetime(new_customers[col], errors='coerce', dayfirst=True, format='%d/%m/%Y')

    xgboost_model  = joblib.load('data/warehouse/xgboost.pkl')
    BEST_THRESHOLD = joblib.load('data/warehouse/best_threshold.pkl')

    # Reconstruir pipeline en vez de cargar el pkl (incompatible con StringDtype)
    full_pipeline = Pipeline([
        ('binary',            BinaryEncoder()),
        ('frequency',         FrequencyEncoder()),
        ('ordinal_ext',       OrdinalExtensionEncoder()),
        ('ordinal_equip',     OrdinalEquipamientoEncoder()),
        ('onehot',            NominalOneHotEncoder()),
        ('gasto_relativo',    GastoRelativoEncoder()),
        ('price_standard',    PriceStandard()),
        #('instance_dropper',  InstanceDropper()),
        ('dropper',           ColumnDropper()),
    ])

    # Re-fit con los mismos datos y split que en el notebook original
    customer_data = pd.read_csv('data/lake/customer_data.csv')
    for col in ['Sales_Date', 'FIN_GARANTIA', 'BASE_DATE']:
        if col in customer_data.columns:
            customer_data[col] = pd.to_datetime(customer_data[col], errors='coerce', dayfirst=True, format='%d/%m/%Y')
    for col in customer_data.select_dtypes(include=['string']).columns:
        customer_data[col] = customer_data[col].astype(object)

    # Aplicar definición ghost (igual que en NB2 sección 2.5)
    CUTOFF = pd.Timestamp('2023-12-31')
    antiguedad_dias = (CUTOFF - customer_data['Sales_Date']).dt.days.clip(lower=0)
    ghost_mask = (customer_data['Revisiones'] == 0) & (antiguedad_dias > 400)
    customer_data['Churn_400'] = (
        (customer_data['Churn_400'] == 'Y') | ghost_mask
    ).map({True: 'Y', False: 'N'})

    train_set_raw, _ = train_test_split(customer_data, test_size=0.2, random_state=42)
    full_pipeline.fit(train_set_raw)

    # Compute medians from training set for predictor fixed fields
    median_cols = [
        'Margen_eur_bruto', 'Margen_eur', 'COSTE_VENTA_NO_IMPUESTOS',
        'Km_medio_por_revision', 'km_ultima_revision',
        'ENCUESTA_CLIENTE_ZONA_TALLER', 'DAYS_LAST_SERVICE',
    ]
    train_medianas = {}
    for col in median_cols:
        if col in train_set_raw.columns:
            train_medianas[col] = float(train_set_raw[col].median())
    if 'DAYS_LAST_SERVICE' not in train_medianas:
        train_medianas['DAYS_LAST_SERVICE'] = 255.0

    # --- Align schema ---
    rename_map = {'Lead_compra_1': 'Fue_Lead'}
    new_customers = new_customers.rename(columns={k: v for k, v in rename_map.items() if k in new_customers.columns})

    test_set = pd.read_csv('data/warehouse/test_set.csv')
    for col in ['Sales_Date', 'FIN_GARANTIA', 'BASE_DATE']:
        if col in test_set.columns:
            test_set[col] = pd.to_datetime(test_set[col], errors='coerce')

    if 'Churn_400' not in new_customers.columns:
        new_customers['Churn_400'] = 'N'
    if 'DAYS_LAST_SERVICE' not in new_customers.columns:
        new_customers['DAYS_LAST_SERVICE'] = np.nan

    extra_cols = set(new_customers.columns) - set(test_set.columns)
    if extra_cols:
        new_customers = new_customers.drop(columns=extra_cols)
    missing_cols = set(test_set.columns) - set(new_customers.columns)
    for col in missing_cols:
        new_customers[col] = np.nan
    new_customers = new_customers[test_set.columns]

    # ── Forzar dtypes clásicos (compatibilidad con pipeline) ────
    for col in new_customers.columns:
        if pd.api.types.is_string_dtype(new_customers[col]):
            new_customers[col] = new_customers[col].astype(object)
    for col in test_set.columns:
        if pd.api.types.is_string_dtype(test_set[col]):
            test_set[col] = test_set[col].astype(object)

    # --- Transform & predict (usando el pipeline ya ajustado en train_set_raw) ---
    new_prepared = full_pipeline.fit_transform(new_customers)
    if 'Churn_400' in new_prepared.columns:
        X_new = new_prepared.drop(columns=['Churn_400'])
    else:
        X_new = new_prepared

    new_proba = xgboost_model.predict_proba(X_new)[:, 1]

    # --- Build df ---
    df = new_prepared.copy()
    df['p_churn'] = new_proba
    df['CODE'] = new_customers.loc[new_prepared.index, 'CODE'].values
    df['Modelo_letra'] = new_customers.loc[new_prepared.index, 'Modelo'].values
    df['Id_Producto'] = new_customers.loc[new_prepared.index, 'Id_Producto'].values
    df['PVP_original'] = new_customers.loc[new_prepared.index, 'PVP'].values
    df['Edad_original'] = new_customers.loc[new_prepared.index, 'Edad'].values
    df['ZONA_original'] = new_customers.loc[new_prepared.index, 'ZONA'].values
    df['Origen_original'] = new_customers.loc[new_prepared.index, 'Origen'].values

    # --- CLTV ---
    HORIZONTE = 10
    MARGEN_NETO = 0.62
    ALPHA_AB = 0.07
    ALPHA_RESTO = 0.10
    costes_dict = costes.set_index('Modelo')['Mantenimiento_medio'].to_dict()

    def get_alpha(m):
        return ALPHA_AB if m in ['A', 'B'] else ALPHA_RESTO

    def calc_cltv(row):
        modelo = row['Modelo_letra']
        p = row['p_churn']
        base = costes_dict.get(modelo, 300)
        alpha = get_alpha(modelo)
        cltv = 0
        for n in range(1, HORIZONTE + 1):
            ingreso = base * (1 + alpha) ** n
            benef = ingreso * MARGEN_NETO
            surv = (1 - p) ** n
            cltv += benef * surv
        return cltv

    df['CLTV'] = df.apply(calc_cltv, axis=1)

    # --- Segmentation ---
    df['riesgo'] = pd.qcut(df['p_churn'], q=5, labels=['MUY_BAJO', 'BAJO', 'MEDIO', 'ALTO', 'MUY_ALTO'])
    df['valor'] = pd.qcut(df['CLTV'], q=3, labels=['Bajo', 'Medio', 'Alto'])
    df['segmento'] = df['riesgo'].astype(str) + ' riesgo / ' + df['valor'].astype(str) + ' valor'

    # --- Tasa de respuesta y coste de contacto ---
    TASA_RESPUESTA = {
        'MUY_ALTO': 0.10, 'ALTO': 0.80, 'MEDIO': 0.55, 'BAJO': 0.45, 'MUY_BAJO': 0.40,
    }
    COSTE_CONTACTO = {
        'MUY_ALTO': 0.20, 'ALTO': 0.10, 'MEDIO': 0.50, 'BAJO': 0.50, 'MUY_BAJO': 0.20,
    }
    # .astype(float) necesario: riesgo es Categorical (pd.qcut) y .map() hereda ese dtype
    df['tasa_respuesta'] = df['riesgo'].map(TASA_RESPUESTA).astype(float)
    df['coste_contacto']  = df['riesgo'].map(COSTE_CONTACTO).astype(float)

    # --- Revenue per revision 1 ---
    def coste_rev_n(modelo, n):
        base = costes_dict.get(modelo, 300)
        alpha = get_alpha(modelo)
        return base * (1 + alpha) ** n

    df['ingreso_rev1'] = df['Modelo_letra'].apply(lambda m: coste_rev_n(m, 1))

    # --- Actions (5 risk levels, fixed service package costs) ---
    def asignar_accion(row):
        r, v = row['riesgo'], row['valor']
        if r == 'MUY_ALTO':
            if v in ['Alto', 'Medio']:
                return 'Contacto prioritario', 0, 20, 'Bono 20€'
            else:
                return 'Contacto mínimo', 0, 0, 'Email de seguimiento'
        elif r == 'ALTO':
            if v in ['Alto', 'Medio']:
                return 'Pack Premium VIP', 100, 50, 'Recogida + lavado + neumáticos + bono 50€'
            else:
                return 'Contacto mínimo', 0, 0, 'Email de seguimiento'
        elif r == 'MEDIO':
            if v in ['Alto', 'Medio']:
                return 'Pack Intermedio', 36, 30, 'Regalo + lavado + bono 30€'
            else:
                return 'Seguimiento estándar', 0, 0, 'Recordatorio de revisión'
        else:  # BAJO o MUY_BAJO
            if v == 'Alto':
                return 'Upselling', 0, 0, 'Oferta ext. garantía / seguro batería'
            elif v == 'Medio':
                return 'Mantenimiento', 0, 0, 'Comunicación periódica'
            else:
                return 'Sin acción', 0, 0, '—'

    acciones = df.apply(asignar_accion, axis=1, result_type='expand')
    df['accion']             = acciones[0]
    df['coste_adicional']    = acciones[1]
    df['coste_descuento']    = acciones[2]
    df['descripcion_accion'] = acciones[3]

    # --- ROI simulation ---
    REDUCCION = {
        'Pack Premium VIP':      0.40,
        'Pack Intermedio':       0.20,
        'Contacto prioritario':  0.15,
        'Upselling':             0.10,
        'Seguimiento estándar':  0.05,
        'Contacto mínimo':       0.05,
        'Mantenimiento':         0.05,
        'Sin acción':            0.00,
    }

    def cltv_post(row):
        red = REDUCCION.get(row['accion'], 0)
        p_new = row['p_churn'] * (1 - red)
        modelo = row['Modelo_letra']
        base = costes_dict.get(modelo, 300)
        alpha = get_alpha(modelo)
        cltv = 0
        for n in range(1, HORIZONTE + 1):
            ingreso = base * (1 + alpha) ** n
            benef = ingreso * MARGEN_NETO
            surv = (1 - p_new) ** n
            cltv += benef * surv
        return cltv

    df['CLTV_con_accion'] = df.apply(cltv_post, axis=1)
    df['ganancia_cltv_respondedor'] = df['CLTV_con_accion'] - df['CLTV']

    # 4-component cost breakdown (A5)
    COSTE_MARKETING_PCT = 0.01
    df['coste_efectivo_contacto']  = df['coste_contacto']
    df['coste_efectivo_adicional'] = df['coste_adicional'] * df['tasa_respuesta']
    df['coste_efectivo_descuento'] = df['coste_descuento'] * df['tasa_respuesta']
    df['coste_efectivo_marketing'] = df['ingreso_rev1'] * COSTE_MARKETING_PCT * df['tasa_respuesta']
    df['coste_accion'] = (
        df['coste_efectivo_contacto']
        + df['coste_efectivo_adicional']
        + df['coste_efectivo_descuento']
        + df['coste_efectivo_marketing']
    )
    # TR-weighted gain and ROI
    df['ganancia_cltv'] = df['ganancia_cltv_respondedor'] * df['tasa_respuesta']
    df['ROI'] = np.where(df['coste_accion'] > 0,
                         (df['ganancia_cltv'] - df['coste_accion']) / df['coste_accion'], np.nan)

    # --- Model comparison metrics ---
    rf_model  = joblib.load('data/warehouse/random_forest.pkl')
    lgb_model = joblib.load('data/warehouse/lightgbm.pkl')

    # Apply ghost to test_set and extract y_test before transformation
    test_eval = test_set.copy()
    if 'Sales_Date' in test_eval.columns:
        test_eval['Sales_Date'] = pd.to_datetime(test_eval['Sales_Date'], errors='coerce')
        ant_test = (CUTOFF - test_eval['Sales_Date']).dt.days.clip(lower=0)
        ghost_test = (test_eval['Revisiones'] == 0) & (ant_test > 400)
        test_eval['Churn_400'] = ((test_eval['Churn_400'] == 'Y') | ghost_test).map({True: 'Y', False: 'N'})
    for col in test_eval.columns:
        if pd.api.types.is_string_dtype(test_eval[col]):
            test_eval[col] = test_eval[col].astype(object)

    test_prepared = full_pipeline.fit_transform(test_eval)
    if 'Churn_400' in test_prepared.columns:
        y_test = test_prepared['Churn_400'].values.astype(int)
        X_test_m = test_prepared.drop(columns=['Churn_400'])
    else:
        y_test = None
        X_test_m = test_prepared

    model_metrics = {}
    if y_test is not None:
        for mname, mobj in [('XGBoost', xgboost_model), ('Random Forest', rf_model), ('LightGBM', lgb_model)]:
            proba = mobj.predict_proba(X_test_m)[:, 1]
            pred  = (proba >= BEST_THRESHOLD).astype(int)
            fpr, tpr, _ = roc_curve(y_test, proba)
            model_metrics[mname] = {
                'proba':     proba,
                'auc':       float(auc(fpr, tpr)),
                'precision': float(precision_score(y_test, pred, zero_division=0)),
                'recall':    float(recall_score(y_test, pred, zero_division=0)),
                'f1':        float(f1_score(y_test, pred, zero_division=0)),
                'cm':        confusion_matrix(y_test, pred),
                'fpr':       fpr,
                'tpr':       tpr,
            }
        # feature importances (use XGBoost feature names as reference)
        feat_names = list(X_test_m.columns)
        for mname, mobj in [('XGBoost', xgboost_model), ('Random Forest', rf_model), ('LightGBM', lgb_model)]:
            imp = mobj.feature_importances_
            model_metrics[mname]['feat_imp'] = dict(zip(feat_names, imp))

    # --- Revision table ---
    modelos_unicos = sorted(df['Modelo_letra'].unique())
    rev_rows = []
    for modelo in modelos_unicos:
        base = costes_dict.get(modelo, 300)
        alpha = get_alpha(modelo)
        for n in range(1, HORIZONTE + 1):
            ingreso = base * (1 + alpha) ** n
            rev_rows.append({'Modelo': modelo, 'Revisión': n,
                             'Ingreso bruto': round(ingreso, 2),
                             'Beneficio neto': round(ingreso * MARGEN_NETO, 2)})
    df_rev = pd.DataFrame(rev_rows)

    return df, costes, df_rev, BEST_THRESHOLD, full_pipeline, xgboost_model, train_medianas, model_metrics


# ═══════════════════════════════════════════════════════════════
# LOAD DATA
# ═══════════════════════════════════════════════════════════════
try:
    df, costes, df_rev, BEST_THRESHOLD, full_pipeline, xgboost_model, train_medianas, model_metrics = load_and_compute()
    data_loaded = True
except Exception as e:
    data_loaded = False
    error_msg = f"{type(e).__name__}: {e}\n\n{traceback.format_exc()}"


# ═══════════════════════════════════════════════════════════════
# SIDEBAR
# ═══════════════════════════════════════════════════════════════
with st.sidebar:
    st.markdown("""
    <div style="text-align:center; padding: 15px 0 25px 0;">
        <span style="font-family: 'Orbitron', monospace; font-size: 1.6rem; font-weight: 800;
                     background: linear-gradient(90deg, #C8A951, #E8D5A0, #C8A951);
                     -webkit-background-clip: text; -webkit-text-fill-color: transparent;">
            UAX CHURN
        </span>
        <br>
        <span style="font-family: 'Rajdhani', sans-serif; font-size: 0.75rem;
                     color: #6B7B8D; letter-spacing: 3px; text-transform: uppercase;">
            Customer Intelligence Platform
        </span>
    </div>
    """, unsafe_allow_html=True)

    section_line()

    if data_loaded:
        st.markdown("### Filtros")
        modelos_disponibles = sorted(df['Modelo_letra'].unique())
        modelos_sel = st.multiselect("Modelo de vehículo", modelos_disponibles, default=modelos_disponibles)

        riesgos_sel = st.multiselect("Nivel de riesgo", ['MUY_BAJO', 'BAJO', 'MEDIO', 'ALTO', 'MUY_ALTO'], default=['MUY_BAJO', 'BAJO', 'MEDIO', 'ALTO', 'MUY_ALTO'])

        pvp_range = st.slider("Rango PVP (€)", int(df['PVP_original'].min()), int(df['PVP_original'].max()),
                              (int(df['PVP_original'].min()), int(df['PVP_original'].max())))

        # Filter — se calcula aquí y es accesible fuera del bloque with (Python no crea scope nuevo)
        mask = (
                df['Modelo_letra'].isin(modelos_sel) &
                df['riesgo'].isin(riesgos_sel) &
                df['PVP_original'].between(pvp_range[0], pvp_range[1])
        )
        dff = df[mask].copy()

        section_line()
        st.markdown(f"""
        <div style="text-align:center; padding:10px;">
            <span style="font-family:'Orbitron'; font-size:1.8rem; color:{UAX_GOLD};">{len(dff):,}</span><br>
            <span style="font-family:'Rajdhani'; color:{UAX_GREY}; letter-spacing:2px; font-size:0.8rem;">CLIENTES FILTRADOS</span>
        </div>
        """, unsafe_allow_html=True)


# ═══════════════════════════════════════════════════════════════
# MAIN CONTENT
# ═══════════════════════════════════════════════════════════════
if not data_loaded:
    st.error("Error al cargar datos")
    st.code(error_msg, language="python")
    st.info("Asegúrate de ejecutar el notebook 5-cltv_acciones.ipynb primero, o de que las rutas a los archivos sean correctas.")
    st.stop()

# HEADER
st.markdown("""
<div style="text-align:center; padding: 15px 0 5px 0;">
    <h1>CLTV & Acciones Comerciales</h1>
    <p style="color:#6B7B8D; font-size:1.05rem; letter-spacing:1px;">
        Predicción de Churn · Segmentación · Estrategia de Retención
    </p>
</div>
""", unsafe_allow_html=True)

section_line()

# ─── TABS ──────────────────────────────────────────────────────
tab1, tab2, tab3, tab4, tab5, tab6, tab7 = st.tabs([
    "🏠 Resumen",
    "📈 CLTV Analysis",
    "🎯 Segmentación",
    "⚡ Acciones Comerciales",
    "🔮 Proyección Revisiones",
    "🔬 Modelos",
    "🧮 Predictor de Churn",
])


# ═══════════════════════════════════════════════════════════════
# TAB 1: RESUMEN
# ═══════════════════════════════════════════════════════════════
with tab1:
    st.markdown("## Resumen Ejecutivo")

    # KPI ROW
    c1, c2, c3, c4, c5 = st.columns(5)

    with c1:
        st.markdown(metric_card("Clientes", f"{len(dff):,}", f"de {len(df):,} totales"), unsafe_allow_html=True)
    with c2:
        st.markdown(metric_card("CLTV Medio", f"{dff['CLTV'].mean():,.0f}€",
                                f"Total cartera: {dff['CLTV'].sum():,.0f}€"), unsafe_allow_html=True)
    with c3:
        st.markdown(metric_card("P(Churn) Media", f"{dff['p_churn'].mean():.2%}",
                                f"Threshold: {BEST_THRESHOLD:.2f}"), unsafe_allow_html=True)
    with c4:
        inv = dff['coste_accion'].sum()
        st.markdown(metric_card("Inversión", f"{inv:,.0f}€",
                                f"{(dff['coste_accion']>0).sum()} acciones activas"), unsafe_allow_html=True)
    with c5:
        _inv = dff['coste_accion'].sum()
        _gan = dff['ganancia_cltv'].sum()
        _neto = _gan - _inv
        _roi = _neto / _inv if _inv > 0 else 0
        st.markdown(metric_card("Beneficio Neto", f"{_neto:,.0f}€",
                                f"ROI campaña: {_roi:.1f}x" if _inv > 0 else "—"),
                    unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)
    section_line()

    # Project description
    st.markdown("## Sobre este Proyecto")

    col_desc1, col_desc2 = st.columns(2)

    with col_desc1:
        st.markdown(f"""
        <div class="info-box">
            <div class="info-box-title">El Problema</div>
            <div class="info-box-text">
                Un concesionario de vehículos eléctricos e híbridos necesita anticiparse a la fuga de clientes
                (churn) antes de que ocurra. Perder un cliente supone no solo la pérdida inmediata de ingresos
                por revisiones, sino también el abandono del vínculo a 10 años vista que representa el CLTV.
                <br><br>
                El objetivo es identificar qué clientes tienen mayor probabilidad de no volver a hacer
                una revisión en el concesionario, y actuar antes de que se vayan.
            </div>
        </div>
        """, unsafe_allow_html=True)

        st.markdown(f"""
        <div class="info-box">
            <div class="info-box-title">Los Datos</div>
            <div class="info-box-text">
                <b>customer_data.csv</b> — histórico de clientes con variables de vehículo (modelo, PVP, kW,
                combustible), contrato (garantía, financiación, seguro batería), geografía (zona, provincia)
                y comportamiento (días desde última revisión, quejas, encuestas).
                <br><br>
                <b>nuevos_clientes.csv</b> — cartera actual sobre la que se aplican las predicciones
                y se priorizan las acciones comerciales.
                <br><br>
                <b>costes.csv</b> — coste medio de mantenimiento por modelo, base del cálculo CLTV.
            </div>
        </div>
        """, unsafe_allow_html=True)

    with col_desc2:
        st.markdown(f"""
        <div class="info-box">
            <div class="info-box-title">La Metodología</div>
            <div class="info-box-text">
                <b>1. Preparación de datos</b> — limpieza, codificación de variables categóricas (binaria,
                frecuencia, ordinal, one-hot) y estandarización de precios mediante un pipeline de
                Scikit-learn reproducible.
                <br><br>
                <b>2. Modelado</b> — se compararon Regresión Logística, Árbol de Decisión, Random Forest
                y <b>XGBoost</b>. El mejor modelo fue XGBoost con un AUC-ROC de <b>0.83</b>.
                <br><br>
                <b>3. Threshold óptimo</b> — se ajustó el umbral de clasificación a <b>{BEST_THRESHOLD:.2f}</b>
                maximizando la métrica de negocio (F-beta), priorizando la detección de churners reales.
                <br><br>
                <b>4. CLTV</b> — se proyecta el valor futuro de cada cliente a 10 revisiones usando
                una tasa de supervivencia basada en su probabilidad de churn individual.
                <br><br>
                <b>5. Acciones</b> — reglas de negocio que asignan la acción óptima según la combinación
                de riesgo de churn y valor CLTV, con estimación de ROI esperado.
            </div>
        </div>
        """, unsafe_allow_html=True)

        st.markdown(f"""
        <div class="info-box">
            <div class="info-box-title">Cómo Usar el Dashboard</div>
            <div class="info-box-text">
                Usa los <b>filtros del sidebar</b> para segmentar la cartera por modelo, nivel de riesgo
                y rango de PVP. Los KPIs y gráficos se actualizan en tiempo real.<br><br>
                La pestaña <b>Predictor de Churn</b> permite introducir los datos de un cliente nuevo
                y obtener su probabilidad de fuga instantáneamente.
            </div>
        </div>
        """, unsafe_allow_html=True)


# ═══════════════════════════════════════════════════════════════
# TAB 2: CLTV ANALYSIS
# ═══════════════════════════════════════════════════════════════
with tab2:
    st.markdown("## Análisis del Customer Lifetime Value")

    col1, col2 = st.columns(2)

    with col1:
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
        st.plotly_chart(fig, width='stretch')

    with col2:
        cltv_modelo = dff.groupby('Modelo_letra')['CLTV'].mean().sort_values()
        fig = go.Figure(go.Bar(x=cltv_modelo.values, y=cltv_modelo.index,
                               orientation='h', marker_color=UAX_GOLD,
                               marker_line_color=UAX_NAVY, marker_line_width=1))
        fig.update_layout(title="CLTV medio por modelo", xaxis_title="CLTV (€)")
        plotly_layout(fig)
        st.plotly_chart(fig, width='stretch')

    # Scatter
    fig = px.scatter(dff, x='p_churn', y='CLTV', color='PVP_original',
                     color_continuous_scale=[[0, UAX_NAVY], [0.5, UAX_ACCENT], [1, UAX_GOLD]],
                     opacity=0.5, hover_data=['CODE', 'Modelo_letra', 'accion'])
    fig.update_layout(title="P(Churn) vs CLTV — coloreado por PVP",
                      xaxis_title="P(Churn)", yaxis_title="CLTV (€)",
                      coloraxis_colorbar_title="PVP (€)")
    plotly_layout(fig, height=500)
    st.plotly_chart(fig, width='stretch')


# ═══════════════════════════════════════════════════════════════
# TAB 3: SEGMENTATION
# ═══════════════════════════════════════════════════════════════
with tab3:
    st.markdown("## Matriz de Segmentación: Riesgo × Valor")

    col1, col2 = st.columns(2)

    riesgo_order = ['MUY_ALTO', 'ALTO', 'MEDIO', 'BAJO', 'MUY_BAJO']
    valor_order = ['Bajo', 'Medio', 'Alto']

    with col1:
        pivot_count = dff.groupby(['riesgo', 'valor'], observed=False).size().unstack(fill_value=0)
        pivot_count = pivot_count.reindex(index=riesgo_order, columns=valor_order, fill_value=0)
        fig = go.Figure(go.Heatmap(
            z=pivot_count.values, x=valor_order, y=riesgo_order,
            text=[[f"{v:,}" for v in row] for row in pivot_count.values],
            texttemplate="%{text}", textfont=dict(size=18, family='Orbitron'),
            colorscale=[[0, UAX_BG], [1, UAX_NAVY]],
            showscale=False, hoverinfo='skip'
        ))
        fig.update_layout(title="Nº de clientes por segmento",
                          xaxis_title="Valor (CLTV)", yaxis_title="Riesgo (Churn)")
        plotly_layout(fig, height=380)
        st.plotly_chart(fig, width='stretch')

    with col2:
        pivot_cltv = dff.groupby(['riesgo', 'valor'], observed=False)['CLTV'].mean().unstack(fill_value=0)
        pivot_cltv = pivot_cltv.reindex(index=riesgo_order, columns=valor_order, fill_value=0)
        fig = go.Figure(go.Heatmap(
            z=pivot_cltv.values, x=valor_order, y=riesgo_order,
            text=[[f"{v:,.0f}€" for v in row] for row in pivot_cltv.values],
            texttemplate="%{text}", textfont=dict(size=16, family='Orbitron'),
            colorscale=[[0, UAX_BG], [0.5, UAX_ACCENT], [1, UAX_GOLD]],
            showscale=False, hoverinfo='skip'
        ))
        fig.update_layout(title="CLTV medio (€) por segmento",
                          xaxis_title="Valor (CLTV)", yaxis_title="Riesgo (Churn)")
        plotly_layout(fig, height=380)
        st.plotly_chart(fig, width='stretch')

    # Segment distribution treemap
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
    st.plotly_chart(fig, width='stretch')


# ═══════════════════════════════════════════════════════════════
# TAB 4: ACCIONES COMERCIALES + ROI
# ═══════════════════════════════════════════════════════════════
with tab4:
    st.markdown("## Acciones Comerciales")

    resumen = dff.groupby('accion').agg(
        clientes=('CODE', 'count'),
        coste_total=('coste_accion', 'sum'),
        cltv_medio=('CLTV', 'mean'),
        p_churn_medio=('p_churn', 'mean'),
    ).sort_values('coste_total', ascending=False).reset_index()

    col1, col2 = st.columns(2)

    with col1:
        fig = go.Figure(go.Bar(
            y=resumen['accion'], x=resumen['coste_total'],
            orientation='h', marker_color=UAX_GOLD,
            marker_line_color=UAX_NAVY, marker_line_width=1,
            text=[f"{v:,.0f}€" for v in resumen['coste_total']],
            textposition='outside', textfont=dict(color=UAX_GOLD, size=12)
        ))
        fig.update_layout(title="Inversión total por acción", xaxis_title="Coste total (€)")
        plotly_layout(fig, height=400)
        st.plotly_chart(fig, width='stretch')

    with col2:
        fig = go.Figure(go.Bar(
            y=resumen['accion'], x=resumen['clientes'],
            orientation='h', marker_color=UAX_NAVY,
            marker_line_color=UAX_GOLD, marker_line_width=1,
            text=[f"{v:,}" for v in resumen['clientes']],
            textposition='outside', textfont=dict(color=UAX_TEXT, size=12)
        ))
        fig.update_layout(title="Clientes por acción", xaxis_title="Nº clientes")
        plotly_layout(fig, height=400)
        st.plotly_chart(fig, width='stretch')

    # Action rules table
    st.markdown("### Reglas de negocio")
    rules = pd.DataFrame({
        'Riesgo': ['MUY_ALTO', 'MUY_ALTO', 'ALTO', 'ALTO', 'MEDIO', 'MEDIO',
                   'BAJO/MUY_BAJO', 'BAJO/MUY_BAJO', 'BAJO/MUY_BAJO'],
        'Valor': ['Alto/Medio', 'Bajo', 'Alto/Medio', 'Bajo', 'Alto/Medio', 'Bajo',
                  'Alto', 'Medio', 'Bajo'],
        'Acción': ['Contacto prioritario', 'Contacto mínimo',
                   'Pack Premium VIP', 'Contacto mínimo',
                   'Pack Intermedio', 'Seguimiento estándar',
                   'Upselling', 'Mantenimiento', 'Sin acción'],
        'Servicios': ['Bono 20€', 'Email',
                      'Recogida+lavado+neumáticos+bono 50€', 'Email',
                      'Regalo+lavado+bono 30€', 'Recordatorio',
                      'Ext. garantía / seguro batería', 'Comunicación periódica', '—'],
        'Coste/respondedor': ['20€', '0€', '150€', '0€', '66€', '0€', '0€', '0€', '0€'],
        'TR': ['10%', '10%', '80%', '80%', '55%', '55%', '45/40%', '45/40%', '45/40%'],
        'Δ churn': ['−15%', '−5%', '−40%', '−5%', '−20%', '−5%', '−10%', '−5%', '0%'],
    })
    st.dataframe(rules, width='stretch', hide_index=True)

    section_line()
    st.markdown("## Simulación Económica — ROI")

    sim = dff.groupby('accion').agg(
        clientes=('CODE', 'count'),
        inversion=('coste_accion', 'sum'),
        ganancia_cltv=('ganancia_cltv', 'sum'),
    ).reset_index()
    sim['beneficio_neto'] = sim['ganancia_cltv'] - sim['inversion']
    sim = sim.sort_values('beneficio_neto', ascending=False)

    col1, col2 = st.columns(2)

    with col1:
        colors_bn = [UAX_GREEN if v > 0 else UAX_RED for v in sim['beneficio_neto']]
        fig = go.Figure(go.Bar(
            y=sim['accion'], x=sim['beneficio_neto'],
            orientation='h', marker_color=colors_bn,
            text=[f"{v:,.0f}€" for v in sim['beneficio_neto']],
            textposition='outside', textfont=dict(size=11)
        ))
        fig.add_vline(x=0, line_color=UAX_GREY, line_width=1)
        fig.update_layout(title="Beneficio neto por acción", xaxis_title="Beneficio neto (€)")
        plotly_layout(fig, height=400)
        st.plotly_chart(fig, width='stretch')

    with col2:
        roi_data = dff[dff['coste_accion'] > 0].groupby('accion')['ROI'].mean().sort_values()
        fig = go.Figure(go.Bar(
            y=roi_data.index, x=roi_data.values,
            orientation='h', marker_color=UAX_ACCENT,
            text=[f"{v:.1f}x" for v in roi_data.values],
            textposition='outside', textfont=dict(color=UAX_GOLD, size=12)
        ))
        fig.add_vline(x=0, line_color=UAX_GREY, line_width=1)
        fig.update_layout(title="ROI medio por acción (solo con inversión)", xaxis_title="ROI (x)")
        plotly_layout(fig, height=400)
        st.plotly_chart(fig, width='stretch')

    # Desglose económico global
    section_line()
    st.markdown("### Desglose económico global")

    total_cltv_sin = dff['CLTV'].sum()
    total_ganancia = dff['ganancia_cltv'].sum()
    total_inversion = dff['coste_accion'].sum()
    total_neto = total_ganancia - total_inversion
    roi_global = total_neto / total_inversion if total_inversion > 0 else 0
    pct_mejora = total_ganancia / total_cltv_sin * 100 if total_cltv_sin > 0 else 0

    # KPI cards — contexto de cartera
    kc1, kc2, kc3, kc4 = st.columns(4)
    with kc1:
        st.markdown(metric_card("CLTV Cartera", f"{total_cltv_sin/1e6:.2f}M€", "sin acciones"), unsafe_allow_html=True)
    with kc2:
        st.markdown(metric_card("Ganancia CLTV", f"+{total_ganancia:,.0f}€", f"+{pct_mejora:.2f}% sobre cartera"), unsafe_allow_html=True)
    with kc3:
        st.markdown(metric_card("Inversión total", f"{total_inversion:,.0f}€", f"{(dff['coste_accion']>0).sum()} clientes con acción"), unsafe_allow_html=True)
    with kc4:
        st.markdown(metric_card("Beneficio neto", f"{total_neto:,.0f}€", f"ROI global: {roi_global:.1f}x"), unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)

    # Waterfall solo con la parte incremental (escala legible)
    col_wf, col_bar = st.columns([3, 2])

    with col_wf:
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
        fig.update_layout(
            title="Plan de acciones — flujo incremental",
            yaxis_title="€",
            yaxis=dict(tickformat=",.0f"),
        )
        plotly_layout(fig, height=420)
        st.plotly_chart(fig, width='stretch')

    with col_bar:
        # Comparativa antes/después por segmento de acción (solo los que tienen inversión)
        sim_inv = sim[sim['inversion'] > 0].copy()
        sim_inv['roi_label'] = sim_inv['beneficio_neto'].apply(lambda v: f"{v:,.0f}€")
        colors_sim = [UAX_GREEN if v > 0 else UAX_RED for v in sim_inv['beneficio_neto']]

        fig2 = go.Figure()
        fig2.add_trace(go.Bar(
            name="Inversión", x=sim_inv['accion'], y=sim_inv['inversion'],
            marker_color=UAX_RED, opacity=0.75,
            text=[f"{v:,.0f}€" for v in sim_inv['inversion']],
            textposition='inside', textfont=dict(size=10, color=UAX_TEXT),
        ))
        fig2.add_trace(go.Bar(
            name="Ganancia CLTV", x=sim_inv['accion'], y=sim_inv['ganancia_cltv'],
            marker_color=UAX_GREEN, opacity=0.75,
            text=[f"{v:,.0f}€" for v in sim_inv['ganancia_cltv']],
            textposition='inside', textfont=dict(size=10, color=UAX_TEXT),
        ))
        fig2.update_layout(
            title="Inversión vs Ganancia por acción",
            barmode='group',
            yaxis_title="€",
            yaxis=dict(tickformat=",.0f"),
            xaxis_tickangle=-25,
        )
        plotly_layout(fig2, height=420)
        st.plotly_chart(fig2, width='stretch')


# ═══════════════════════════════════════════════════════════════
# TAB 5: REVISION PROJECTION
# ═══════════════════════════════════════════════════════════════
with tab5:
    st.markdown("## Proyección de Revisiones Futuras")

    modelos_rev = st.multiselect("Modelos a visualizar", sorted(df_rev['Modelo'].unique()),
                                 default=sorted(df_rev['Modelo'].unique())[:6],
                                 key="rev_modelos")

    df_rev_f = df_rev[df_rev['Modelo'].isin(modelos_rev)]

    col_colors = {m: c for m, c in zip(sorted(df_rev['Modelo'].unique()),
                                       px.colors.sample_colorscale('Viridis', np.linspace(0.15, 0.95, len(df_rev['Modelo'].unique()))))}

    fig = go.Figure()
    for modelo in modelos_rev:
        subset = df_rev_f[df_rev_f['Modelo'] == modelo]
        fig.add_trace(go.Scatter(
            x=subset['Revisión'], y=subset['Beneficio neto'],
            mode='lines+markers', name=f'Modelo {modelo}',
            line=dict(width=2.5), marker=dict(size=7)
        ))

    fig.add_vline(x=5, line_dash="dash", line_color=UAX_RED, line_width=2,
                  annotation_text="n≥5: elegible dto 2º vehículo",
                  annotation_font_color=UAX_RED)
    fig.update_layout(title="Evolución del beneficio neto por revisión",
                      xaxis_title="Nº de revisión", yaxis_title="Beneficio neto (€)",
                      xaxis=dict(dtick=1))
    plotly_layout(fig, height=500)
    st.plotly_chart(fig, width='stretch')

    # Tabla
    st.markdown("### Tabla de beneficios por revisión y modelo")
    pivot = df_rev_f.pivot_table(index='Modelo', columns='Revisión', values='Beneficio neto').round(0)
    st.dataframe(pivot.style.format("{:,.0f}€").background_gradient(
        cmap='YlOrRd', axis=1), width='stretch')

    # Dto 2o vehiculo note
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


# ═══════════════════════════════════════════════════════════════
# TAB 6: COMPARACIÓN DE MODELOS
# ═══════════════════════════════════════════════════════════════
with tab6:
    st.markdown("## Comparación de Modelos")

    if not model_metrics:
        st.warning("No hay métricas de modelos disponibles. Verifica que los archivos .pkl estén en data/warehouse/.")
    else:
        MODEL_COLORS = {'XGBoost': UAX_GOLD, 'Random Forest': UAX_GREEN, 'LightGBM': '#6EC6FF'}

        # ── Selector de modelo + 4 KPI cards ──────────────────
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

        # ── Confusion matrix + comparison table ───────────────
        col_cm, col_tbl = st.columns([1, 1])

        with col_cm:
            st.markdown(f"### Matriz de Confusión — {sel_model}")
            cm = m['cm']
            fig_cm = go.Figure(go.Heatmap(
                z=cm,
                x=['Pred No Churn', 'Pred Churn'],
                y=['Real No Churn', 'Real Churn'],
                text=[[str(v) for v in row] for row in cm],
                texttemplate="%{text}",
                textfont=dict(size=20, color=UAX_TEXT),
                colorscale=[[0, UAX_CARD], [1, UAX_GOLD]],
                showscale=False,
            ))
            plotly_layout(fig_cm, height=320)
            fig_cm.update_layout(
                xaxis=dict(side='bottom'),
                margin=dict(l=40, r=20, t=40, b=40),
            )
            st.plotly_chart(fig_cm, width='stretch')

        with col_tbl:
            st.markdown("### Tabla Comparativa")
            rows = []
            for mn, mm in model_metrics.items():
                rows.append({
                    'Modelo': mn,
                    'AUC':       f"{mm['auc']:.4f}",
                    'Precision': f"{mm['precision']:.4f}",
                    'Recall':    f"{mm['recall']:.4f}",
                    'F1':        f"{mm['f1']:.4f}",
                })
            st.dataframe(pd.DataFrame(rows).set_index('Modelo'), width='stretch')

        section_line()

        # ── ROC curves (all 3 models) ──────────────────────────
        st.markdown("### Curvas ROC")
        fig_roc = go.Figure()
        fig_roc.add_shape(type='line', x0=0, y0=0, x1=1, y1=1,
                          line=dict(dash='dash', color=UAX_GREY, width=1))
        for mn, mm in model_metrics.items():
            fig_roc.add_trace(go.Scatter(
                x=mm['fpr'], y=mm['tpr'],
                name=f"{mn} (AUC={mm['auc']:.3f})",
                mode='lines',
                line=dict(width=2.5, color=MODEL_COLORS.get(mn, UAX_GOLD)),
            ))
        fig_roc.update_layout(
            xaxis_title="Tasa de Falsos Positivos",
            yaxis_title="Tasa de Verdaderos Positivos",
        )
        plotly_layout(fig_roc, height=420)
        st.plotly_chart(fig_roc, width='stretch')

        section_line()

        # ── Feature importance + bimodal distribution ──────────
        col_fi, col_dist = st.columns([1, 1])

        with col_fi:
            st.markdown(f"### Feature Importance — {sel_model}")
            fi = m.get('feat_imp', {})
            if fi:
                fi_series = pd.Series(fi).sort_values(ascending=True).tail(10)
                fig_fi = go.Figure(go.Bar(
                    x=fi_series.values,
                    y=fi_series.index,
                    orientation='h',
                    marker_color=UAX_GOLD,
                    opacity=0.85,
                ))
                fig_fi.update_layout(xaxis_title="Importancia")
                plotly_layout(fig_fi, height=380)
                st.plotly_chart(fig_fi, width='stretch')

        with col_dist:
            st.markdown(f"### Distribución de Probabilidades — {sel_model}")
            fig_hist = go.Figure()
            fig_hist.add_trace(go.Histogram(
                x=m['proba'],
                nbinsx=40,
                marker_color=UAX_GOLD,
                opacity=0.75,
                name=sel_model,
            ))
            fig_hist.update_layout(
                xaxis_title="P(Churn)",
                yaxis_title="Frecuencia",
                bargap=0.05,
            )
            plotly_layout(fig_hist, height=380)
            st.plotly_chart(fig_hist, width='stretch')


# ═══════════════════════════════════════════════════════════════
# TAB 7: PREDICTOR DE CHURN
# ═══════════════════════════════════════════════════════════════
with tab7:
    st.markdown("## Predictor Individual de Churn")
    st.markdown(f"""
    <div class="info-box">
        <div class="info-box-text">
            Introduce los datos de un cliente para obtener su probabilidad de churn en tiempo real.
            El modelo utilizado es <b>XGBoost</b> con threshold óptimo de <b>{BEST_THRESHOLD:.2f}</b>
            (AUC-ROC 0.83). Los campos marcados con * son obligatorios.
        </div>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)

    with st.form("predictor_form"):
        col_v, col_c, col_cli = st.columns(3)

        # ── Columna 1: Vehículo ────────────────────────────────
        with col_v:
            st.markdown(f"<div style='font-family:Orbitron; color:{UAX_GOLD}; font-size:0.85rem; letter-spacing:2px; text-transform:uppercase; margin-bottom:12px;'>Vehículo</div>", unsafe_allow_html=True)

            modelo_input = st.selectbox("Modelo *", ['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J', 'K'])
            pvp_input = st.number_input("PVP (€) *", min_value=10528, max_value=37970, value=22000, step=500)
            kw_input = st.number_input("Potencia (kW) *", min_value=48, max_value=193, value=100, step=5)
            fuel_input = st.selectbox("Combustible *", ['ELÉCTRICO', 'HÍBRIDO'])
            trans_input = st.selectbox("Transmisión *", ['A', 'M'])
            carroceria_input = st.selectbox("Tipo carrocería *", ['TIPO1', 'TIPO2', 'TIPO3', 'TIPO4', 'TIPO5', 'TIPO6', 'TIPO7', 'TIPO8'])
            equip_input = st.selectbox("Equipamiento *", ['Low', 'Mid', 'Mid-High', 'High'])

        # ── Columna 2: Contrato y garantía ────────────────────
        with col_c:
            st.markdown(f"<div style='font-family:Orbitron; color:{UAX_GOLD}; font-size:0.85rem; letter-spacing:2px; text-transform:uppercase; margin-bottom:12px;'>Contrato y Garantía</div>", unsafe_allow_html=True)

            garantia_input = st.radio("En garantía *", ['SI', 'NO'], horizontal=True)
            ext_garantia_input = st.selectbox("Extensión garantía *", ['NO', 'SI', 'SI, Financiera'])
            seguro_bat_input = st.radio("Seguro batería largo plazo *", ['SI', 'NO'], horizontal=True)
            mant_gratis_input = st.slider("Mantenimientos gratuitos", min_value=0, max_value=4, value=0)
            forma_pago_input = st.selectbox("Forma de pago *", ['Contado', 'Financiera Marca', 'Otros', 'Prestamo Bancario'])
            motivo_venta_input = st.selectbox("Motivo de venta *", ['Particular', 'No Particular'])

        # ── Columna 3: Cliente y geografía ────────────────────
        with col_cli:
            st.markdown(f"<div style='font-family:Orbitron; color:{UAX_GOLD}; font-size:0.85rem; letter-spacing:2px; text-transform:uppercase; margin-bottom:12px;'>Cliente y Geografía</div>", unsafe_allow_html=True)

            edad_input = st.number_input("Edad *", min_value=20, max_value=78, value=40, step=1)
            renta_input = st.number_input("Renta media estimada (€) *", min_value=0, max_value=37777, value=18000, step=500)
            zona_input = st.selectbox("Zona *", ['CENTRO', 'ESTE', 'NORTE', 'SUR'])
            prov_input = st.selectbox("Provincia *", ['BARCELONA', 'BILBAO', 'LA CORUÑA', 'MADRID', 'MALAGA', 'SEVILLA', 'VALENCIA'])
            origen_input = st.selectbox("Origen *", ['Internet', 'Tienda'])
            queja_input = st.radio("¿Ha presentado quejas? *", ['NO', 'SI'], horizontal=True)
            lead_compra_input = st.radio("¿Lead de compra? *", ['No', 'Sí'], horizontal=True)
            fue_lead_input = st.radio("¿Fue lead? *", ['No', 'Sí'], horizontal=True)

        st.markdown("<br>", unsafe_allow_html=True)
        submitted = st.form_submit_button(
            "CALCULAR PROBABILIDAD DE CHURN",
            use_container_width=True,
        )

    # ── Resultado ─────────────────────────────────────────────
    if submitted:
        now = pd.Timestamp.now()

        # Campos fijos
        days_svc   = train_medianas.get('DAYS_LAST_SERVICE', 255.0)
        km_rev     = train_medianas.get('Km_medio_por_revision', np.nan)
        km_ultima  = train_medianas.get('km_ultima_revision', np.nan)
        encuesta   = train_medianas.get('ENCUESTA_CLIENTE_ZONA_TALLER', np.nan)
        margen_bruto = train_medianas.get('Margen_eur_bruto', np.nan)
        margen_eur   = train_medianas.get('Margen_eur', np.nan)
        coste_venta  = train_medianas.get('COSTE_VENTA_NO_IMPUESTOS', np.nan)

        row = {
            # Inputs del usuario
            'Modelo':                     modelo_input,
            'PVP':                        float(pvp_input),
            'Kw':                         float(kw_input),
            'Fuel':                       fuel_input,
            'TRANSMISION_ID':             trans_input,
            'TIPO_CARROCERIA':            carroceria_input,
            'Equipamiento':               equip_input,
            'EN_GARANTIA':                garantia_input,
            'EXTENSION_GARANTIA':         ext_garantia_input,
            'SEGURO_BATERIA_LARGO_PLAZO': seguro_bat_input,
            'MANTENIMIENTO_GRATUITO':     float(mant_gratis_input),
            'FORMA_PAGO':                 forma_pago_input,
            'MOTIVO_VENTA':               motivo_venta_input,
            'Edad':                       float(edad_input),
            'RENTA_MEDIA_ESTIMADA':       float(renta_input),
            'ZONA':                       zona_input,
            'PROV_DESC':                  prov_input,
            'Origen':                     origen_input,
            'QUEJA':                      queja_input,
            'Lead_compra':                1 if lead_compra_input == 'Sí' else 0,
            'Fue_Lead':                   1 if fue_lead_input == 'Sí' else 0,
            # Campos fijos / dummy
            'Churn_400':                  'N',
            'CODE':                       'PRED_001',
            'Id_Producto':                'PROD_001',
            'Customer_ID':                'CUST_001',
            'Sales_Date':                 now,
            'FIN_GARANTIA':               now,
            'BASE_DATE':                  now,
            'CODIGO_POSTAL':              '28001',
            'TIENDA_DESC':                'MADRID CENTRO',
            'Margen_eur_bruto':           margen_bruto,
            'Margen_eur':                 margen_eur,
            'COSTE_VENTA_NO_IMPUESTOS':   coste_venta,
            'Km_medio_por_revision':      km_rev,
            'km_ultima_revision':         km_ultima,
            'ENCUESTA_CLIENTE_ZONA_TALLER': encuesta,
            'DAYS_LAST_SERVICE':          days_svc,
            'Revisiones':                 1,
            'STATUS_SOCIAL':              'A',
            'GENERO':                     'M',
        }

        df_pred = pd.DataFrame([row])

        # Forzar dtypes object en columnas string
        for col in df_pred.select_dtypes(include=['string']).columns:
            df_pred[col] = df_pred[col].astype(object)

        try:
            X_pred = full_pipeline.fit_transform(df_pred)
            if 'Churn_400' in X_pred.columns:
                X_pred = X_pred.drop(columns=['Churn_400'])

            # Alinear columnas con las que espera XGBoost
            expected_cols = xgboost_model.get_booster().feature_names
            for c in expected_cols:
                if c not in X_pred.columns:
                    X_pred[c] = 0
            X_pred = X_pred[expected_cols]

            prob = float(xgboost_model.predict_proba(X_pred)[:, 1][0])
            pred_ok = True
        except Exception as pred_err:
            pred_ok = False
            pred_error_msg = f"{type(pred_err).__name__}: {pred_err}\n\n{traceback.format_exc()}"

        if pred_ok:
            es_churn = prob >= BEST_THRESHOLD
            pct = prob * 100

            # Clasificación de riesgo
            if prob < 0.15:
                nivel = "MUY BAJO RIESGO"
                nivel_color = UAX_GREEN
                accion_msg = "Cliente muy estable. Candidato ideal para upselling o pack 5 revisiones con bono 2º vehículo."
            elif prob < 0.28:
                nivel = "BAJO RIESGO"
                nivel_color = UAX_GREEN
                accion_msg = "Cliente estable. Considera upselling (ext. garantía / seguro batería) o mantenimiento periódico."
            elif prob < 0.42:
                nivel = "RIESGO MEDIO"
                nivel_color = UAX_GOLD
                accion_msg = "Atención proactiva. Pack Intermedio: regalo + lavado + bono 30€ (TR 55%)."
            elif prob < 0.60:
                nivel = "ALTO RIESGO"
                nivel_color = UAX_RED
                accion_msg = "Acción prioritaria. Pack Premium VIP: recogida + lavado + neumáticos + bono 50€ (TR 80%)."
            else:
                nivel = "MUY ALTO RIESGO"
                nivel_color = UAX_RED
                accion_msg = "Cliente muy difícil de recuperar (TR 10%). Contacto prioritario con bono 20€."

            # Gauge Plotly
            fig_gauge = go.Figure(go.Indicator(
                mode="gauge+number",
                value=pct,
                number={'suffix': '%', 'font': {'family': 'Orbitron', 'color': UAX_GOLD, 'size': 36}},
                title={'text': "Probabilidad de Churn", 'font': {'family': 'Orbitron', 'color': UAX_TEXT, 'size': 14}},
                gauge={
                    'axis': {'range': [0, 100], 'tickcolor': UAX_GREY,
                             'tickfont': {'family': 'Rajdhani', 'color': UAX_GREY}},
                    'bar': {'color': nivel_color, 'thickness': 0.25},
                    'bgcolor': UAX_CARD2,
                    'borderwidth': 1,
                    'bordercolor': UAX_GREY,
                    'steps': [
                        {'range': [0, 15], 'color': 'rgba(39,174,96,0.15)'},
                        {'range': [15, 35], 'color': 'rgba(200,169,81,0.15)'},
                        {'range': [35, 100], 'color': 'rgba(192,57,43,0.15)'},
                    ],
                    'threshold': {
                        'line': {'color': UAX_ACCENT, 'width': 3},
                        'thickness': 0.75,
                        'value': BEST_THRESHOLD * 100,
                    },
                }
            ))
            fig_gauge.update_layout(
                paper_bgcolor=UAX_CARD,
                font=dict(family='Rajdhani', color=UAX_TEXT),
                height=320,
                margin=dict(l=30, r=30, t=60, b=20),
            )

            # CLTV estimado
            costes_dict = costes.set_index('Modelo')['Mantenimiento_medio'].to_dict()
            HORIZONTE = 10
            MARGEN_NETO = 0.62
            alpha = 0.07 if modelo_input in ['A', 'B'] else 0.10
            base = costes_dict.get(modelo_input, 300)
            cltv_pred = sum(
                base * (1 + alpha) ** n * MARGEN_NETO * (1 - prob) ** n
                for n in range(1, HORIZONTE + 1)
            )

            # Layout resultado
            section_line()
            st.markdown("## Resultado de la Predicción")

            res_col1, res_col2 = st.columns([1, 1])

            with res_col1:
                st.plotly_chart(fig_gauge, width='stretch')

            with res_col2:
                churn_label = "CHURN" if es_churn else "NO CHURN"
                churn_color = UAX_RED if es_churn else UAX_GREEN
                st.markdown(f"""
                <div style="margin-top:20px;">
                    <div class="result-badge" style="background:rgba({','.join(str(int(churn_color[i:i+2], 16)) for i in (1,3,5))},0.15);
                                                     border:2px solid {churn_color}; color:{churn_color}; margin-bottom:16px;">
                        {churn_label}
                    </div>
                    <div class="result-badge" style="background:rgba({','.join(str(int(nivel_color[i:i+2], 16)) for i in (1,3,5))},0.10);
                                                     border:1px solid {nivel_color}; color:{nivel_color}; margin-bottom:16px; font-size:0.9rem;">
                        {nivel}
                    </div>
                </div>
                """, unsafe_allow_html=True)

                st.markdown(f"""
                <div class="info-box" style="margin-top:8px;">
                    <div class="info-box-title">Detalle</div>
                    <div class="info-box-text">
                        <b>P(Churn):</b> {prob:.4f} ({pct:.1f}%)<br>
                        <b>Threshold:</b> {BEST_THRESHOLD:.2f} ({BEST_THRESHOLD*100:.0f}%)<br>
                        <b>CLTV estimado:</b> {cltv_pred:,.0f}€<br>
                        <b>Modelo XGBoost</b> | AUC-ROC 0.83
                    </div>
                </div>
                """, unsafe_allow_html=True)

                st.markdown(f"""
                <div class="info-box" style="margin-top:8px; border-color:rgba({','.join(str(int(nivel_color[i:i+2], 16)) for i in (1,3,5))},0.35);">
                    <div class="info-box-title" style="color:{nivel_color};">Acción Recomendada</div>
                    <div class="info-box-text">{accion_msg}</div>
                </div>
                """, unsafe_allow_html=True)

            # ── Contextual scatter — portfolio positioning ─────
            section_line()
            st.markdown("### Posición en la Cartera")
            RIESGO_COLORS = {
                'MUY_BAJO': UAX_GREEN,
                'BAJO':     '#80D080',
                'MEDIO':    UAX_GOLD,
                'ALTO':     UAX_RED,
                'MUY_ALTO': '#FF4444',
            }
            fig_scatter = go.Figure()
            for seg in ['MUY_BAJO', 'BAJO', 'MEDIO', 'ALTO', 'MUY_ALTO']:
                sub = df[df['riesgo'] == seg]
                fig_scatter.add_trace(go.Scatter(
                    x=sub['p_churn'], y=sub['PVP_original'],
                    mode='markers',
                    name=seg,
                    marker=dict(size=6, color=RIESGO_COLORS[seg], opacity=0.55),
                ))
            fig_scatter.add_trace(go.Scatter(
                x=[prob], y=[pvp_input],
                mode='markers',
                name='Este cliente',
                marker=dict(size=16, color='#FFFFFF', symbol='star',
                            line=dict(color=UAX_GOLD, width=2)),
                showlegend=True,
            ))
            fig_scatter.update_layout(
                xaxis_title="P(Churn)",
                yaxis_title="PVP (€)",
                xaxis=dict(tickformat='.0%'),
                yaxis=dict(tickformat=",.0f"),
            )
            plotly_layout(fig_scatter, height=400)
            st.plotly_chart(fig_scatter, width='stretch')

        else:
            st.error("Error al ejecutar la predicción. Verifica que el pipeline sea compatible con los campos introducidos.")
            st.code(pred_error_msg, language="python")


# ═══════════════════════════════════════════════════════════════
# FOOTER
# ═══════════════════════════════════════════════════════════════
section_line()
st.markdown(f"""
<div style="text-align:center; padding:20px 0;">
    <span style="font-family:'Orbitron'; font-size:0.7rem; color:{UAX_GREY}; letter-spacing:3px;">
        UAX CHURN INTELLIGENCE · UNIVERSIDAD ALFONSO X EL SABIO · 2025
    </span>
</div>
""", unsafe_allow_html=True)
