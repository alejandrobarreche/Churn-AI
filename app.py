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
# SIDEBAR  (CHANGE 1)
# ═══════════════════════════════════════════════════════════════
with st.sidebar:
    # Logo / brand
    st.markdown("""
    <div style="text-align:center; padding: 15px 0 20px 0;">
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

    # NAVEGACIÓN section header
    st.markdown(f"""
    <div style="margin: 4px 0 8px 0;">
        <span style="font-family:'Orbitron',monospace; font-size:0.65rem; letter-spacing:3px;
                     color:{UAX_GREY}; text-transform:uppercase;">NAVEGACIÓN</span>
    </div>
    <div style="height:1px; background:linear-gradient(90deg,transparent,rgba(200,169,81,0.3),transparent); margin:8px 0 16px 0;"></div>
    <div style="font-family:'Rajdhani',sans-serif; font-size:0.85rem; color:{UAX_GREY}; line-height:2;">
        🏠 Resumen &nbsp;·&nbsp; 📈 CLTV &nbsp;·&nbsp; 🎯 Segmentación<br>
        🔮 Proyección &nbsp;·&nbsp; 🔬 Modelos &nbsp;·&nbsp; 🧮 Predictor
    </div>
    <div style="height:1px; background:linear-gradient(90deg,transparent,rgba(200,169,81,0.3),transparent); margin:16px 0;"></div>
    """, unsafe_allow_html=True)

    if data_loaded:
        # FILTROS section header
        st.markdown(f"""
        <div style="margin: 8px 0 4px 0;">
            <span style="font-family:'Orbitron',monospace; font-size:0.65rem; letter-spacing:3px;
                         color:{UAX_GREY}; text-transform:uppercase;">FILTROS</span>
        </div>
        <div style="height:1px; background:linear-gradient(90deg,transparent,rgba(200,169,81,0.3),transparent); margin:8px 0 12px 0;"></div>
        """, unsafe_allow_html=True)

        st.markdown(f"""
        <div style="font-family:'Rajdhani',sans-serif; font-size:0.78rem; color:{UAX_GOLD};
                    letter-spacing:1px; text-transform:uppercase; margin-bottom:4px;">
            🚗 Modelo de vehículo
        </div>
        """, unsafe_allow_html=True)
        modelos_disponibles = sorted(df['Modelo_letra'].unique())
        modelos_sel = st.multiselect("Modelo de vehículo", modelos_disponibles, default=modelos_disponibles, label_visibility="collapsed")

        st.markdown(f"""
        <div style="font-family:'Rajdhani',sans-serif; font-size:0.78rem; color:{UAX_GOLD};
                    letter-spacing:1px; text-transform:uppercase; margin-top:10px; margin-bottom:4px;">
            ⚠️ Nivel de riesgo
        </div>
        """, unsafe_allow_html=True)
        riesgos_sel = st.multiselect("Nivel de riesgo", ['MUY_BAJO', 'BAJO', 'MEDIO', 'ALTO', 'MUY_ALTO'], default=['MUY_BAJO', 'BAJO', 'MEDIO', 'ALTO', 'MUY_ALTO'], label_visibility="collapsed")

        st.markdown(f"""
        <div style="font-family:'Rajdhani',sans-serif; font-size:0.78rem; color:{UAX_GOLD};
                    letter-spacing:1px; text-transform:uppercase; margin-top:10px; margin-bottom:4px;">
            💰 Rango PVP (€)
        </div>
        """, unsafe_allow_html=True)
        pvp_range = st.slider("Rango PVP (€)", int(df['PVP_original'].min()), int(df['PVP_original'].max()),
                              (int(df['PVP_original'].min()), int(df['PVP_original'].max())), label_visibility="collapsed")

        # Filter — se calcula aquí y es accesible fuera del bloque with
        mask = (
                df['Modelo_letra'].isin(modelos_sel) &
                df['riesgo'].isin(riesgos_sel) &
                df['PVP_original'].between(pvp_range[0], pvp_range[1])
        )
        dff = df[mask].copy()

        # DATASET section
        st.markdown(f"""
        <div style="height:1px; background:linear-gradient(90deg,transparent,rgba(200,169,81,0.3),transparent); margin:16px 0;"></div>
        <div style="margin: 8px 0 4px 0;">
            <span style="font-family:'Orbitron',monospace; font-size:0.65rem; letter-spacing:3px;
                         color:{UAX_GREY}; text-transform:uppercase;">DATASET</span>
        </div>
        <div style="height:1px; background:linear-gradient(90deg,transparent,rgba(200,169,81,0.3),transparent); margin:8px 0 12px 0;"></div>
        <div style="font-family:'Rajdhani',sans-serif; font-size:0.85rem; color:{UAX_TEXT}; line-height:2.0;">
            <span style="color:{UAX_GREY};">Total clientes:</span>
            <span style="color:{UAX_GOLD}; font-weight:600;"> {len(df):,}</span><br>
            <span style="color:{UAX_GREY};">P(churn) media:</span>
            <span style="color:{UAX_GOLD}; font-weight:600;"> {df['p_churn'].mean():.2%}</span><br>
            <span style="color:{UAX_GREY};">CLTV medio:</span>
            <span style="color:{UAX_GOLD}; font-weight:600;"> {df['CLTV'].mean():,.0f}€</span>
        </div>
        <div style="height:1px; background:linear-gradient(90deg,transparent,rgba(200,169,81,0.3),transparent); margin:16px 0;"></div>
        """, unsafe_allow_html=True)

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

# ─── TABS (CHANGE 4 — 6 tabs instead of 7) ─────────────────────
tab1, tab2, tab3, tab4, tab5, tab6 = st.tabs([
    "🏠 Resumen",
    "📈 CLTV Analysis",
    "🎯 Segmentación & Acciones",
    "🔮 Proyección Revisiones",
    "🔬 Modelos",
    "🧮 Predictor de Churn",
])


# ═══════════════════════════════════════════════════════════════
# TAB 1: RESUMEN  (CHANGE 2)
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

    # ── A) Pipeline visual diagram — columnas nativas Streamlit ──
    st.markdown("## Pipeline del Proyecto")

    _pipe_steps = [
        ("📊", "DATOS",        "customer_data\nnuevos_clientes\ncostes.csv",    False),
        ("🔧", "PIPELINE ML",  "Encoders\nEstandarización\nSklearn Pipeline",   False),
        ("🤖", "MODELO",       f"XGBoost ★\nAUC-ROC 0.83\nRF · LightGBM",     True),
        ("📐", "THRESHOLD",    f"Óptimo: {BEST_THRESHOLD:.2f}\nF-beta\nNegocio-driven", False),
        ("💰", "CLTV",         "10 revisiones\nMargen 62%\nSupervivencia",      False),
        ("🎯", "ACCIONES",     "5 segmentos\nReglas negocio\nROI estimado",     False),
    ]
    # Intercalamos columnas de paso (6) y flechas (5) → 11 columnas, anchos alternados
    _col_widths = []
    for i in range(len(_pipe_steps)):
        _col_widths.append(2)        # paso
        if i < len(_pipe_steps) - 1:
            _col_widths.append(0.3)  # flecha

    _pipe_cols = st.columns(_col_widths)
    _col_idx = 0
    for i, (icon, label, desc, highlight) in enumerate(_pipe_steps):
        border = f"2px solid {UAX_GOLD}" if highlight else "1px solid rgba(200,169,81,0.25)"
        bg     = f"linear-gradient(145deg,{UAX_NAVY},{UAX_CARD2})" if highlight else f"linear-gradient(145deg,{UAX_CARD},{UAX_CARD2})"
        with _pipe_cols[_col_idx]:
            st.markdown(f"""
            <div style="background:{bg}; border:{border}; border-radius:10px;
                        padding:14px 8px 12px 8px; text-align:center;">
              <div style="font-size:1.6rem;">{icon}</div>
              <div style="font-family:'Orbitron',monospace; font-size:0.65rem; color:{UAX_GOLD};
                          letter-spacing:2px; margin:6px 0 4px;">{label}</div>
              <div style="font-family:'Rajdhani',sans-serif; font-size:0.78rem; color:{UAX_GREY};
                          line-height:1.5;">{'<br>'.join(desc.split(chr(10)))}</div>
            </div>
            """, unsafe_allow_html=True)
        _col_idx += 1
        if i < len(_pipe_steps) - 1:
            with _pipe_cols[_col_idx]:
                st.markdown(f"""
                <div style="text-align:center; padding-top:38px;
                            font-size:1.4rem; color:{UAX_GOLD};">→</div>
                """, unsafe_allow_html=True)
            _col_idx += 1

    section_line()

    # ── B) Model comparison cards ───────────────────────────────
    st.markdown("## Comparación de Modelos")

    _model_display = [
        ('XGBoost', True),
        ('Random Forest', False),
        ('LightGBM', False),
    ]
    mod_cols = st.columns(3)
    for col_idx, (mname, is_best) in enumerate(_model_display):
        with mod_cols[col_idx]:
            if model_metrics and mname in model_metrics:
                mm = model_metrics[mname]
                auc_val  = f"{mm['auc']:.4f}"
                prec_val = f"{mm['precision']:.4f}"
                rec_val  = f"{mm['recall']:.4f}"
                f1_val   = f"{mm['f1']:.4f}"
            else:
                auc_val  = "0.83" if mname == "XGBoost" else ("0.81" if mname == "Random Forest" else "0.80")
                prec_val = "—"
                rec_val  = "—"
                f1_val   = "—"

            badge = f'<span style="background:{UAX_GOLD}; color:{UAX_BG}; font-family:Orbitron,monospace; font-size:0.6rem; padding:3px 8px; border-radius:20px; letter-spacing:1px;">MODELO SELECCIONADO</span>' if is_best else ''
            border_style = f"2px solid {UAX_GOLD}" if is_best else "1px solid rgba(200,169,81,0.2)"
            st.markdown(f"""
            <div style="background:linear-gradient(145deg,{UAX_CARD},{UAX_CARD2}); border:{border_style};
                        border-radius:12px; padding:20px 16px; text-align:center; position:relative;">
                <div style="margin-bottom:8px;">{badge}</div>
                <div style="font-family:'Orbitron',monospace; font-size:1rem; color:{UAX_GOLD}; margin:8px 0 14px; letter-spacing:1px;">{mname}</div>
                <div style="display:grid; grid-template-columns:1fr 1fr; gap:8px;">
                    <div style="background:rgba(200,169,81,0.06); border-radius:8px; padding:10px 6px;">
                        <div style="font-family:'Rajdhani'; font-size:0.72rem; color:{UAX_GREY}; letter-spacing:1px; text-transform:uppercase;">AUC-ROC</div>
                        <div style="font-family:'Orbitron'; font-size:1.1rem; color:{UAX_TEXT}; margin-top:4px;">{auc_val}</div>
                    </div>
                    <div style="background:rgba(200,169,81,0.06); border-radius:8px; padding:10px 6px;">
                        <div style="font-family:'Rajdhani'; font-size:0.72rem; color:{UAX_GREY}; letter-spacing:1px; text-transform:uppercase;">Precision</div>
                        <div style="font-family:'Orbitron'; font-size:1.1rem; color:{UAX_TEXT}; margin-top:4px;">{prec_val}</div>
                    </div>
                    <div style="background:rgba(200,169,81,0.06); border-radius:8px; padding:10px 6px;">
                        <div style="font-family:'Rajdhani'; font-size:0.72rem; color:{UAX_GREY}; letter-spacing:1px; text-transform:uppercase;">Recall</div>
                        <div style="font-family:'Orbitron'; font-size:1.1rem; color:{UAX_TEXT}; margin-top:4px;">{rec_val}</div>
                    </div>
                    <div style="background:rgba(200,169,81,0.06); border-radius:8px; padding:10px 6px;">
                        <div style="font-family:'Rajdhani'; font-size:0.72rem; color:{UAX_GREY}; letter-spacing:1px; text-transform:uppercase;">F1</div>
                        <div style="font-family:'Orbitron'; font-size:1.1rem; color:{UAX_TEXT}; margin-top:4px;">{f1_val}</div>
                    </div>
                </div>
            </div>
            """, unsafe_allow_html=True)


# ═══════════════════════════════════════════════════════════════
# TAB 2: CLTV ANALYSIS  (CHANGE 3)
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

    # ── Per-client analysis ─────────────────────────────────────
    section_line()
    st.markdown("## Análisis Individual de Cliente")

    if len(dff) > 0:
        RIESGO_COLORS_CLTV = {
            'MUY_BAJO': UAX_GREEN,
            'BAJO':     '#80D080',
            'MEDIO':    UAX_GOLD,
            'ALTO':     UAX_RED,
            'MUY_ALTO': '#FF4444',
        }

        cliente_cltv_sel = st.selectbox(
            "Selecciona cliente (CODE)",
            sorted(dff['CODE'].unique()),
            key="cltv_cliente_sel"
        )
        cliente_row = dff[dff['CODE'] == cliente_cltv_sel].iloc[0]

        cl_col1, cl_col2 = st.columns(2)

        with cl_col1:
            pchurn_pct = float(cliente_row['p_churn']) * 100
            riesgo_color_gauge = RIESGO_COLORS_CLTV.get(str(cliente_row['riesgo']), UAX_GOLD)
            fig_g = go.Figure(go.Indicator(
                mode="gauge+number",
                value=pchurn_pct,
                number={'suffix': '%', 'font': {'family': 'Orbitron', 'color': UAX_GOLD, 'size': 32}},
                title={'text': "P(Churn)", 'font': {'family': 'Orbitron', 'color': UAX_TEXT, 'size': 13}},
                gauge={
                    'axis': {'range': [0, 100], 'tickcolor': UAX_GREY,
                             'tickfont': {'family': 'Rajdhani', 'color': UAX_GREY}},
                    'bar': {'color': riesgo_color_gauge, 'thickness': 0.25},
                    'bgcolor': UAX_CARD2,
                    'borderwidth': 1,
                    'bordercolor': UAX_GREY,
                    'steps': [
                        {'range': [0, 20],  'color': 'rgba(39,174,96,0.15)'},
                        {'range': [20, 40], 'color': 'rgba(200,169,81,0.12)'},
                        {'range': [40, 60], 'color': 'rgba(200,169,81,0.20)'},
                        {'range': [60, 80], 'color': 'rgba(192,57,43,0.15)'},
                        {'range': [80, 100],'color': 'rgba(192,57,43,0.25)'},
                    ],
                }
            ))
            fig_g.update_layout(
                paper_bgcolor=UAX_CARD,
                font=dict(family='Rajdhani', color=UAX_TEXT),
                height=300,
                margin=dict(l=20, r=20, t=50, b=10),
            )
            st.plotly_chart(fig_g, width='stretch')

        with cl_col2:
            cltv_fmt = f"{float(cliente_row['CLTV']):,.0f}€"
            pchurn_fmt = f"{float(cliente_row['p_churn']):.2%}"
            pvp_fmt = f"{float(cliente_row['PVP_original']):,.0f}€"
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
                        <td style="color:{UAX_GOLD}; font-weight:600; text-align:right;">{cltv_fmt}</td></tr>
                    <tr><td style="color:{UAX_GREY}; padding:4px 0;">P(Churn)</td>
                        <td style="color:{UAX_TEXT}; font-weight:600; text-align:right;">{pchurn_fmt}</td></tr>
                    <tr><td style="color:{UAX_GREY}; padding:4px 0;">Riesgo</td>
                        <td style="color:{RIESGO_COLORS_CLTV.get(str(cliente_row['riesgo']), UAX_GOLD)}; font-weight:700; text-align:right;">{cliente_row['riesgo']}</td></tr>
                    <tr><td style="color:{UAX_GREY}; padding:4px 0;">Acción</td>
                        <td style="color:{UAX_TEXT}; font-weight:600; text-align:right;">{cliente_row['accion']}</td></tr>
                    <tr><td style="color:{UAX_GREY}; padding:4px 0;">PVP original</td>
                        <td style="color:{UAX_TEXT}; font-weight:600; text-align:right;">{pvp_fmt}</td></tr>
                    <tr><td style="color:{UAX_GREY}; padding:4px 0;">Zona</td>
                        <td style="color:{UAX_TEXT}; font-weight:600; text-align:right;">{cliente_row['ZONA_original']}</td></tr>
                </table>
            </div>
            """, unsafe_allow_html=True)

        # Scatter with selected client highlighted
        fig_sel = go.Figure()
        for seg in ['MUY_BAJO', 'BAJO', 'MEDIO', 'ALTO', 'MUY_ALTO']:
            sub = dff[dff['riesgo'] == seg]
            fig_sel.add_trace(go.Scatter(
                x=sub['p_churn'], y=sub['CLTV'],
                mode='markers',
                name=seg,
                marker=dict(size=6, color=RIESGO_COLORS_CLTV[seg], opacity=0.55),
                hovertemplate='<b>%{text}</b><br>P(Churn): %{x:.2%}<br>CLTV: %{y:,.0f}€<extra></extra>',
                text=sub['CODE'].astype(str),
            ))
        fig_sel.add_trace(go.Scatter(
            x=[float(cliente_row['p_churn'])],
            y=[float(cliente_row['CLTV'])],
            mode='markers',
            name='Cliente seleccionado',
            marker=dict(size=18, color=UAX_GOLD, symbol='star',
                        line=dict(color='#FFFFFF', width=2)),
            hovertemplate=f'<b>{cliente_cltv_sel}</b><br>P(Churn): {float(cliente_row["p_churn"]):.2%}<br>CLTV: {float(cliente_row["CLTV"]):,.0f}€<extra></extra>',
        ))
        fig_sel.update_layout(
            title="Posición del cliente en la cartera (P(Churn) vs CLTV)",
            xaxis_title="P(Churn)",
            yaxis_title="CLTV (€)",
            xaxis=dict(tickformat='.0%'),
            yaxis=dict(tickformat=",.0f"),
        )
        plotly_layout(fig_sel, height=450)
        st.plotly_chart(fig_sel, width='stretch')


# ═══════════════════════════════════════════════════════════════
# TAB 3: SEGMENTACIÓN & ACCIONES  (CHANGE 4 — merged tab)
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

    # ── ACCIONES COMERCIALES (merged from old tab4) ─────────────
    section_line()
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

    # ── Visual segment cards replacing plain dataframe ──────────
    st.markdown("### Reglas de negocio")

    RIESGO_CARD_COLORS = {
        'MUY_ALTO': '#ef4444',
        'ALTO':     '#f97316',
        'MEDIO':    '#f59e0b',
        'BAJO':     '#22c55e',
        'MUY_BAJO': '#10b981',
    }
    RIESGO_ICONS = {
        'MUY_ALTO': '🔴',
        'ALTO':     '🟠',
        'MEDIO':    '🟡',
        'BAJO':     '🟢',
        'MUY_BAJO': '✅',
    }
    # Rules data per riesgo level
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

    for riesgo_key in ['MUY_ALTO', 'ALTO', 'MEDIO', 'BAJO', 'MUY_BAJO']:
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
                st.markdown(f"""
                <div style="background:rgba({int(rcolor[1:3],16)},{int(rcolor[3:5],16)},{int(rcolor[5:7],16)},0.08);
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
# TAB 4: PROYECCIÓN REVISIONES  (CHANGE 5 — formerly tab5)
# ═══════════════════════════════════════════════════════════════
with tab4:
    st.markdown("## Proyección de Revisiones Futuras")

    # Auto-select all models (no multiselect)
    modelos_rev_all = sorted(df_rev['Modelo'].unique())
    df_rev_f = df_rev[df_rev['Modelo'].isin(modelos_rev_all)]

    fig = go.Figure()
    for modelo in modelos_rev_all:
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

    # ── Formula explanation ──────────────────────────────────────
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

    # ── Sensitivity analysis chart ───────────────────────────────
    st.markdown("### Sensibilidad del CLTV según riesgo de churn")

    _costes_dict_rev = costes.set_index('Modelo')['Mantenimiento_medio'].to_dict()
    _base_rev = _costes_dict_rev.get('A', 300)
    _alpha_rev = 0.07
    _MARGEN_REV = 0.62
    _p_values = [0.1, 0.3, 0.5, 0.7, 0.9]
    _p_labels = ['p=0.1 (muy bajo)', 'p=0.3 (bajo)', 'p=0.5 (medio)', 'p=0.7 (alto)', 'p=0.9 (muy alto)']
    _p_colors = [UAX_GREEN, '#80D080', UAX_GOLD, UAX_RED, '#FF4444']
    _revisiones = list(range(1, 11))

    fig_sens = go.Figure()
    for p_val, p_label, p_color in zip(_p_values, _p_labels, _p_colors):
        cltv_acum = []
        running = 0
        for n in _revisiones:
            ingreso = _base_rev * (1 + _alpha_rev) ** n
            running += ingreso * _MARGEN_REV * (1 - p_val) ** n
            cltv_acum.append(running)
        fig_sens.add_trace(go.Scatter(
            x=_revisiones, y=cltv_acum,
            mode='lines+markers',
            name=p_label,
            line=dict(color=p_color, width=2.5),
            marker=dict(size=7, color=p_color),
        ))
    fig_sens.update_layout(
        title="Sensibilidad del CLTV según riesgo de churn (Modelo A, α=7%)",
        xaxis_title="Nº de revisiones",
        yaxis_title="CLTV acumulado (€)",
        xaxis=dict(dtick=1),
        yaxis=dict(tickformat=",.0f"),
    )
    plotly_layout(fig_sens, height=450)
    st.plotly_chart(fig_sens, width='stretch')


# ═══════════════════════════════════════════════════════════════
# TAB 5: COMPARACIÓN DE MODELOS  (formerly tab6)
# ═══════════════════════════════════════════════════════════════
with tab5:
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
# TAB 6: PREDICTOR DE CHURN  (CHANGE 6 — formerly tab7)
# ═══════════════════════════════════════════════════════════════
with tab6:
    st.markdown("## Predictor Individual de Churn")

    st.info(
        "Los campos numéricos no tienen restricciones de rango — puedes introducir valores extremos "
        "para ver cómo responde el modelo. Los valores por defecto corresponden a la mediana del "
        "conjunto de entrenamiento."
    )

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
            pvp_input = st.number_input("PVP (€) *", value=22000, step=500)
            kw_input = st.number_input("Potencia (kW) *", value=100, step=5)
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

            edad_input = st.number_input("Edad *", value=40, step=1)
            renta_input = st.number_input("Renta media estimada (€) *", value=18000, step=500)
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