import sys
sys.path.insert(0, '.')

import streamlit as st
import pandas as pd
import numpy as np
import joblib
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split
from sklearn.metrics import (roc_curve, auc, precision_score, recall_score,
                              f1_score, confusion_matrix)

from transformer import (BinaryEncoder, FrequencyEncoder, OrdinalExtensionEncoder,
                         OrdinalEquipamientoEncoder, NominalOneHotEncoder,
                         GastoRelativoEncoder, PriceStandard, InstanceDropper, ColumnDropper)


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
