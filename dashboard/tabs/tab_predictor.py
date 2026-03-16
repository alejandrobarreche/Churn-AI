import streamlit as st
import pandas as pd
import numpy as np
import traceback
import plotly.graph_objects as go
from ..config import (UAX_GOLD, UAX_GREY, UAX_TEXT, UAX_CARD, UAX_CARD2,
                      UAX_RED, UAX_GREEN, UAX_ACCENT)
from ..utils import section_line, plotly_layout


def render(df, BEST_THRESHOLD, full_pipeline, xgboost_model, train_medianas, costes):
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

        # ── Columna 1: Vehículo ────────────────────────────────────
        with col_v:
            st.markdown(f"<div style='font-family:Orbitron; color:{UAX_GOLD}; font-size:0.85rem; letter-spacing:2px; text-transform:uppercase; margin-bottom:12px;'>Vehículo</div>", unsafe_allow_html=True)

            modelo_input = st.selectbox("Modelo *", ['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J', 'K'])
            pvp_input = st.number_input("PVP (€) *", value=22000, step=500)
            kw_input = st.number_input("Potencia (kW) *", value=100, step=5)
            fuel_input = st.selectbox("Combustible *", ['ELÉCTRICO', 'HÍBRIDO'])
            trans_input = st.selectbox("Transmisión *", ['A', 'M'])
            carroceria_input = st.selectbox("Tipo carrocería *", ['TIPO1', 'TIPO2', 'TIPO3', 'TIPO4', 'TIPO5', 'TIPO6', 'TIPO7', 'TIPO8'])
            equip_input = st.selectbox("Equipamiento *", ['Low', 'Mid', 'Mid-High', 'High'])

        # ── Columna 2: Contrato y garantía ────────────────────────
        with col_c:
            st.markdown(f"<div style='font-family:Orbitron; color:{UAX_GOLD}; font-size:0.85rem; letter-spacing:2px; text-transform:uppercase; margin-bottom:12px;'>Contrato y Garantía</div>", unsafe_allow_html=True)

            garantia_input = st.radio("En garantía *", ['SI', 'NO'], horizontal=True)
            ext_garantia_input = st.selectbox("Extensión garantía *", ['NO', 'SI', 'SI, Financiera'])
            seguro_bat_input = st.radio("Seguro batería largo plazo *", ['SI', 'NO'], horizontal=True)
            mant_gratis_input = st.slider("Mantenimientos gratuitos", min_value=0, max_value=4, value=0)
            forma_pago_input = st.selectbox("Forma de pago *", ['Contado', 'Financiera Marca', 'Otros', 'Prestamo Bancario'])
            motivo_venta_input = st.selectbox("Motivo de venta *", ['Particular', 'No Particular'])

        # ── Columna 3: Cliente y geografía ────────────────────────
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

    # ── Resultado ──────────────────────────────────────────────────
    if not submitted:
        return

    now = pd.Timestamp.now()

    days_svc   = train_medianas.get('DAYS_LAST_SERVICE', 255.0)
    km_rev     = train_medianas.get('Km_medio_por_revision', np.nan)
    km_ultima  = train_medianas.get('km_ultima_revision', np.nan)
    encuesta   = train_medianas.get('ENCUESTA_CLIENTE_ZONA_TALLER', np.nan)
    margen_bruto = train_medianas.get('Margen_eur_bruto', np.nan)
    margen_eur   = train_medianas.get('Margen_eur', np.nan)
    coste_venta  = train_medianas.get('COSTE_VENTA_NO_IMPUESTOS', np.nan)

    row = {
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
    for col in df_pred.select_dtypes(include=['string']).columns:
        df_pred[col] = df_pred[col].astype(object)

    try:
        X_pred = full_pipeline.fit_transform(df_pred)
        if 'Churn_400' in X_pred.columns:
            X_pred = X_pred.drop(columns=['Churn_400'])

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

    if not pred_ok:
        st.error("Error al ejecutar la predicción. Verifica que el pipeline sea compatible con los campos introducidos.")
        st.code(pred_error_msg, language="python")
        return

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

    # ── Contextual scatter — portfolio positioning ─────────────────
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
