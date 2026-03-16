import streamlit as st
from ..config import UAX_GOLD, UAX_GREY, UAX_NAVY, UAX_CARD, UAX_CARD2, UAX_TEXT, UAX_BG
from ..utils import metric_card, section_line


def render(dff, df, BEST_THRESHOLD, model_metrics):
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

    # ── A) Pipeline visual diagram ────────────────────────────────
    st.markdown("## Pipeline del Proyecto")

    _pipe_steps = [
        ("📊", "DATOS",       "customer_data\nnuevos_clientes\ncostes.csv",    False),
        ("🔧", "PIPELINE ML", "Encoders\nEstandarización\nSklearn Pipeline",   False),
        ("🤖", "MODELO",      f"XGBoost ★\nAUC-ROC 0.83\nRF · LightGBM",     True),
        ("📐", "THRESHOLD",   f"Óptimo: {BEST_THRESHOLD:.2f}\nF-beta\nNegocio-driven", False),
        ("💰", "CLTV",        "10 revisiones\nMargen 62%\nSupervivencia",      False),
        ("🎯", "ACCIONES",    "5 segmentos\nReglas negocio\nROI estimado",     False),
    ]
    _col_widths = []
    for i in range(len(_pipe_steps)):
        _col_widths.append(2)
        if i < len(_pipe_steps) - 1:
            _col_widths.append(0.3)

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

    # ── B) Model comparison cards ─────────────────────────────────
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
