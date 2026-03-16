import streamlit as st
from .config import UAX_GREY, UAX_GOLD, UAX_TEXT, UAX_BG, UAX_CARD


def render(df, data_loaded):
    """Render sidebar filters. Returns filtered dataframe (dff)."""
    dff = df.copy() if data_loaded else None

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
        <div style="font-family:'Rajdhani',sans-serif; font-size:0.85rem; color:{UAX_GREY}; line-height:2.4;">
            🏠 Resumen &nbsp;·&nbsp;
            📈 CLTV &nbsp;·&nbsp;
            🎯 Segmentación<br>
            🔮 Proyección &nbsp;·&nbsp;
            🔬 Modelos &nbsp;·&nbsp;
            🧮 Predictor
        </div>
        <div style="height:1px; background:linear-gradient(90deg,transparent,rgba(200,169,81,0.3),transparent); margin:16px 0;"></div>
        """, unsafe_allow_html=True)

        if not data_loaded:
            return None

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

    return dff
