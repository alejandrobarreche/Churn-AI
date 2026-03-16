# Plan de Implementación — Mejoras de ruru a los notebooks modulares
**Última actualización:** 2026-03-16

---

## Cómo retomar este plan

Al inicio de cada nueva sesión, indicar:
> *"Continúa con el plan en `md/plan_implementacion.md`. El último paso completado fue [ID]. Ahora toca [siguiente ID]."*

---

## Fase 1 — Cambios seguros (sin re-ejecutar cadena de artefactos)

- [x] **C1** — Corregir bug `fit_transform` → `transform` en NB4 ✓ (2026-03-14)
  - Archivo: `4-model_verification_.ipynb`
  - Celdas: la de transformación del test set y la de nuevos clientes
  - El código dice `full_pipeline.fit_transform(...)` pero el propio markdown del NB advierte que debe ser `transform`. Re-aprende los mapas de frecuencia sobre datos que no debería ver.
  - Re-ejecutar: solo NB4 (cambio aislado)

- [x] **I2** — Añadir sección de análisis de leakage lógico en NB3 ✓ (2026-03-14)
  - Archivo: `3-modeling_.ipynb`
  - Dónde: inicio del notebook, entre carga de datos y funciones auxiliares
  - Contenido: explicar por qué `DAYS_LAST_SERVICE`, `Revisiones` y `km_ultima_revision` se excluyen (leakage directo, proxy). Añadir celda demostrativa: entrenar RF *con* `DAYS_LAST_SERVICE` y mostrar AUC artificial ~0.95 vs ~0.83 real.
  - Re-ejecutar: no guarda artefactos

- [x] **I3** — Hacer `StratifiedKFold` explícito en NB3 ✓ (2026-03-14)
  - Archivo: `3-modeling_.ipynb`
  - Cambio: definir `cv_strategy = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)` y reemplazar `cv=5` en todas las llamadas de CV y tuning
  - Re-ejecutar: NB3 (métricas cambiarán mínimamente, gana reproducibilidad)

- [x] **M2** — Análisis temporal del churn por año de venta en NB1 ✓ (2026-03-14)
  - Archivo: `1-data_understanding_.ipynb`
  - Dónde: nueva subsección tras la distribución del target (sección 4)
  - Contenido: tasa de churn por año (2018-2023), gráfico de línea en paleta UAX
  - Re-ejecutar: ninguno

- [x] **M3** — Análisis económico de márgenes por modelo en NB1 ✓ (2026-03-14)
  - Archivo: `1-data_understanding_.ipynb`
  - Dónde: nueva sección 6 (antes de correlaciones)
  - Contenido: barras horizontales con `Margen_eur` promedio por modelo, paleta UAX
  - Re-ejecutar: ninguno

---

## Fase 2 — Re-ejecutar NB3 + NB4

- [x] **I1** — Añadir LightGBM como tercer modelo de ensemble ✓ (2026-03-14)
  - Archivos: `3-modeling_.ipynb` + `4-model_verification_.ipynb`
  - En NB3: nueva sección tras XGBoost, mismo patrón (baseline → tuning con RandomizedSearchCV → curvas → feature importance). Guardar `lightgbm.pkl`.
  - En NB4: añadir a la carga de modelos y a todos los bucles de evaluación
  - Re-ejecutar: NB3 → NB4

- [x] **M1** — Segmentación ALTO/MEDIO/BAJO + valor económico en riesgo ✓ (2026-03-14)
  - Archivo: `4-model_verification_.ipynb`
  - Dónde: nueva sección 6 tras el análisis de riesgo relativo
  - Contenido: asignar ALTO (≥0.70) / MEDIO (0.40-0.70) / BAJO (<0.40) a cada cliente nuevo. Calcular PVP total por segmento. Visualización en paleta UAX.
  - Re-ejecutar: ninguno adicional (se basa en predicciones ya existentes en NB4)

---

## Fase 3 — Re-ejecutar cadena completa desde NB2

- [x] **I4** — Feature `gasto_relativo = PVP / RENTA_MEDIA_ESTIMADA` en el pipeline ✓ (2026-03-14)
  - Archivos: `transformer.py` + `2-data_preparation_.ipynb`
  - En `transformer.py`: nuevo transformador `GastoRelativoEncoder` que calcula `PVP / RENTA_MEDIA_ESTIMADA.clip(lower=1)`. Colocarlo en el pipeline **antes** de `PriceStandard` (PVP debe estar en euros al calcular el ratio).
  - En NB2: añadir `GastoRelativoEncoder` al pipeline entre `NominalOneHotEncoder` y `PriceStandard`
  - Re-ejecutar: NB2 → NB3 → NB4
  - **Nota:** agrupar con C2 para hacer una sola re-ejecución completa

- [x] **C2** — Redefinir el target: clientes ghost también son churn (8.8% → ~33%) ✓ (2026-03-14)
  - Archivos: `1-data_understanding_.ipynb` (doc) + `2-data_preparation_.ipynb` (impl) + `transformer.py`
  - En NB1: nueva subsección que explica el concepto ghost con tabla comparativa
  - En NB2: crear `Churn_Final` antes del split: `(Churn_400=='Y') | (Revisiones==0 & antiguedad_dias>400)`
  - En `transformer.py`: cambiar `BinaryEncoder` para que mapee `Churn_Final` (ya llega como 0/1, solo pasar sin transformar) en lugar de `Churn_400`
  - Re-ejecutar: NB2 → NB3 → NB4
  - **Advertencia:** es el cambio de mayor impacto. Hacer tras I4 en la misma sesión de re-ejecución.

---

---

## Fase 4 — Mejoras en `5-cltv_acciones.ipynb` + `app.py` desde ruru

- [x] **A1** — Ampliar segmentación de riesgo: 3 → 5 grupos ✓ (2026-03-16)
  - Archivos: `5-cltv_acciones.ipynb` + `app.py`
  - Cambio: `pd.qcut(q=3, ['Bajo','Medio','Alto'])` → `pd.qcut(q=5, ['MUY_BAJO','BAJO','MEDIO','ALTO','MUY_ALTO'])`
  - Actualizada tabla de reglas y función `asignar_accion` para manejar 5 niveles
  - En `app.py`: sidebar filter, heatmap ordering y clasificación individual actualizados

- [x] **A2** — Añadir tasa de respuesta (TR) por segmento ✓ (2026-03-16)
  - Archivos: `5-cltv_acciones.ipynb` + `app.py`
  - `TASA_RESPUESTA`: MUY_ALTO=10%, ALTO=80%, MEDIO=55%, BAJO=45%, MUY_BAJO=40%
  - Ganancia CLTV y coste del servicio ponderados por TR (solo respondedores se benefician)
  - `COSTE_CONTACTO` (email/SMS) aplicado al 100% de clientes

- [x] **A3** — Reemplazar descuentos genéricos por paquetes de servicio concretos ✓ (2026-03-16)
  - Archivos: `5-cltv_acciones.ipynb` + `app.py`
  - Pack Premium VIP (ALTO): recogida(35) + lavado(30) + neumáticos(35) + bono(50) = 150€/respondedor
  - Pack Intermedio (MEDIO): regalo(6) + lavado(30) + bono(30) = 66€/respondedor
  - Contacto prioritario (MUY_ALTO): bono 20€/respondedor
  - `REDUCCION_CHURN` actualizado: VIP=−40%, Intermedio=−20%, Prioritario=−15%

- [x] **A4** — Pack de fidelización por 5 revisiones ✓ (2026-03-16)
  - Archivo: `5-cltv_acciones.ipynb` (sección 8.1)
  - Elegibles: BAJO/MUY_BAJO riesgo + Alto valor
  - −10% sobre acumulado 5 revisiones + bono 1.000€ para 2º vehículo
  - Tabla de rentabilidad por modelo (algunos modelos de gama baja no cubren el bono)

- [x] **A5** — Desglosar costes en 4 componentes ✓ (2026-03-16)
  - Archivos: `5-cltv_acciones.ipynb` + `app.py`
  - `Coste_Contacto` (100% clientes) + `Coste_Adicional` (TR%) + `Coste_Descuento` (TR%) + `Coste_Marketing` 1% rev1 (TR%)
  - CSV exportado incluye los 4 componentes por cliente

---

## Notas técnicas

- `BinaryEncoder` en `transformer.py` tiene hardcodeado `X['Churn_400'].map({'Y':1,'N':0})`. Al implementar C2 cambiar a `Churn_Final` (ya es int, no necesita map).
- `GastoRelativoEncoder` debe ir antes de `PriceStandard` para que el PVP esté en euros al calcular el ratio.
- C2 e I4 deben hacerse en la misma sesión para evitar dos re-ejecuciones completas de la cadena.
