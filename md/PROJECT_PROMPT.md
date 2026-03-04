# Churn Prediction — Revisión completa del proyecto

Eres un experto en Machine Learning y ciencia de datos. Te pido que revises este proyecto de predicción de churn en busca de errores conceptuales, de implementación, de leakage, y de inconsistencias entre las decisiones de análisis y el código final. Al final de cada sección identifico las áreas de riesgo que me preocupan. Sé exhaustivo.

---

## 1. Contexto del problema

**Objetivo:** construir un modelo de clasificación binaria que prediga si un cliente de un concesionario de coches tendrá churn (`churn_400 = Y`) o no (`churn_400 = N`).

**Definición del target:**
`churn_400 = Y` cuando el cliente lleva más de 400 días sin pasar por el taller desde su última revisión (medido en `base_date = 2023-12-31`).

**Anomalías confirmadas en el target (verificadas empíricamente):**
- **Anomalía 1:** 14.236 clientes nunca fueron al taller, llevan >400 días desde la compra (media 665 días) y son `churn = N`. La regla simple `days_last_service > 400 → churn=Y` no se cumple en este grupo. El label parece requerir al menos una visita al taller para activarse.
- **Anomalía 2:** 1.338 clientes tienen `days_last_service > 400` y son `churn = N`. Probablemente son clientes multi-compra: el servicio es de un vehículo anterior pero la venta registrada es reciente y "reinicia el reloj".

**Regla real inferida:** `churn_400 = Y` ↔ el cliente tiene `revisiones ≥ 1` AND `days_last_service > 400`.

---

## 2. Esquema de datos

```
PostgreSQL (churn_db) → schema personal del usuario
```

| Tabla | Filas | Rol |
|---|---|---|
| `sales` | 58.049 | Tabla principal de train (una fila = una transacción) |
| `new_sales` | 10.000 | Conjunto de scoring (sin `churn_400`) |
| `customers` | 52.553 | Perfil demográfico del cliente |
| `products` | 504 | Catálogo de vehículos (11 modelos, ~46 configs/modelo) |
| `stores` | 12 | Tiendas (12 concesionarios en 4 zonas) |
| `model_costs` | 11 | Costes por modelo — descartada completamente |

**Estadísticas del dataset:**
- 44.053 clientes únicos en 58.049 transacciones (clientes con múltiples compras)
- Tasa de churn: 8.8% (5.093 Y / 52.956 N) — dataset **fuertemente desbalanceado**
- El modelo trabaja a nivel de **transacción**, no de cliente

---

## 3. Pipeline completo

```
data_understanding.ipynb → data_preparation.ipynb → modeling.ipynb
```

### 3.1 `data_preparation.ipynb`

1. **Copia de tablas** al schema personal del usuario en PostgreSQL
2. **JOIN principal** (`sales ⋈ customers ⋈ products ⋈ stores`) → 58.049 filas × 31 cols
3. **Agregados históricos por cliente** (RFM) vía SQL con window functions → merge a ambos dataframes
4. **Feature engineering** (`feature_engineering()`)
5. **Tratamiento de nulos** (`tratar_nulos()`)
6. **Selección de features** (lista `FEATURES` de 36 variables)
7. **Guardado** en PostgreSQL y CSV (`train.csv`, `scoring.csv`)

### 3.2 `modeling.ipynb`

1. Carga de `train.csv`
2. Análisis exploratorio (distribuciones por churn)
3. Split train/test 80/20 estratificado
4. Definición de grupos de features
5. Entrenamiento de 4 modelos con sklearn pipelines
6. Evaluación y comparación
7. Predicciones sobre `scoring.csv`

---

## 4. Feature Engineering — Código exacto

### 4.1 Agregados históricos por cliente (computed en SQL)

```python
# Ejecutado con LAG() sobre TODO el histórico de sales
df_cust_agg = agent.execute_dml(f"""
    WITH sales_lag AS (
        SELECT customer_id, sales_date, pvp,
               LAG(sales_date) OVER (PARTITION BY customer_id ORDER BY sales_date) AS prev_date
        FROM {SCHEMA}.sales
    )
    SELECT
        customer_id,
        COUNT(*)                                               AS frequency_total,
        MIN(sales_date)                                        AS first_sale_date,
        AVG(pvp)                                               AS ticket_avg,
        AVG(CASE WHEN prev_date IS NOT NULL
                 THEN (sales_date - prev_date)::float END)     AS service_interval_mean_days,
        STDDEV_SAMP(CASE WHEN prev_date IS NOT NULL
                 THEN (sales_date - prev_date)::float END)     AS service_interval_std_days
    FROM sales_lag
    GROUP BY customer_id
""")
df_train   = df_train.merge(df_cust_agg, on='customer_id', how='left')
df_scoring = df_scoring.merge(df_cust_agg, on='customer_id', how='left')
```

### 4.2 `feature_engineering()` — código completo

```python
def feature_engineering(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()

    # ── Temporal ──────────────────────────────────────────────────────────────
    base_date = df['sales_date'].max()
    df['dias_desde_compra']      = (base_date - df['sales_date']).dt.days
    df['tenure_days']            = (df['sales_date'] - df['first_sale_date']).dt.days
    df['garantia_dias_restantes'] = (df['fin_garantia'] - base_date).dt.days

    # ── Económicas ────────────────────────────────────────────────────────────
    df['margen_relativo']     = df['margen_eur'] / df['pvp'].replace(0, pd.NA)
    df['margen_eur_negativo'] = (df['margen_eur'] < 0).astype(int)

    # ── Producto ──────────────────────────────────────────────────────────────
    df['es_electrico'] = (df['fuel'] != 'HÍBRIDO').astype(int)

    # ── Garantía ──────────────────────────────────────────────────────────────
    df['ext_garantia_tiene']      = (df['extension_garantia'] != 'NO').astype(int)
    df['ext_garantia_financiera'] = df['extension_garantia'].str.contains('Financiera', na=False).astype(int)

    # ── Servicios — BUG FIX: mantenimiento_gratuito es 0/4, no 'SI'/'NO'
    df['mantenimiento_gratuito']     = (df['mantenimiento_gratuito'].astype(str) != '0').astype(int)
    df['seguro_bateria_largo_plazo'] = (df['seguro_bateria_largo_plazo'] == 'SI').astype(int)
    df['en_garantia']                = (df['en_garantia'] == 'SI').astype(int)
    df['sin_revisiones']             = (df['revisiones'] == 0).astype(int)

    # ── Renta ─────────────────────────────────────────────────────────────────
    df['renta_desconocida'] = (df['renta_media_estimada'] == 0).astype(int)

    # ── Satisfacción / quejas ─────────────────────────────────────────────────
    df['tiene_queja'] = (df['queja'] == 'SI').astype(int)

    return df
```

### 4.3 `tratar_nulos()` — código completo

```python
def tratar_nulos(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df['genero']               = df['genero'].fillna('Desconocido')
    df['status_social']        = df['status_social'].fillna('Sin_dato')
    df['renta_media_estimada'] = df['renta_media_estimada'].fillna(0)
    df['margen_relativo']      = df['margen_relativo'].fillna(0)
    df['encuesta_cliente_zona_taller'] = df['encuesta_cliente_zona_taller'].fillna(
        df['encuesta_cliente_zona_taller'].median()
    )
    df['km_ultima_revision']    = df['km_ultima_revision'].fillna(0)
    df['km_medio_por_revision'] = df['km_medio_por_revision'].fillna(0)
    df['garantia_dias_restantes'] = df['garantia_dias_restantes'].fillna(-999)
    df['days_last_service'] = df['days_last_service'].fillna(0)
    df['fue_lead']          = df['fue_lead'].fillna(0)
    df['lead_compra']       = df['lead_compra'].fillna(0)
    df['frequency_total']            = df['frequency_total'].fillna(0)
    df['tenure_days']                = df['tenure_days'].fillna(0)
    df['ticket_avg']                 = df['ticket_avg'].fillna(df['pvp'])
    df['service_interval_mean_days'] = df['service_interval_mean_days'].fillna(0)
    df['service_interval_std_days']  = df['service_interval_std_days'].fillna(0)
    return df
```

### 4.4 Lista completa de FEATURES (36)

```python
FEATURES = [
    # Económicas
    'pvp', 'margen_relativo', 'margen_eur_negativo', 'coste_venta_no_impuestos', 'forma_pago',
    # Temporales
    'dias_desde_compra', 'tenure_days', 'days_last_service',
    # Garantía
    'en_garantia', 'ext_garantia_tiene', 'ext_garantia_financiera', 'garantia_dias_restantes',
    # Servicios / mantenimiento
    'mantenimiento_gratuito', 'seguro_bateria_largo_plazo', 'sin_revisiones', 'revisiones',
    # Odómetro
    'km_ultima_revision', 'km_medio_por_revision',
    # Satisfacción
    'encuesta_cliente_zona_taller', 'tiene_queja',
    # Comportamiento comercial
    'lead_compra', 'fue_lead',
    # RFM cross-cliente
    'frequency_total', 'ticket_avg', 'service_interval_mean_days', 'service_interval_std_days',
    # Cliente
    'edad', 'genero', 'renta_media_estimada', 'renta_desconocida', 'status_social',
    # Producto
    'modelo', 'equipamiento', 'es_electrico', 'kw',
    # Geografía
    'zona',
]
```

**Nota importante:** La lista FEATURES incluye features que el análisis exploratorio marcó explícitamente para descartar (ver sección 5). El usuario tiene la intención de comentar progresivamente las que no aporten.

---

## 5. Decisiones de feature engineering y sus justificaciones

### 5.1 Bloque económico

| Feature | Decisión análisis | En FEATURES |
|---|---|---|
| `pvp` | MANTENER | ✅ |
| `margen_relativo` | MANTENER (absorbe margen_eur) | ✅ |
| `margen_eur_negativo` | CREAR (flag, churn 10.2% vs 8.3%) | ✅ |
| `coste_venta_no_impuestos` | MANTENER | ✅ |
| `forma_pago` | MANTENER (Financiera Marca: 4.7% vs Contado: 10.5%) | ✅ |
| `fue_lead` | DESCARTAR (churn rate idéntico: 8.7% vs 8.9%) | ✅ ⚠️ |
| `lead_compra` | DESCARTAR (churn rate idéntico: 8.8% vs 8.6%) | ✅ ⚠️ |
| `margen_eur`, `margen_eur_bruto`, `descuento_implicito` | DESCARTAR | ❌ |

### 5.2 Bloque garantía y servicios

| Feature | Decisión análisis | En FEATURES |
|---|---|---|
| `en_garantia` | MANTENER (r=-0.246, más discriminante del dataset) | ✅ |
| `mantenimiento_gratuito` | MANTENER con BUG FIX (0/4, no SI/NO) | ✅ |
| `seguro_bateria_largo_plazo` | MANTENER (churn 2.16% vs 9.67%) | ✅ |
| `ext_garantia_tiene` | CREAR (flag any extension) | ✅ |
| `ext_garantia_financiera` | CREAR (churn 4.87% vs 9.78%) | ✅ |
| `garantia_dias_restantes` | CREAR (nuevo en data_prep) | ✅ |
| `fin_garantia` | DESCARTAR (análisis previo: std=0, constante) | ❌ |

**Nota sobre `garantia_dias_restantes`:** En el análisis previo se descartó `dias_hasta_fin_garantia` calculado como `fin_garantia - sales_date` porque tenía varianza cero (siempre 4 años). Pero `garantia_dias_restantes` se calcula como `fin_garantia - base_date` (fecha fija = 2023-12-31), lo cual sí tiene varianza porque cada vehículo tiene una fecha de venta distinta. Son cálculos conceptualmente distintos.

### 5.3 Bloque post-venta — zona de leakage

| Feature | Decisión análisis | En FEATURES |
|---|---|---|
| `days_last_service` | **EXCLUIR — leakage directo** (define el target al 100%) | ✅ ⚠️⚠️ |
| `sin_revisiones` | MANTENER (codifica regla de negocio) | ✅ |
| `revisiones` | MANTENER con cautela (corr real -0.080) | ✅ |
| `km_medio_por_revision` | **EXCLUIR — correlación espuria** (corr global 0.259, corr real -0.004) | ✅ ⚠️⚠️ |
| `km_ultima_revision` | EXCLUIR (corr real -0.069, contraintuitiva) | ✅ ⚠️ |

### 5.4 Bloque engagement

| Feature | Decisión análisis | En FEATURES |
|---|---|---|
| `encuesta_cliente_zona_taller` | **EXCLUIR** (corr real ≈ 0, métrica de zona no individual) | ✅ ⚠️ |
| `tiene_queja` | **EXCLUIR** (corr=-0.008, 57% NaN, churn rates casi idénticos) | ✅ ⚠️ |

### 5.5 Bloque clientes

| Feature | Decisión análisis | En FEATURES |
|---|---|---|
| `edad` | MANTENER (r=-0.079, relación monotónica) | ✅ |
| `genero` | MANTENER (diferencia M/F real, 1.4pp) | ✅ |
| `renta_media_estimada` | MANTENER (corr real ≈ 0, útil con el flag) | ✅ |
| `renta_desconocida` | CREAR (renta=0 → churn 5.9% vs 9.6%) | ✅ |
| `status_social` | MANTENER (spread 3.9pp) | ✅ |

### 5.6 Bloque productos

| Feature | Decisión análisis | En FEATURES |
|---|---|---|
| `modelo` | MANTENER (spread 0.03%–24%, la más discriminante) | ✅ |
| `equipamiento` | MANTENER ordinal (spread 3.2%–17%) | ✅ |
| `es_electrico` | CREAR flag (churn 2.9% vs 9.3%) | ✅ |
| `kw` | MANTENER (r=-0.098, complementario a equipamiento) | ✅ |
| `fuel` | DESCARTAR (sustituida por `es_electrico`) | ❌ |

### 5.7 Bloque geografía

| Feature | Decisión análisis | En FEATURES |
|---|---|---|
| `zona` | **DESCARTAR** (spread 0.48pp, dentro del ruido estadístico) | ✅ ⚠️ |
| `tienda_desc` | DESCARTAR (spread 1.36pp mínimo) | ❌ |

### 5.8 Agregados RFM (nuevos)

| Feature | Decisión análisis | En FEATURES |
|---|---|---|
| `frequency_total` | CREAR (nº compras históricas del cliente) | ✅ |
| `tenure_days` | CREAR (días desde primera compra) | ✅ |
| `ticket_avg` | CREAR (ticket medio histórico) | ✅ |
| `service_interval_mean_days` | CREAR (intervalo medio entre compras) | ✅ |
| `service_interval_std_days` | CREAR (irregularidad del intervalo) | ✅ |

---

## 6. Modelado — configuración actual

### 6.1 Grupos de features en `modeling.ipynb`

```python
num_features = [
    'pvp', 'margen_relativo', 'coste_venta_no_impuestos',
    'edad', 'renta_media_estimada', 'kw',
    'dias_desde_compra', 'tenure_days', 'days_last_service',
    'garantia_dias_restantes', 'km_ultima_revision', 'km_medio_por_revision',
    'encuesta_cliente_zona_taller', 'lead_compra',
    'frequency_total', 'ticket_avg',
    'service_interval_mean_days', 'service_interval_std_days',
    'revisiones',
]

cat_ohe_features     = ['forma_pago', 'genero']
cat_ordinal_features = ['equipamiento']
equipamiento_order   = [['Low', 'Mid', 'Mid-High', 'High']]
cat_many_features    = ['status_social', 'modelo', 'zona']
cat_all_features     = cat_ohe_features + cat_ordinal_features + cat_many_features

binary_features = [
    'margen_eur_negativo', 'en_garantia', 'ext_garantia_tiene',
    'ext_garantia_financiera', 'mantenimiento_gratuito', 'seguro_bateria_largo_plazo',
    'sin_revisiones', 'tiene_queja', 'fue_lead', 'renta_desconocida', 'es_electrico',
]
```

### 6.2 Pipelines por modelo

**Logistic Regression:**
```python
lr_preprocessor = ColumnTransformer([
    ('num', StandardScaler(),                                                 num_features),
    ('ohe', OneHotEncoder(drop='first', handle_unknown='ignore', sparse_output=False),
                                                                              cat_ohe_features + cat_many_features),
    ('ord', OrdinalEncoder(categories=equipamiento_order),                    cat_ordinal_features),
    ('bin', 'passthrough',                                                    binary_features),
], remainder='drop')

lr_pipeline = Pipeline([
    ('prep', lr_preprocessor),
    ('clf',  LogisticRegression(class_weight='balanced', max_iter=2000, C=0.1, random_state=42)),
])
```

**Random Forest:**
```python
rf_preprocessor = ColumnTransformer([
    ('num', 'passthrough',                                                    num_features),
    ('cat', OrdinalEncoder(handle_unknown='use_encoded_value', unknown_value=-1),
                                                                              cat_ohe_features + cat_many_features),
    ('ord', OrdinalEncoder(categories=equipamiento_order),                    cat_ordinal_features),
    ('bin', 'passthrough',                                                    binary_features),
], remainder='drop')

rf_pipeline = Pipeline([
    ('prep', rf_preprocessor),
    ('clf',  RandomForestClassifier(n_estimators=300, max_depth=10,
                                    min_samples_leaf=10, class_weight='balanced',
                                    random_state=42, n_jobs=-1)),
])
```

**XGBoost:** mismo preprocessor que RF.
```python
XGBClassifier(scale_pos_weight=10.40, n_estimators=300, max_depth=6,
              learning_rate=0.05, subsample=0.8, colsample_bytree=0.8, random_state=42)
```

**LightGBM:** preprocessing especial (dtype `category` nativo).
```python
def prep_lgbm(X):
    feature_cols = num_features + cat_all_features + binary_features
    X_out = X[feature_cols].copy()
    for col in cat_all_features:
        X_out[col] = X_out[col].astype('category')
    return X_out

lgbm_model = LGBMClassifier(is_unbalance=True, n_estimators=300, max_depth=6,
                              learning_rate=0.05, subsample=0.8, colsample_bytree=0.8,
                              random_state=42, verbose=-1)
```

### 6.3 Función de evaluación

```python
def evaluate_model(name, model, X_te, y_te):
    y_proba = model.predict_proba(X_te)[:, 1]
    roc_auc = roc_auc_score(y_te, y_proba)
    pr_auc  = average_precision_score(y_te, y_proba)
    precision_arr, recall_arr, thresholds = precision_recall_curve(y_te, y_proba)
    f1_arr = 2 * precision_arr * recall_arr / (precision_arr + recall_arr + 1e-8)
    opt_idx       = np.argmax(f1_arr[:-1])
    opt_threshold = float(thresholds[opt_idx])
    opt_f1        = float(f1_arr[opt_idx])
    y_pred = (y_proba >= opt_threshold).astype(int)
    return {'model': name, 'roc_auc': roc_auc, 'pr_auc': pr_auc,
            'opt_threshold': opt_threshold, 'f1_opt': opt_f1,
            'y_proba': y_proba, 'y_pred': y_pred}
```

### 6.4 Resultados obtenidos (con feature set ANTERIOR — 17 features, sin las nuevas 36)

| Modelo | ROC-AUC | PR-AUC | F1 (umbral óptimo) | Umbral |
|---|---|---|---|---|
| **Random Forest** | **0.8489** | **0.3017** | **0.3964** | 0.672 |
| XGBoost | 0.8482 | 0.3036 | 0.3959 | 0.729 |
| LightGBM | 0.8470 | 0.2990 | 0.3912 | 0.706 |
| Logistic Regression | 0.8346 | 0.2812 | 0.3823 | 0.703 |

**Modelo ganador: Random Forest** (mínima ventaja sobre XGBoost).

### 6.5 Predicciones sobre scoring — problema observado

Con el feature set anterior, las predicciones sobre `scoring.csv` producen **0 clientes con churn predicho** (0.0%) con probabilidades muy bajas (máximo ~0.4). Los clientes de scoring son nuevos (días desde compra muy pequeños), lo que causa sesgo sistemático del modelo.

---

## 7. Tratamiento de nulos en scoring vs train

| Variable | En train | En scoring | Imputación |
|---|---|---|---|
| `days_last_service` | ~47% NaN (clientes sin revisiones) | 100% NaN (columna no existe en `new_sales`) | fillna(0) |
| `fue_lead` | disponible | 100% NaN | fillna(0) |
| `queja` / `tiene_queja` | 57.4% NaN → se convierte a 0 vía `queja == 'SI'` | igual | NaN → 0 (False) |
| `genero` | 1.5% NaN | igual | fillna('Desconocido') |
| `status_social` | 22% NaN | igual | fillna('Sin_dato') |
| `renta_media_estimada` | 22.5% = 0 (missing enmascarados) | igual | fillna(0) |
| `encuesta_cliente_zona_taller` | NaN → median train | NaN → median train | ⚠️ la mediana se calcula sobre todo df, no solo train |
| `garantia_dias_restantes` | NaN si fin_garantia era NULL | igual | fillna(-999) |
| Agregados RFM | NaN si cliente sin historial | NaN si cliente nuevo (no estaba en sales) | fillna(0) |

---

## 8. Áreas de riesgo y posibles errores — para que revises

### 🔴 Crítico — Leakage confirmado

**8.1 `days_last_service` como feature:**
- Esta variable define directamente el target: todos los `churn=Y` tienen `days_last_service > 400` sin excepción.
- En scoring, se imputa con 0 (clientes nuevos sin historial de taller).
- El modelo aprende: `days_last_service > 400 → churn`. En scoring todos valen 0 → churn predicho ≈ 0 para todos. Esto explica el problema de 0 churn predichos observado antes (con solo 17 features), pero ahora se ha añadido de vuelta al feature set con las 36 features.

**8.2 Leakage en los agregados RFM:**
- `frequency_total`, `ticket_avg`, `service_interval_mean_days`, `service_interval_std_days` y `first_sale_date` se computan sobre **todo el histórico de `sales`**, incluyendo la transacción actual.
- Para clientes con múltiples compras, los agregados calculados desde la query SQL incluyen datos de compras **posteriores** a la que se está prediciendo (la query no filtra por fecha anterior a `sales_date` de cada fila).
- ¿Es esto leakage? Depende de si el scoring se hace en el momento de cada venta o sobre el snapshot completo. Si se hace en tiempo real en el momento de la venta, estos agregados deben excluir la transacción actual y las posteriores.

**8.3 `km_medio_por_revision` — correlación espuria:**
- La correlación global con churn es 0.259, pero es 100% artificial (el grupo sin revisiones tiene `km_medio = 0` y `churn = 0` siempre).
- La correlación real entre clientes CON revisiones es -0.004 (prácticamente cero).
- Si se incluye en el modelo, el modelo puede aprender la correlación espuria en lugar de la real.

### 🟠 Importante — Decisiones cuestionables

**8.4 `garantia_dias_restantes` — posible redefinición inconsistente:**
- El análisis descartó `dias_hasta_fin_garantia = fin_garantia - sales_date` por std=0 (siempre 4 años exactos).
- Pero `garantia_dias_restantes = fin_garantia - base_date` sí tiene varianza (la fecha de referencia es fija = 2023-12-31, y cada venta tiene distinta fecha → distintos días restantes).
- Sin embargo, esta variable es una función casi perfecta de `dias_desde_compra` (clientes que compraron hace más tiempo tienen menos garantía restante). ¿Hay multicolinealidad real entre ambas? ¿Aporta algo independiente?

**8.5 `tenure_days` — posible interpretación incorrecta:**
- Se calcula como `sales_date - first_sale_date` (días desde la primera compra del cliente).
- Para la mayoría de filas de clientes con una sola compra, `tenure_days = 0` (la venta actual es la primera y única).
- Para clientes multi-compra, la fila de la primera compra también tendrá `tenure_days = 0`.
- ¿Cuál es el valor real de esta variable? ¿No correlaciona simplemente con si el cliente es recurrente o no?
- En `tratar_nulos`, se imputa con `fillna(0)`. Pero `tenure_days` viene de `sales_date - first_sale_date`, y `first_sale_date` proviene del join con `df_cust_agg`. Si el cliente no tiene historial en `sales` (imposible para train pero posible para scoring), se rellena con 0, lo que tiene sentido.

**8.6 `tiene_queja` — tratamiento de NaN ambiguo:**
- `queja` tiene 57.4% NaN.
- El código: `df['tiene_queja'] = (df['queja'] == 'SI').astype(int)` convierte NaN → 0 (False).
- Esto hace que NaN (desconocido) y 'NO' sean indistinguibles (ambos → 0).
- En `tratar_nulos` no hay imputación explícita de `queja` porque `tiene_queja` ya está creado.
- El análisis exploratorio muestra: churn(NaN) = 8.0%, churn(NO) = 10.6%, churn(SI) = 8.2%. Son distribuciones distintas: fusionarlas puede perder señal.

**8.7 `encuesta_cliente_zona_taller` — métrica de zona, no individual:**
- El análisis concluye que esta encuesta es una métrica agregada por zona de taller, no una puntuación individual del cliente.
- Todos los registros, incluyendo los 27.070 sin revisiones (que nunca fueron al taller), tienen valores para esta variable.
- Esto sugiere que su información ya está capturada por `zona`. Si ambas se incluyen, pueden estar duplicando información o introduciendo ruido.
- La imputación de NaN con la mediana del dataframe completo (en `tratar_nulos`) calcula la mediana sobre todo el dataset antes del split train/test → **leakage de preprocessing** (la mediana del test contamina el train).

**8.8 `zona` — incluida pese al análisis:**
- El análisis determinó que el spread de 0.48pp entre zonas (ESTE 9.04%, NORTE 8.90%, CENTRO 8.63%, SUR 8.56%) está dentro del ruido estadístico (error estándar de la proporción ~0.4% con n>5.000 por zona).
- Se ha añadido al feature set de modelado como `cat_many_features` con OrdinalEncoder para RF/XGB.
- Riesgo: puede añadir ruido o capturar patrones del dataset que no generalizan.

**8.9 `fue_lead` y `lead_compra` — incluidas pese al análisis:**
- El análisis determinó que la diferencia de churn rate es <0.3pp (prácticamente cero).
- Se han añadido al feature set.

### 🟡 Menor — Posibles mejoras

**8.10 Encoding de `modelo` en Logistic Regression:**
- `modelo` se pasa a `OneHotEncoder` junto con `cat_many_features`.
- Algunos modelos (F con n=29, G con n=25) tienen muy pocos ejemplos → las categorías OHE tendrán muy poco soporte.
- El análisis recomendaba target encoding con smoothing para estos modelos de baja frecuencia. El OHE puede sobreajustar.

**8.11 Imputación de `encuesta_cliente_zona_taller` con mediana global:**
- `df['encuesta_cliente_zona_taller'] = df['encuesta_cliente_zona_taller'].fillna(df['encuesta_cliente_zona_taller'].median())`
- Esta línea se ejecuta sobre el dataframe completo (antes del split train/test en `data_preparation.ipynb`).
- Significa que la mediana del test set está influyendo en la imputación del train set → leakage menor de preprocessing.

**8.12 `scale_pos_weight` calculado sobre `y_train` — correcto:**
- `scale_pos_weight = (y_train == 0).sum() / y_train.sum()` → 10.40
- Correcto: se calcula después del split, sobre train solamente.

**8.13 Umbral óptimo buscado en test set:**
- El umbral óptimo (el que maximiza F1) se busca evaluando sobre el test set.
- Para uso en producción, esto puede estar ligeramente sobreajustado al test. Lo ideal sería buscarlo con cross-validation o en un conjunto de validación separado.

**8.14 `dias_desde_compra` — sesgo en scoring:**
- Los clientes de scoring son nuevos (acaban de comprar), por lo que `dias_desde_compra` será pequeño (decenas de días).
- El modelo entrenado en `sales` ve clientes con `dias_desde_compra` de hasta ~2.000 días.
- Si el modelo aprende que mayor `dias_desde_compra` → mayor churn (tiene correlación r=0.287), los clientes de scoring sistemáticamente recibirán probabilidades bajas.
- Esto agrava el problema de 0 churns predichos.

**8.15 Clientes de scoring con `customer_id` no visto en `sales`:**
- Los agregados RFM (`frequency_total`, `ticket_avg`, etc.) se imputan con 0 si el cliente no tiene historial en `sales`.
- Estos son clientes completamente nuevos. ¿Es 0 la imputación correcta para `frequency_total`? En realidad deberían tener `frequency_total = 1` (la venta actual de new_sales es su primera compra). Pero la query SQL solo computa desde `SCHEMA.sales`, no desde `new_sales`.

---

## 9. Preguntas abiertas (sin resolver en el proyecto)

1. ¿Se aplicará SMOTE u oversampling además de `class_weight`/`scale_pos_weight`?
2. ¿Se hará hyperparameter tuning (GridSearch / Optuna)?
3. ¿Qué umbral de decisión usar en producción (depende del coste de FP vs FN del negocio)?
4. ¿El modelo es **prospectivo** (predice en el momento de la venta) o **retrospectivo** (predice con historial post-venta)? Esto condiciona qué features son válidas.
5. ¿Debería aplicarse un filtro de antigüedad mínima? Hay ~21.700 clientes en train con `dias_desde_compra < 600` que son "imposibles de churnar" porque su ventana temporal es insuficiente. Están etiquetados como no-churn pero son negativos no informativos.

---

## 10. Información técnica adicional

- **Python:** sklearn, XGBoost, LightGBM, pandas, numpy, matplotlib, seaborn
- **Base de datos:** PostgreSQL 5433 en Docker local
- **Archivos de entrada:** `data/warehouse/train.csv` (58.049 × 37), `data/warehouse/scoring.csv` (10.000 × 37)
- **Archivos de salida:** `data/warehouse/predictions.csv` (code, churn_proba, churn_pred)
- Los notebooks están en la raíz del proyecto. No hay tests unitarios.

---

**Por favor, identifica todos los errores que encuentres, priorizados por severidad (crítico / importante / menor), e indica exactamente dónde está el problema y cómo corregirlo.**