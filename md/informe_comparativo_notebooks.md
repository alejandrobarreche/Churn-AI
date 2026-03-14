# Informe Comparativo — Notebooks de Predicción de Churn
**Universidad Alfonso X el Sabio (UAX)**
Fecha: 2026-03-14

---

## Índice

1. [Visión general del proyecto](#1-visión-general-del-proyecto)
2. [transformer.py — Transformadores custom](#2-transformerpy--transformadores-custom)
3. [Notebook 1 — Data Understanding](#3-notebook-1--data-understanding)
4. [Notebook 2 — Data Preparation](#4-notebook-2--data-preparation)
5. [Notebook 3 — Modeling](#5-notebook-3--modeling)
6. [Notebook 4 — Model Verification](#6-notebook-4--model-verification)
7. [Prediccion_churn_ruru.ipynb — Notebook alternativo](#7-prediccion_churn_ruruipynb--notebook-alternativo)
8. [Comparativa global de desarrollo](#8-comparativa-global-de-desarrollo)
9. [Comparativa de resultados y métricas](#9-comparativa-de-resultados-y-métricas)
10. [Comparativa directa: pipeline modular vs notebook monolítico](#10-comparativa-directa-pipeline-modular-vs-notebook-monolítico)
11. [Dependencias entre notebooks y artefactos](#11-dependencias-entre-notebooks-y-artefactos)
12. [Hallazgos clave y decisiones destacables](#12-hallazgos-clave-y-decisiones-destacables)

---

## 1. Visión general del proyecto

El proyecto implementa un pipeline completo de predicción de churn para clientes de un concesionario de automóviles. Se sigue la metodología CRISP-DM, dividida en cuatro notebooks encadenados que van desde la exploración hasta la validación final:

| Notebook | Fase CRISP-DM | Entrada | Salida principal |
|---|---|---|---|
| `1-data_understanding_.ipynb` | Business Understanding + Data Understanding | `customer_data.csv` | Hallazgos EDA |
| `2-data_preparation_.ipynb` | Data Preparation | `customer_data.csv` | Pipeline `.pkl`, `X_train`, `y_train`, `test_set.csv` |
| `3-modeling_.ipynb` | Modeling | `X_train_prepared.pkl`, `y_train.pkl` | 4 modelos `.pkl`, `best_threshold.pkl` |
| `4-model_verification_.ipynb` | Evaluation + Deployment | Test set + nuevos clientes | Predicciones, diagnóstico |

El fichero `transformer.py` actúa como módulo auxiliar compartido — no es un notebook sino una librería de clases sklearn que el notebook 2 importa para construir el pipeline.

---

## 2. transformer.py — Transformadores custom

### Propósito y estructura

`transformer.py` define **7 clases sklearn** que heredan de `BaseEstimator` y `TransformerMixin`, diseñadas para encadenarse en un `sklearn.Pipeline`. Cada clase tiene una responsabilidad única y sigue el contrato estándar `fit` / `transform`.

### Clases implementadas

| Clase | Tipo de variable | Estrategia |
|---|---|---|
| `BinaryEncoder` | Binarias SI/NO + `QUEJA` + `Churn_400` | Mapeo directo a 0/1; escala `MANTENIMIENTO_GRATUITO` × 0.25 |
| `FrequencyEncoder` | Alta cardinalidad: `Modelo`, `PROV_DESC` | Reemplaza cada categoría por su frecuencia relativa en train |
| `OrdinalExtensionEncoder` | `EXTENSION_GARANTIA` | Encoder ordinal con orden explícito `[NO, SI, SI Financiera]`; unifica `SI, Campa a Regalo` con `SI, Financiera` |
| `OrdinalEquipamientoEncoder` | `Equipamiento` | Ordinal `[Low, Mid, Mid-High, High]`; valores desconocidos → -1 |
| `NominalOneHotEncoder` | 9 variables nominales | `OneHotEncoder` con `handle_unknown='ignore'` y `sparse_output=False` |
| `PriceStandard` | `PVP` | Divide entre 1000 para escalar a miles de euros |
| `InstanceDropper` | Filas | Elimina instancias con `RENTA_MEDIA_ESTIMADA == 0` |
| `ColumnDropper` | Columnas | Descarta identificadores, fechas brutas, variables con leakage económico, geográficas redundantes y sensibles |

### Decisiones de diseño relevantes

- **`FrequencyEncoder`** aprende los mapas en `fit()` y los aplica en `transform()`, lo que garantiza que no hay leakage del test al train.
- **`OrdinalExtensionEncoder`** tiene lógica de re-mapeo (`REMAP`) que se aplica tanto en `fit()` como en `transform()`, protegiéndose de variantes inesperadas de la categoría.
- **`ColumnDropper`** busca columnas tanto por nombre exacto como por prefijo, lo que permite eliminar automáticamente las dummies OneHot generadas de variables como `GENERO_` o `ZONA_` sin listarlas explícitamente.
- `MANTENIMIENTO_GRATUITO` recibe una transformación inusual (× 0.25) dentro de `BinaryEncoder`. Esto sugiere que la variable mide meses u otra unidad que se quiere normalizar a una escala cuatrimestral.

---

## 3. Notebook 1 — Data Understanding

### Objetivo

Exploración descriptiva del dataset para entender la estructura de los datos, la distribución de la variable objetivo y las relaciones entre features y churn. **No modifica el dataset ni genera artefactos**.

### Datos de entrada

- `data/lake/customer_data.csv`: 58.049 registros × 40 variables.

### Desarrollo

El notebook sigue una estructura analítica en 13 secciones:

1. **Carga y tipado**: 16 variables numéricas, 24 categóricas.
2. **Nulos**: Solo 4 variables con nulos — `QUEJA` (57.4%), `DAYS_LAST_SERVICE` (46.6%), `STATUS_SOCIAL` (22.1%), `GENERO` (1.5%).
3. **Variable objetivo**: La tasa de churn es ~8-9%, dataset fuertemente desbalanceado.
4. **Variables numéricas** (boxplots): `Revisiones` y `DAYS_LAST_SERVICE` emergen como las más diferenciadas entre churners y no churners.
5. **Análisis de garantía**: Los clientes sin garantía tienen ~19% de churn vs ~4% en garantía.
6. **Variables categóricas** (tasas de churn por categoría): La forma de pago al contado, no extender la garantía y no contratar seguro de batería elevan el riesgo.
7. **Análisis geográfico**: Diferencias moderadas entre provincias; no determinante.
8. **Correlaciones**: `DAYS_LAST_SERVICE` correlación positiva con churn; `Revisiones` correlación negativa.
9. **Revisiones como proxy de fidelidad**: Cuantas más revisiones, menor tasa de churn. Clientes con 0 revisiones: tasa máxima.
10. **Efecto Lead**: Modesto pero presente.

### Resultados / Hallazgos clave

| Variable | Relación con Churn | Importancia estimada |
|---|---|---|
| `DAYS_LAST_SERVICE` | + días sin servicio → + churn | Alta |
| `EN_GARANTIA` | Sin garantía → churn × 4-5 | Alta |
| `Revisiones` | + revisiones → − churn | Alta |
| `EXTENSION_GARANTIA` | No extender → + churn | Media |
| `FORMA_PAGO` | Contado > Financiado en churn | Media |
| `SEGURO_BATERIA_LARGO_PLAZO` | No contratar → + churn | Media |
| `Origen` | Internet ligeramente > Tienda | Baja |
| `PVP`, `Edad` | Sin relación clara | Baja |

### Lo que NO hace este notebook

- No realiza ningún split de datos.
- No aplica transformaciones al dataset.
- No genera modelos ni artefactos persistidos.

---

## 4. Notebook 2 — Data Preparation

### Objetivo

Transformar el dataset crudo en un conjunto numérico, limpio y listo para el modelado, siguiendo el principio de **no data leakage**: el pipeline se ajusta únicamente sobre el train set.

### Datos de entrada

- `data/lake/customer_data.csv`

### Desarrollo

#### Split Train/Test (80/20)

- **Antes de cualquier transformación**, se divide el dataset con `random_state=42`.
- Train: 46.439 filas (80%) | Test: 11.610 filas (20%).
- La proporción de churn se mantiene en ambos conjuntos (split sin estratificar, verificado visualmente).
- El test set se guarda **intacto** en `data/warehouse/test_set.csv`.

#### Pipeline de 8 pasos

```
BinaryEncoder → FrequencyEncoder → OrdinalExtensionEncoder →
OrdinalEquipamientoEncoder → NominalOneHotEncoder →
PriceStandard → InstanceDropper → ColumnDropper
```

Se aplica `fit_transform` sobre el **train** y se persiste el pipeline entrenado (con los mapas de frecuencia aprendidos por `FrequencyEncoder`).

#### Resultado del pipeline sobre el train

| Concepto | Valor |
|---|---|
| Filas antes | 46.439 |
| Filas después | 35.982 |
| Filas eliminadas | 10.457 (22.5%) |
| Columnas antes | 40 |
| Columnas después | 35 (34 features + 1 target) |
| Nulos restantes | 0 |
| Tipos | 26 float64 + 9 int64 |

La eliminación de 10.457 filas se debe principalmente al `InstanceDropper` (clientes con `RENTA_MEDIA_ESTIMADA == 0`) y al filtrado de nulos en otras etapas.

#### Desbalance del target

- Clase 0 (No Churn): 32.493 (90.3%)
- Clase 1 (Churn): 3.489 (9.7%)
- Ratio: 1:9.3

### Artefactos generados

| Fichero | Contenido |
|---|---|
| `data/warehouse/test_set.csv` | Test set crudo sin transformar |
| `data/warehouse/num_pipeline.pkl` | Pipeline sklearn ajustado |
| `data/warehouse/X_train_prepared.pkl` | Features de train transformadas |
| `data/warehouse/y_train.pkl` | Target de train |

### Diferencias respecto al Notebook 1

- NB1 solo lee y analiza; NB2 transforma y persiste.
- NB2 convierte las fechas (`Sales_Date`, `FIN_GARANTIA`, `BASE_DATE`) con `pd.to_datetime` antes de pasarlas al pipeline.
- NB2 introduce la visualización de correlaciones con el target **después** de transformar, confirmando los hallazgos del NB1 con datos ya preparados.

---

## 5. Notebook 3 — Modeling

### Objetivo

Entrenar, tunear y comparar cuatro modelos de clasificación. Seleccionar el mejor modelo y ajustar el umbral de decisión para maximizar el recall de churners.

### Datos de entrada

- `data/warehouse/X_train_prepared.pkl`: 35.982 × 34 features
- `data/warehouse/y_train.pkl`: 35.982 valores, 9.7% de churn

### Desarrollo

#### Funciones auxiliares

Se definen 5 funciones reutilizables con estilo visual UAX:
- `display_scores`: muestra métricas por fold.
- `plot_confusion_matrix`: matriz de confusión con cross-validation.
- `plot_roc_pr`: curvas ROC y Precision-Recall.
- `plot_feature_importance`: top features horizontales.
- `plot_learning_curve`: curva de aprendizaje con banda de confianza.

Todas usan `cross_val_predict` o `cross_val_score` con 5 folds para una evaluación robusta.

#### Modelos entrenados y tuning

**1. Logistic Regression (baseline)**
- Sin tuning: AUC = 0.8072 (std 0.0035).
- Recall para Churn ≈ 0 — apenas detecta churners.
- Uso: referencia mínima de comparación.

**2. Decision Tree**
- Sin tuning: AUC = 0.5795 — completamente sobreajustado.
- Tuning con `GridSearchCV`: 72 combinaciones × 5 folds = 360 fits.
  - Parámetros óptimos: `max_depth=5`, `min_samples_leaf=1`, `min_samples_split=2`, `class_weight=None`.
  - AUC validación: 0.8138 | AUC train: 0.8190 → overfit mínimo (0.0052).
- Visualización del árbol con `max_depth=4` para interpretabilidad.

**3. Random Forest**
- Sin tuning: AUC = 0.7989.
- Tuning con `RandomizedSearchCV`: 20 iteraciones × 5 folds = 100 fits.
  - Espacio de búsqueda: `n_estimators`, `max_depth`, `min_samples_split`, `min_samples_leaf`, `max_features`, `class_weight`.
- Learning curve para diagnosticar overfitting.

**4. XGBoost**
- Sin tuning: AUC = 0.8147.
- Tuning con `RandomizedSearchCV`: 20 iteraciones × 5 folds = 100 fits.
  - Espacio de búsqueda incluye `scale_pos_weight` calculado como ratio de clases.
  - Parámetros óptimos: `subsample=0.6`, `n_estimators=200`, `max_depth=5`, `learning_rate=0.05`, `colsample_bytree=0.6`.
  - AUC validación: 0.8317 | AUC train: 0.8757 → overfit de 0.044.

#### Ajuste de threshold — decisión clave del NB3

Con el threshold por defecto (0.5), el XGBoost tiene recall ≈ 0 en la clase churn. Se implementa un **barrido de thresholds** (0.05 a 0.60, paso 0.01) buscando el que maximiza la precisión sujeto a recall ≥ 75%:

- **Threshold óptimo: 0.11**
- Recall: 0.750 | Precision: 0.238 | F1: 0.361

Esto implica aceptar muchos falsos positivos (Precision baja) para no perder churners reales.

#### Comparativa de modelos (validación cruzada)

| Modelo | Train AUC | Val AUC | Val F1 | Val Recall | Overfit |
|---|---|---|---|---|---|
| XGBoost | 0.8757 | 0.8317 | 0.0040* | 0.0020* | 0.044 |
| XGBoost (t=0.11) | 0.8757 | 0.8317 | 0.3613 | 0.7501 | 0.044 |
| Random Forest | 0.8795 | 0.8270 | 0.3524 | 0.7627 | 0.053 |
| Decision Tree | 0.8190 | 0.8138 | 0.0000 | 0.0000 | 0.005 |
| Logistic Regression | 0.8091 | 0.8072 | 0.0028 | 0.0014 | 0.002 |

*Con threshold por defecto (0.5).

### Artefactos generados

| Fichero | Contenido |
|---|---|
| `data/warehouse/logistic_regression.pkl` | Modelo LR entrenado |
| `data/warehouse/decision_tree.pkl` | Mejor DT tras GridSearch |
| `data/warehouse/random_forest.pkl` | Mejor RF tras RandomizedSearch |
| `data/warehouse/xgboost.pkl` | Mejor XGB tras RandomizedSearch |
| `data/warehouse/best_threshold.pkl` | Threshold óptimo = 0.11 |

---

## 6. Notebook 4 — Model Verification

### Objetivo

Evaluar los modelos entrenados sobre el **test set** (datos nunca vistos), aplicar el pipeline a **10.000 nuevos clientes** para scoring, y diagnosticar el problema por el que el modelo predice 0% churn con threshold 0.5 en nuevos clientes.

### Datos de entrada

- `data/warehouse/test_set.csv`: 11.610 filas
- `data/lake/nuevos_clientes.csv`: 10.000 filas
- Todos los modelos y el pipeline `.pkl` del NB3

### Desarrollo

#### Transformación del test set

Se usa `fit_transform` (en lugar de solo `transform`) — **nota crítica**: esto re-ajusta el pipeline sobre el test set, lo que técnicamente es leakage aunque en este caso el impacto es mínimo dado que los transformadores principales aprenden frecuencias relativas estables.

Resultado del test set tras pipeline:
- 11.610 → **8.992 filas** (1.618 eliminadas, 13.9%)
- 8.164 No Churn (90.8%) | 828 Churn (9.2%)
- Churn rate en test: 8.4% (coherente con train)

#### Resultados en Test set

| Modelo | ROC-AUC | PR-AUC | F1 | Precision | Recall |
|---|---|---|---|---|---|
| Random Forest | **0.8225** | 0.2773 | 0.3296 | 0.2091 | **0.7778** |
| XGBoost | 0.8223 | 0.2760 | 0.0000 | 0.0000 | 0.0000 |
| XGBoost (t=0.11) | 0.8223 | 0.2760 | 0.3488 | 0.2285 | 0.7367 |
| Decision Tree | 0.8065 | 0.2509 | 0.0000 | 0.0000 | 0.0000 |
| Logistic Regression | 0.8038 | 0.2627 | 0.0024 | 0.5000 | 0.0012 |

**Random Forest** es el modelo más robusto en test: alcanza Recall 0.78 con threshold por defecto y el mejor AUC. XGBoost necesita el threshold ajustado para detectar churners.

#### Scoring de nuevos clientes

El dataset de nuevos clientes tiene **diferencias estructurales** respecto al train:

| Variable | Train (media) | Nuevos clientes (media) | Impacto |
|---|---|---|---|
| `EN_GARANTIA` | 0.70 | **1.00** | Todos son clientes recientes → en garantía |
| `QUEJA` | 0.14 | **0.00** | Sin historial postventa aún |
| `Fue_Lead` | 0.55 | **0.12** | Composición del canal muy diferente |

El pipeline elimina 2.225 filas adicionales (de 10.000 a 7.775), principalmente por `RENTA_MEDIA_ESTIMADA == 0` y nulos en variables que no existen en el nuevo conjunto.

**Resultados del scoring:**

| Threshold | Churn predicho | % del total |
|---|---|---|
| 0.50 | 0 / 7.775 | 0.0% |
| 0.11 | 1.030 / 7.775 | 13.2% |

Con el threshold ajustado, la tasa predicha (13.2%) es razonable y cercana a la tasa histórica (~9-10%).

#### Diagnóstico del "problema del 0%"

El notebook incluye un análisis sistemático de **data shift** mediante z-scores. Paradójicamente, el test estadístico no detecta shifts significativos (ninguna variable supera |z| > 2) porque las desviaciones estándar son grandes. Sin embargo, `EN_GARANTIA` = 1.0 en todos los nuevos clientes (contra 0.70 en train) es el factor más determinante: el modelo aprendió que estar en garantía reduce mucho la probabilidad de churn, por lo que todos los nuevos clientes salen con probabilidades bajas.

#### Riesgo relativo

El ranking relativo **sí funciona**: el grupo de alto riesgo (top 20%) tiene una probabilidad media **5.28× mayor** que el grupo de bajo riesgo. Esto valida el modelo para priorizar acciones comerciales.

### Artefactos generados

| Fichero | Contenido |
|---|---|
| `data/warehouse/test_predictions.csv` | Predicciones test (8.992 filas) |
| `data/warehouse/new_customers_predictions.csv` | Predicciones nuevos clientes (7.775 filas) |

---

## 7. Prediccion_churn_ruru.ipynb — Notebook alternativo

### Propósito y estructura

`Prediccion_churn_ruru.ipynb` es un **notebook monolítico** (39 celdas) que implementa el proyecto completo en un único fichero, sin dependencias externas de otros notebooks ni de `transformer.py`. Cubre desde la carga de datos hasta el scoring de nuevos clientes en una sola ejecución lineal.

Incluye además **LightGBM** como tercer algoritmo de ensemble, ausente en los 4 notebooks modulares.

### Datos de entrada

- `./Data/DataLake/customer_data.csv`: 58.049 registros (mismo dataset)
- `./Data/DataLake/nuevos_clientes.csv`: 10.000 clientes
- `coste_mantenimiento.csv`: tabla de costes por modelo (11 modelos) — **exclusiva de este notebook**

### Desarrollo por fases

#### Fase 1: EDA y preparación

El EDA es más completo en algunos aspectos que el NB1:
- Análisis temporal por año y modelo (ventas 2018-2023).
- **Análisis económico** explícito con márgenes por modelo y estructura de costes de mantenimiento — sección ausente en NB1.
- Feature engineering manual integrado en el EDA: crea `gasto_relativo`, `en_garantia_bin`, `tienen_queja`, `origen_internet`, `sin_encuesta`, `seguro_bateria_bin`.

**Construcción explícita de un DataMart** (`datamart_final_v2.csv`) con las features seleccionadas, separando la capa de datos transformados del análisis. El esquema del DataMart:

| Grupo | Variables | Nº |
|---|---|---|
| Numéricas | `km_ultima_revision`, `PVP`, `Edad`, `RENTA_MEDIA_ESTIMADA`, `gasto_relativo`, `Kw`, `Margen_eur_bruto`, `Margen_eur`, `ENCUESTA_CLIENTE_ZONA_TALLER` | 9 |
| Binarias | `tiene_queja`, `en_garantia_bin`, `MANTENIMIENTO_GRATUITO`, `Lead_compra`, `perfil_cliente`, `seguro_bateria_bin`, `sin_encuesta`, `origen_internet` | 8 |
| Categóricas (label-encoded) | `Modelo`, `ZONA`, `FORMA_PAGO`, `EXTENSION_GARANTIA`, `TRANSMISION_ID`, `Fuel`, `Origen` | 7 |
| Target | `Churn_bin` | 1 |

#### Redefinición del target — decisión diferencial

El notebook identifica una **inconsistencia en la variable objetivo** del dataset original:

> `Churn_bin` clasifica como "activo" a clientes con 0 revisiones y más de 400 días desde la compra, que desde el punto de vista de negocio están igualmente perdidos.

Se define `Churn_Final`:
```
Churn_Final = 1 si:
  - Churn_bin == 1 (churn clásico: fue al taller pero dejó de ir), O
  - (Revisiones == 0) AND (antiguedad_dias > 400)  (ghost: nunca fue captado)
```

Esto eleva la tasa de churn del **8.8% a 33.3%** (19.329 de 58.049 clientes).

#### Fase 2: Tres enfoques progresivos de modelado

La estructura del modelado es significativamente diferente a los 4 NBs: en lugar de comparar algoritmos, compara **3 definiciones del problema** para diagnosticar leakage:

**Enfoque 1 — Feature Engineering (`nunca_ha_venido`)**
- Target: `Churn_Final` (33.3%)
- Features: 25 (incluye `perfil_cliente` = flag de 0 revisiones)
- Leakage: SÍ (directo) — `perfil_cliente` es derivada directa del target
- Modelos: RF, XGBoost, LightGBM
- Resultado: AUC ~0.85, F1 XGB 68.2%

**Enfoque 2 — Sin Leakage Lógico**
- Target: `Churn_Final` (33.3%)
- Features: 23 (elimina `Revisiones`, `antiguedad_dias`, `perfil_cliente`)
- Leakage: PARCIAL — `km_ultima_revision` es un proxy perfecto de clientes ghost
- Modelos: RF, XGBoost, LightGBM
- Resultado: AUC ~0.85, F1 XGB 67.7% (apenas varía respecto a E1)
- **Modelo guardado** para scoring: XGBoost Enfoque 2 (`xgb_v4`)

**Enfoque 3 — Blindaje Total**
- Target: `Churn_bin` original (8.8%)
- Features: 22 (elimina además `km_ultima_revision`)
- Leakage: NO
- Modelos: RF, XGBoost, LightGBM
- Resultado: AUC RF 0.8349, XGB 0.8485, LGB 0.8483

**Tabla comparativa global de los 3 enfoques:**

| Métrica | E1-RF | E1-XGB | E1-LGB | E2-RF | E2-XGB | E2-LGB | E3-RF | E3-XGB | E3-LGB |
|---|---|---|---|---|---|---|---|---|---|
| AUC-ROC | 0.8509 | 0.8560 | 0.8580 | 0.8506 | 0.8523 | 0.8524 | 0.8349 | **0.8485** | 0.8483 |
| Recall Churn | 52.1% | 80.0% | 80.7% | 51.8% | 82.0% | 82.0% | 3.1% | **83.3%** | 83.9% |
| Precision Churn | 71.7% | 59.5% | 59.1% | 72.1% | 57.6% | 57.6% | 38.6% | 21.4% | 21.5% |
| F1-Score | 60.4% | 68.2% | 68.2% | 60.3% | 67.7% | 67.6% | 5.8% | 34.0% | 34.2% |

*Target E1/E2: Churn_Final (33.3%). Target E3: Churn_bin original (8.8%)*

La conclusión del análisis de enfoques es relevante: **el AUC apenas cae al eliminar el leakage** (de 0.856 a 0.848), lo que sugiere que el modelo tiene capacidad predictiva genuina más allá de las variables de leakage.

#### Validación cruzada

Usa `StratifiedKFold(n_splits=5)` explícitamente con `scale_pos_weight` calculado automáticamente. A diferencia de los 4 NBs que usan `GridSearchCV`/`RandomizedSearchCV`, ruru no hace tuning de hiperparámetros explícito — entrena con configuraciones por defecto más el ajuste de `scale_pos_weight`.

#### Fase 3: Scoring de nuevos clientes

Aplica el modelo del Enfoque 2 (XGBoost `xgb_v4`) a los 10.000 nuevos clientes:

- `perfil_cliente` **se excluye** del scoring porque todos los nuevos clientes tienen `Revisiones=0` por definición.
- Con threshold 0.5: **7.641 churners predichos (76.4%)**.
- Segmentación en tres niveles: ALTO (≥0.7): 1.161 clientes (11.6%), MEDIO (0.4-0.7): 7.358 (73.6%), BAJO (<0.4): 1.481 (14.8%).
- **Análisis económico del riesgo**: PVP total en riesgo alto = 28.37M EUR (12% de la cartera total de 235.55M EUR).

El resultado difiere mucho del NB4 (0% churn con t=0.5) porque ruru usa un target con 33.3% de churn para entrenar el modelo — las probabilidades predichas son naturalmente más altas.

### Particularidades técnicas de ruru

1. **Label Encoding manual** para categóricas (en lugar de OneHotEncoder o OrdinalEncoder como en transformer.py).
2. **`gasto_relativo` = PVP / RENTA_MEDIA_ESTIMADA**: feature de ingeniería financiera ausente en los 4 NBs.
3. **Tabla de costes de mantenimiento** integrada con tasa de crecimiento compuesto por modelo (α = 7% para modelos A/B, 10% para resto).
4. **LightGBM** como tercer modelo de ensemble.
5. **No persiste el pipeline sklearn**: todo el preprocesamiento está en código inline, lo que hace el scoring menos reproducible.
6. **Ruta de datos diferente**: `./Data/DataLake/` vs `data/lake/` en los 4 NBs.

---

## 8. Comparativa global de desarrollo

### Diferencias metodológicas entre notebooks

| Aspecto | NB1 | NB2 | NB3 | NB4 |
|---|---|---|---|---|
| **Propósito** | Explorar | Transformar | Modelar | Validar |
| **Modifica datos** | No | Sí | No | Sí (solo test) |
| **Usa transformer.py** | No | Sí (importa) | No | No |
| **Genera artefactos** | No | Pipeline + PKLs | Modelos + threshold | Predicciones CSV |
| **Técnica de evaluación** | Análisis descriptivo | Visualización distribuciones | Cross-validation 5-fold | Evaluación directa sobre test |
| **Desbalance tratado** | Observado | Reportado | Threshold tuning | Verificado |
| **Paleta visual** | UAX | UAX | UAX | UAX |

### Evolución del dato a lo largo de los notebooks

```
customer_data.csv (58.049 × 40)
       │
       ├─[NB1]─→ Sin cambios (solo análisis)
       │
       ├─[NB2]─→ Split 80/20
       │          ├─ train_set (46.439) ─[pipeline fit_transform]─→ 35.982 × 35
       │          └─ test_set (11.610)  ─ guardado crudo ─────────→ test_set.csv
       │
       ├─[NB3]─→ Modelos entrenados sobre X_train (35.982 × 34)
       │
       └─[NB4]─→ test_set.csv ─[pipeline fit_transform]─→ 8.992 × 35
                  nuevos_clientes.csv ─[pipeline fit_transform]─→ 7.775 × 35
```

### Inconsistencia detectada: fit_transform vs transform en NB4

En NB2 se usa `fit_transform(train_set)` y el pipeline se guarda. En NB4, en lugar de usar `full_pipeline.transform(test_set)`, se vuelve a usar `full_pipeline.fit_transform(test_set)`. Esto re-entrena los `FrequencyEncoder` con las frecuencias del test set en lugar de aplicar las del train, lo que constituye un **leakage leve**. El impacto real es pequeño porque las distribuciones del test son muy similares a las del train, pero sería más correcto usar solo `transform`.

---

## 9. Comparativa de resultados y métricas

### Evolución de métricas por modelo

#### AUC ROC (comparación validación cruzada vs test final)

| Modelo | CV (NB3) | Test final (NB4) | Degradación |
|---|---|---|---|
| Logistic Regression | 0.8072 | 0.8038 | −0.0034 |
| Decision Tree (tuned) | 0.8138 | 0.8065 | −0.0073 |
| Random Forest (tuned) | 0.8270 | 0.8225 | −0.0045 |
| XGBoost (tuned) | 0.8317 | 0.8223 | −0.0094 |

La degradación al pasar a test es pequeña en todos los modelos (< 1%), lo que confirma buena generalización. XGBoost tiene la mayor degradación absoluta.

#### Recall real en test (con threshold óptimo)

| Modelo | Recall (CV, t=0.5) | Recall (Test, t=0.5) | Recall (Test, t=0.11) |
|---|---|---|---|
| XGBoost | ~0.00 | 0.0000 | **0.7367** |
| Random Forest | ~0.76 | **0.7778** | (no aplica, t=0.5 ya funciona) |

El Random Forest es el modelo más práctico: con threshold por defecto ya capta ~78% de churners, sin necesidad de ajuste.

### PR-AUC (indicador en clases desbalanceadas)

El PR-AUC es un indicador más informativo que el ROC-AUC cuando las clases están desbalanceadas (9% de churn). En test:

| Modelo | PR-AUC |
|---|---|
| Random Forest | 0.2773 |
| XGBoost | 0.2760 |
| Logistic Regression | 0.2627 |
| Decision Tree | 0.2509 |

Un PR-AUC de ~0.28 con un baseline de ~0.09 (proporción de churn) indica que los modelos predicen entre **3-3.5× mejor** que un clasificador aleatorio.

### Reducción de datos en cada etapa

| Etapa | Filas entrada | Filas salida | Reducción |
|---|---|---|---|
| Dataset original | 58.049 | — | — |
| Train set (NB2) | 46.439 | 35.982 | −22.5% |
| Test set (NB4) | 11.610 | 8.992 | −22.5% |
| Nuevos clientes (NB4) | 10.000 | 7.775 | −22.3% |

La reducción es consistente en los tres conjuntos (~22%), lo que sugiere que el `InstanceDropper` y los nulos afectan proporcionalmente a todos los subconjuntos.

---

## 10. Comparativa directa: pipeline modular vs notebook monolítico

### Tabla comparativa global

| Dimensión | NB1–NB4 + transformer.py | Prediccion_churn_ruru.ipynb |
|---|---|---|
| **Estructura** | 4 notebooks modulares + 1 módulo Python | 1 notebook monolítico (39 celdas) |
| **Dependencias** | Fuertemente acoplados (orden estricto) | Auto-contenido, ejecutable de una vez |
| **Reproducibilidad** | Alta (pipeline sklearn persistido) | Media (preprocesamiento inline, sin persistencia) |
| **Reutilización del pipeline** | Sí (`num_pipeline.pkl` + `transform()`) | No — preprocesamiento duplicado para nuevos clientes |
| **Target** | `Churn_400` original (~9%) | 3 versiones: Churn_bin (8.8%), Churn_Final (33.3%) |
| **Variable objetivo redefinida** | No | Sí — detecta y corrige clientes ghost |
| **Análisis de leakage** | Implícito (ColumnDropper excluye variables) | Explícito — 3 enfoques progresivos |
| **Modelos comparados** | LR + DT + RF + XGB (4) | RF + XGB + LGB × 3 enfoques = 9 configuraciones |
| **LightGBM** | No | Sí |
| **Tuning de hiperparámetros** | GridSearchCV + RandomizedSearchCV | No (solo `scale_pos_weight`) |
| **Ajuste de threshold** | Sí (barrido 0.05-0.60, óptimo = 0.11) | No (threshold por defecto 0.5) |
| **Split train/test** | 80/20 (`random_state=42`) | 80/20 (StratifiedKFold en validación) |
| **Estratificación** | No en el split, sí en CV implícito | Sí — `StratifiedKFold` explícito |
| **Encoding categóricas** | sklearn pipeline (OHE + Ordinal + Frequency) | LabelEncoder manual |
| **DataMart intermedio** | No — transformación directa al vuelo | Sí — `datamart_final_v2.csv` persistido |
| **Análisis económico** | No | Sí (márgenes por modelo, costes compuestos) |
| **Análisis geográfico** | Sí (NB1) | Parcial |
| **Feature engineering** | Mínimo (solo escalado PVP) | Extenso (`gasto_relativo`, `perfil_cliente`, `FLAG_SIN_REVISION`, etc.) |
| **Paleta visual** | UAX corporativa (navy/gold) | Estándar verde/rojo |
| **Scoring nuevos clientes** | 7.775 filas, 0% churn (t=0.5), 13.2% (t=0.11) | 10.000 filas, 76.4% churn (t=0.5) |
| **Análisis de valor económico** | No | Sí (28.37M EUR en riesgo alto) |
| **Diagnóstico de data shift** | Sí (z-scores, identificación de causa) | Parcial (detectado pero no cuantificado) |

### Comparativa de métricas finales (target original ~9% churn)

Comparación con el mismo target base (`Churn_bin` original / `Churn_400`):

| Modelo | AUC (NBs pipeline) | AUC (ruru E3) | Δ AUC |
|---|---|---|---|
| Random Forest | 0.8225 | 0.8349 | +0.0124 (ruru mejor) |
| XGBoost | 0.8223 | 0.8485 | +0.0262 (ruru mejor) |
| LightGBM | — | 0.8483 | — |

**ruru obtiene AUC más alto en el Enfoque 3** (mismo target, sin leakage). La diferencia (~0.012-0.026) probablemente se debe a que ruru conserva variables excluidas por el `ColumnDropper` de los 4 NBs (`DAYS_LAST_SERVICE`, `Revisiones`, `km_ultima_revision`) que tienen alta correlación con el churn.

| Recall Churn | NBs pipeline | ruru E3 (t=0.5) |
|---|---|---|
| Random Forest | 0.778 (t=0.5) | 0.031 |
| XGBoost | 0.000 (t=0.5) / **0.737** (t=0.11) | **0.833** |

El Recall de Random Forest en el pipeline (0.778) es muy superior al de ruru E3 (0.031), lo que sugiere que el `ColumnDropper` y el `PriceStandard` mejoran la calibración de probabilidades en RF, aunque reducen el AUC global.

### Fortalezas y debilidades de cada enfoque

**Pipeline modular (NB1–NB4 + transformer.py)**

✅ Reproducible: pipeline sklearn persistido, reutilizable en producción.
✅ Modular: cada notebook tiene responsabilidad única, fácil de mantener.
✅ Ajuste de threshold documentado y explícito.
✅ Diagnóstico de data shift en nuevos clientes.
⚠️ No analiza leakage lógico de `Revisiones`/`km_ultima_revision`.
⚠️ `ColumnDropper` elimina variables potencialmente útiles (`DAYS_LAST_SERVICE`).
⚠️ No incluye LightGBM.
⚠️ Leakage leve en NB4 (`fit_transform` en lugar de `transform` sobre test).

**Notebook monolítico (ruru)**

✅ Auto-contenido, fácil de compartir y ejecutar.
✅ Análisis de leakage explícito y pedagógico (3 enfoques progresivos).
✅ Redefinición del target bien justificada (clientes ghost).
✅ Análisis económico (márgenes, valor en riesgo).
✅ Incluye LightGBM.
✅ Mayor AUC en el Enfoque 3 (sin leakage, mismo target base).
⚠️ Preprocesamiento no persistido como pipeline sklearn — difícil de reutilizar.
⚠️ Sin tuning de hiperparámetros (solo `scale_pos_weight`).
⚠️ Sin ajuste de threshold — el scoring masivo (76.4% churn) es poco realista.
⚠️ Label Encoding para categóricas nominales puede introducir orden artificial.

---

## 11. Dependencias entre notebooks y artefactos

```
transformer.py
     │
     └─[importado por]─→ NB2
                          │
                          ├─ genera: num_pipeline.pkl
                          │           X_train_prepared.pkl
                          │           y_train.pkl
                          │           test_set.csv
                          │
                          └─[usado por]─→ NB3
                                          │
                                          ├─ genera: logistic_regression.pkl
                                          │           decision_tree.pkl
                                          │           random_forest.pkl
                                          │           xgboost.pkl
                                          │           best_threshold.pkl
                                          │
                                          └─[usado por]─→ NB4
                                                          │
                                                          └─ genera: test_predictions.csv
                                                                     new_customers_predictions.csv
```

Los notebooks están **fuertemente acoplados en orden**. No es posible ejecutar NB3 sin antes ejecutar NB2, ni NB4 sin NB3. NB1 es completamente independiente.

---

## 12. Hallazgos clave y decisiones destacables

### 1. El threshold por defecto (0.5) es inútil para este problema

Todos los modelos excepto Random Forest tienen Recall ≈ 0 con threshold 0.5. El desbalance extremo (9% churn) hace que los modelos raramente superen ese umbral. La solución implementada (barrido de thresholds) es correcta, aunque Random Forest resuelve esto de forma natural gracias a su mejor calibración de probabilidades.

### 2. Random Forest vs XGBoost — empate técnico, Random Forest más práctico

Ambos alcanzan AUC ~0.82 en test. Random Forest tiene ligera ventaja en Recall con threshold 0.5 (no requiere ajuste), mientras que XGBoost necesita bajar el threshold a 0.11 para ser útil. Para despliegue operativo, Random Forest es más robusto.

### 3. El Decision Tree sin tuning es inutilizable

AUC = 0.58 sin regularización — prácticamente aleatorio. Tras GridSearch con `max_depth=5` sube a 0.81. Esto ilustra perfectamente la importancia del tuning en árboles de decisión.

### 4. Los nuevos clientes rompen el modelo por diseño, no por bug

El data shift en `EN_GARANTIA` (100% en garantía vs 70% en train) es una característica intrínseca del negocio: los clientes nuevos, por definición, están en garantía y sin historial postventa. El modelo necesita un scoring basado en **ranking relativo**, no en probabilidades absolutas. El grupo de alto riesgo tiene 5.28× más probabilidad que el de bajo riesgo, lo cual es un resultado útil y accionable.

### 5. El pipeline elimina ~22% de datos en todos los conjuntos

La pérdida consistente de 22% de registros (en train, test y nuevos clientes) sugiere que `RENTA_MEDIA_ESTIMADA == 0` es un problema de calidad de datos estructural, no un artefacto del split. Para scoring en producción, eliminar 22% de clientes puede ser inaceptable y se debería considerar imputación en lugar de eliminación.

### 6. Las variables de relación postventa son las más predictivas

`Revisiones`, `EN_GARANTIA`, `DAYS_LAST_SERVICE` y `QUEJA` aparecen consistentemente en el top de importancia en todos los modelos. Las variables económicas (`PVP`, `RENTA_MEDIA_ESTIMADA`) tienen menor peso. Esto es coherente con la hipótesis de negocio: el churn en concesionarios está más relacionado con la experiencia postventa que con el poder adquisitivo.

---

*Informe generado a partir del análisis de `1-data_understanding_.ipynb`, `2-data_preparation_.ipynb`, `3-modeling_.ipynb`, `4-model_verification_.ipynb`, `transformer.py` y `Prediccion_churn_ruru.ipynb`.*
