# Modeling — Churn Prediction

## Dataset de entrada

`data/warehouse/train.csv` — 58.049 filas, 22 features + `churn` (8.8% positivos).

Features categóricas que requieren encoding: `forma_pago`, `genero`, `status_social`, `modelo`, `equipamiento`.
Features numéricas y flags binarios: el resto.

---

## 1. Split train / test

- Estratificado por `churn` para preservar el ratio 8.8%
- 80% train / 20% test
- `random_state=42`
- El split se hace **antes** de cualquier preprocessing para evitar data leakage

---

## 2. Modelos a entrenar

| # | Modelo | Motivo |
|---|---|---|
| 1 | Logistic Regression | Baseline interpretable |
| 2 | Random Forest | Robusto, maneja bien mixto numérico/categórico |
| 3 | XGBoost | Estándar para tabular con desbalanceo |
| 4 | LightGBM | Más rápido que XGBoost, soporte nativo de categorías |

---

## 3. Preprocessing por modelo

Cada modelo usa un `sklearn.Pipeline` propio. El encoding se decide aquí, no en el feature engineering.

### Logistic Regression
- Categóricas → `OneHotEncoder(drop='first')`
- Numéricas → `StandardScaler`
- Desbalanceo → `class_weight='balanced'`

### Random Forest
- Categóricas → `OrdinalEncoder`
- Numéricas → sin escalar
- Desbalanceo → `class_weight='balanced'`

### XGBoost
- Categóricas → `OrdinalEncoder`
- Numéricas → sin escalar
- Desbalanceo → `scale_pos_weight = n_negativos / n_positivos`

### LightGBM
- Categóricas → pasar directamente como `category` dtype
- Numéricas → sin escalar
- Desbalanceo → `is_unbalance=True`

---

## 4. Métricas de evaluación

El dataset está desbalanceado (8.8% churn). `accuracy` no es útil aquí.

| Métrica | Motivo |
|---|---|
| **ROC-AUC** | Métrica principal — ranking general del modelo |
| **PR-AUC** | Más informativa que ROC en datasets desbalanceados |
| **F1 en umbral óptimo** | Equilibrio precision/recall en el punto de corte elegido |
| **Matriz de confusión** | Para entender falsos positivos y falsos negativos |

Umbral de decisión: por defecto 0.5, pero se buscará el umbral óptimo en la curva PR.

---

## 5. Comparación y selección

Tabla resumen con ROC-AUC y PR-AUC en test para todos los modelos.
El modelo ganador se usará para generar predicciones sobre `scoring.csv`.

---

## Pendiente de decidir

- ¿Aplicar SMOTE u oversampling además de class_weight?
- ¿Hacer hyperparameter tuning (GridSearch / Optuna) o partir de defaults razonables?
- ¿Qué umbral de decisión usar en producción (coste de FP vs FN)?
