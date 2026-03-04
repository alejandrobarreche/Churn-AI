# Análisis del `modeling_codex.ipynb` — Qué mejoró, qué sigue mal y por qué

---

## 1. Cambios correctos que Codex implementó

### ✅ 1.1 Fix temporal `base_date` (hallazgo crítico #1 de Codex)

**Antes** (`data_preparation.ipynb`):
```python
base_date = df['sales_date'].max()   # una sola fecha máxima del dataframe
df['dias_desde_compra']      = (base_date - df['sales_date']).dt.days
df['garantia_dias_restantes'] = (df['fin_garantia'] - base_date).dt.days
```

**Después** (`train_codex.csv` / `scoring_codex.csv`):
```python
df['dias_desde_compra']      = (df['base_date'] - df['sales_date']).dt.days
df['garantia_dias_restantes'] = (df['fin_garantia'] - df['base_date']).dt.days
```

**Efecto observable:** El scoring ahora muestra `dias_desde_compra` negativos (-9, -40...) porque los clientes de new_sales compraron en enero-marzo 2024, después del `base_date = 2023-12-31`. Antes eran 0-90 (offset artificial). El cálculo es ahora correcto.

---

### ✅ 1.2 Split por cliente con `GroupShuffleSplit` (hallazgo crítico #3)

**Antes:** `train_test_split` aleatorio por fila → ~25% de filas en test compartían `customer_id` con train.

**Después:**
```python
gss = GroupShuffleSplit(n_splits=1, test_size=0.2, random_state=42)
idx_train, idx_test = next(gss.split(X_all, y_all, groups=customer_id))
```

**Resultado verificado:** `Overlap clientes train-test: 0` y `Overlap clientes val-test: 0`. El leakage por multi-compra del mismo cliente está eliminado.

---

### ✅ 1.3 Split en 3 partes: train / val / test (hallazgo crítico #5)

Antes el umbral se buscaba en el mismo test set donde se reportaban las métricas. Ahora:
- El **umbral óptimo** se selecciona en el conjunto de **validación**
- El **test** solo se toca una vez, para la estimación final
- El modelo ganador se elige por `VAL PR-AUC`, no por `TEST PR-AUC`

---

### ✅ 1.4 Imputación dentro del pipeline (hallazgo menor #2)

Antes: `tratar_nulos()` se ejecutaba sobre el dataframe completo antes del split (leakage menor en la mediana de `encuesta_cliente_zona_taller`).

Ahora: `SimpleImputer(strategy='median')` está encapsulado dentro de cada `ColumnTransformer`, por lo que se ajusta solo con datos de train.

---

### ✅ 1.5 El scoring ya predice churns no triviales

El problema de `0 churns predichos` está resuelto: ahora predice **1.871 churns (18.7%)**. El fix de `base_date` y la eliminación de `days_last_service` son los responsables principales.

---

## 2. Por qué las curvas ROC y PR siguen siendo malas

### Comparación directa de métricas

| Versión | Modelo | ROC-AUC | PR-AUC | Nota |
|---|---|---|---|---|
| `modeling.ipynb` (original) | Random Forest | **0.8489** | **0.3017** | Con leakage parcial |
| `modeling_codex.ipynb` (corregido) | LightGBM | 0.7880 | 0.2282 | Sin leakage, pero... |

La caída es de **-0.061 en ROC-AUC** y **-0.073 en PR-AUC**. El fix fue correcto metodológicamente, pero el rendimiento cayó. Hay tres causas.

---

### 🔴 Causa principal: Codex eliminó features VÁLIDAS que no eran leakage

Codex adoptó un criterio demasiado conservador: eliminó **todas** las features post-venta por considerarlas "retrospectivas", incluyendo variables que **sí están disponibles en `new_sales`** en el momento del scoring.

| Feature | Presente en `new_sales` | Leakage real | Acción de Codex | Consecuencia |
|---|---|---|---|---|
| `days_last_service` | ❌ (no existe) | ✅ **SÍ** (define el target) | Eliminada ✅ | Correcto |
| `en_garantia` | ✅ SÍ | ❌ NO | **Eliminada** ⚠️ | Pierde r=-0.246 |
| `sin_revisiones` | ✅ SÍ | ❌ NO | **Eliminada** ⚠️ | Pierde regla de negocio clave |
| `revisiones` | ✅ SÍ | ❌ NO | **Eliminada** ⚠️ | Pierde señal real (-0.080) |
| `mantenimiento_gratuito` | ✅ SÍ | ❌ NO | **Comentada** ⚠️ | Pierde 0.46% vs 9.91% churn |
| `km_medio_por_revision` | ✅ SÍ | Correlación espuria | Eliminada ✅ | Correcto |

`en_garantia` era la variable más discriminante de todo el dataset (r=-0.246). `sin_revisiones` codifica la regla de negocio exacta: un cliente sin ninguna revisión **nunca puede ser churn** por definición del target (`churn_400 = Y` requiere `revisiones ≥ 1`). Al eliminarla, el modelo pierde el separador más limpio del espacio de decisión.

**Evidencia numérica en train:**
- `sin_revisiones=1`: **0% churn** (27.070 registros, 46.6% del dataset)
- `en_garantia=SI`: 4.15% churn vs `en_garantia=NO`: 19.26% churn
- `mantenimiento_gratuito=1`: 0.46% churn vs `=0`: 9.91% churn

Estas tres features juntas explicaban gran parte del poder predictivo del modelo original. Sin ellas, el modelo trabaja con señales mucho más débiles.

---

### 🔴 Causa secundaria: `dias_desde_compra` fuera de la distribución de entrenamiento en scoring

Con el fix de `base_date`, en scoring `dias_desde_compra` es negativo (-9 a -91 días). En train, `dias_desde_compra` va de ~0 a ~2.000 días. El modelo nunca vio valores negativos durante el entrenamiento.

Codex eliminó `dias_desde_compra` del feature set activo para evitar este shift. Esto es razonable, pero también elimina información temporal valiosa disponible en train.

La solución correcta no es eliminarla, sino reconocer que hay dos poblaciones distintas:
- Train: clientes con compra ya realizada, con historial post-venta acumulado
- Scoring: clientes que acaban de comprar (incluso en el futuro respecto a `base_date`)

Esto apunta a un problema de framing más profundo (ver sección 3).

---

### 🟠 Causa terciaria: los agregados RFM tienen distribución diferente en scoring

Los clientes de `new_sales` que no aparecen en `sales` histórico tienen `frequency_total = 0`, `tenure_days = 0`, `ticket_avg = pvp` (imputado). Esto les da perfil de "cliente nuevo sin historial".

Sin embargo, en train todos los clientes tienen al menos 1 compra registrada en `sales`, por lo que `frequency_total ≥ 1` siempre. El modelo aprende patrones sobre `frequency_total ≥ 1` y encuentra `frequency_total = 0` solo en scoring, fuera de su distribución de entrenamiento.

---

## 3. El problema de fondo: framing de negocio sin resolver

Las malas curvas son el síntoma de un problema conceptual más profundo que ninguna de las dos versiones ha resuelto del todo:

**¿Para qué sirve el modelo?**

| Escenario | ¿Tiene sentido? | Features válidas |
|---|---|---|
| **Retrospectivo:** predecir churn de clientes que ya compraron hace tiempo, usando su historial de taller | ✅ Sí, para campañas de retención sobre cartera existente | `en_garantia`, `sin_revisiones`, `revisiones`, `days_last_service` (excepto leakage directo) |
| **Prospectivo:** predecir churn en el momento exacto de la venta, para nuevas ventas | ✅ Sí, para personalizar contratos o seguimientos | Solo features conocidas al firma: garantía contratada, tipo de pago, producto, perfil cliente |

**El problema:** `train.csv` mezcla clientes con antigüedades de 0 a ~2.000 días. Un cliente con 30 días de antigüedad aún no ha tenido oportunidad de churnar, pero está en el mismo dataset que uno con 1.800 días. El target es válido para los de larga antigüedad; para los nuevos, el `churn=0` es simplemente porque no han tenido tiempo, no porque sean leales.

**Consecuencia directa:** cuando el modelo se evalúa en test (que contiene clientes de todas las antigüedades), las métricas ROC/PR están mezclando dos problemas distintos, y cuando se aplica a scoring (todos nuevos), la distribución es radicalmente diferente a lo que vio en train.

---

## 4. Problema adicional detectado en el scoring: tasa 18.7% (demasiado alta)

El modelo predice **18.7% de churn** en scoring vs **8.8%** en train. Esta sobreestimación es problemática.

**Por qué ocurre:**
- `sin_revisiones` no está en el modelo. En scoring, todos los clientes tienen `revisiones=0` (acaban de comprar). Si esta feature estuviera activa, el modelo diría "sin revisiones → churn=0 siempre", prediciendo 0% churn para todos. Su ausencia hace que el modelo no tenga el freno más importante.
- `en_garantia` no está en el modelo. En scoring, casi todos los clientes deberían estar en garantía (compraron hace semanas). Con `en_garantia=SI` activo, el modelo asignaría probabilidad baja a estos clientes.
- Los clientes de scoring tienen `frequency_total=0` (no tienen historial en `sales`). El modelo puede asociar "ninguna compra previa" con mayor riesgo, al contrario de la lógica real.

---

## 5. Síntesis: qué mantener, qué revisar

### Mantener del codex (correcciones válidas)
- ✅ Cálculo de `dias_desde_compra` y `garantia_dias_restantes` usando `base_date` fila a fila
- ✅ `GroupShuffleSplit` por `customer_id`
- ✅ Split en 3 partes con selección de umbral en validación
- ✅ `SimpleImputer` dentro del pipeline
- ✅ Eliminación de `days_last_service` del feature set (el único leakage real confirmado)

### Revisar / revertir
- ⚠️ **Reintroducir `en_garantia`, `sin_revisiones`, `revisiones`, `mantenimiento_gratuito`** — están en `new_sales` y son legítimas para scoring
- ⚠️ **Decidir qué hacer con `dias_desde_compra`** — ahora es negativo en scoring (shift distribucional), pero puede ser informativo si se acepta que el modelo es "al momento de la venta" (días=0 o negativos para scoring, positivos para train histórico)
- ⚠️ **Investigar la sobreestimación del 18.7%** — comparar distribuciones de features en train vs scoring para identificar qué variable está fuera de rango

### El problema de fondo a decidir
Antes de entrenar cualquier versión nueva, hay que fijar el framing:

> **¿Se entrena y evalúa solo sobre clientes con antigüedad suficiente para churnar (`dias_desde_compra > 400`), aceptando que el scoring será prospectivo con distributional shift?**

Si la respuesta es sí: filtrar train a `dias_desde_compra > 400` y aceptar que en scoring las probabilidades son orientativas (riesgo relativo entre clientes nuevos, no probabilidad absoluta).

Si la respuesta es no: entrenar con todo el dataset pero modelar explícitamente la componente temporal (supervivencia, hazard functions) — lo que está fuera del alcance actual.

---

## 6. Resumen de métricas

| Versión | ROC-AUC | PR-AUC | Churns en scoring | Features activas |
|---|---|---|---|---|
| Original (`modeling.ipynb`) | 0.8489 | 0.3017 | 0 (0.0%) | 17 (prospectivas + algunas retro) |
| Codex (`modeling_codex.ipynb`) | 0.7880 | 0.2282 | 1.871 (18.7%) | 22 (solo prospectivas puras) |

El codex resolvió el problema de scoring (ya predice churns) a costa de perder poder predictivo al eliminar features legítimas. El siguiente paso es reintroducir `en_garantia`, `sin_revisiones`, `revisiones` y `mantenimiento_gratuito` —que existen en `new_sales`— y medir si las curvas recuperan el nivel del original sin perder la corrección del scoring.
