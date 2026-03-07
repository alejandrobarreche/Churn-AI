# Comparación de enfoques: Ruru vs Mi proyecto

---

## 0. Nota previa

Ruru usa el mismo dataset fuente (mismas tablas, mismo número de registros: 58.049 ventas). Sin embargo, su notebook incluye features como `TIPO_CARROCERIA` y `TRANSMISION_ID` que en mi proyecto no aparecen en la consulta SQL sobre la tabla `products`. Esto significa que Ruru extrajo más columnas de los mismos datos, aunque yo no seleccioné esas columnas al construir el dataset analítico. Esta duda queda abierta: esas features podrían existir en `products` pero no fueron consideradas en mi análisis.

---

## 1. La diferencia más importante: el target

Esta es la divergencia raíz de la que depende todo lo demás.

### Ruru — Target redefinido: `Churn_Corregido`

Ruru no usa el label original del dataset. En su lugar, **construye un target nuevo** que captura dos tipos de pérdida:

```
Churn = 1 si:
  - El cliente tiene revisiones Y lleva >400 días sin volver al taller, O
  - El cliente tiene 0 revisiones Y lleva >400 días desde su compra (nunca captado)

Churn = 0 si:
  - El cliente acude regularmente, O
  - El cliente es nuevo (compró hace <400 días)
```

**Resultado:** tasa de churn del **33.3%** (19.329 de 58.049 clientes).

La decisión de negocio implícita es que un cliente que compró un coche y **nunca fue al taller** también es una pérdida, no solo quien fue y dejó de ir. Ruru llama a estos clientes "ghost customers".

### Mi proyecto — Target original: `churn_400`

Uso el label que viene en los datos sin modificarlo. El label original define churn solo como:

```
churn_400 = Y si el cliente fue al taller al menos una vez Y lleva >400 días sin volver
```

Los clientes que nunca fueron al taller están etiquetados como `churn = N` en el dataset original, y mi proyecto respeta esa etiqueta.

**Resultado:** tasa de churn del **8.8%** (5.093 de 58.049 clientes).

### Consecuencia de esta diferencia

Al cambiar el target, Ruru cambia el problema que está resolviendo. Su modelo aprende a distinguir entre "clientes activos" y "clientes perdidos (ya sea por abandono o por no captación)". Mi modelo aprende a distinguir entre "clientes activos" y "clientes que fueron al taller y luego dejaron de volver".

Esto tiene un impacto enorme en las métricas: con 33.3% de churn, el dataset de Ruru está mucho menos desbalanceado que el mío (8.8%), lo que facilita el entrenamiento y hace que los AUC sean más comparables entre modelos.

---

## 2. Features: qué incluye cada proyecto

### Features que solo usa Ruru

| Feature | Cómo se calcula | Por qué yo no la uso |
|---|---|---|
| `gasto_relativo` | `PVP / RENTA_MEDIA_ESTIMADA.clip(lower=1)` | No la construí, es una interacción PVP × renta |
| `TIPO_CARROCERIA` | Columna de la tabla de productos | No la seleccioné en mi query SQL |
| `TRANSMISION_ID` | Columna de la tabla de productos | No la seleccioné en mi query SQL |
| `Fuel` (categoría directa) | Columna original como categórica | Yo la convertí en el flag `es_electrico` |
| `Margen_eur_bruto` | Columna directa de ventas | Yo la descarté por correlación alta con pvp (r=0.637) |
| `EXTENSION_GARANTIA` (directa) | La columna original de la tabla | Yo la descompuse en dos flags: `ext_garantia_tiene` y `ext_garantia_financiera` |
| `perfil_cliente` | `(revisiones == 0).astype(int)` | Yo tengo `sin_revisiones` equivalente, pero lo excluyo en el modelo prospectivo porque es constante en scoring |

### Features que solo uso yo

| Feature | Cómo se calcula | Por qué es diferente |
|---|---|---|
| `margen_relativo` | `margen_eur / pvp` | Normalizo el margen por el precio. Ruru usa `margen_eur_bruto` sin normalizar |
| `margen_eur_negativo` | Flag si `margen_eur < 0` | Ruru no crea este flag aunque analiza el margen negativo |
| `renta_desconocida` | Flag si `renta_media_estimada == 0` | Ruru imputa directamente con mediana. Yo creo un flag explícito porque los ceros tienen comportamiento distinto (5.9% vs 9.6% churn) |
| `ext_garantia_tiene` | Flag si tiene alguna extensión de garantía | En vez de la categoría completa, separo en dos flags |
| `ext_garantia_financiera` | Flag si la extensión es "Financiera" | Solo esta variante discrimina fuertemente (4.87% vs 9.78% churn) |
| `es_electrico` | Flag si `fuel != 'HÍBRIDO'` | Ruru usa `fuel` como categórica completa |
| `frequency_total` | Nº de compras previas del cliente (causal) | RFM: historial del cliente. Ruru no computa agregados cross-cliente |
| `tenure_days` | Días desde la primera compra anterior (causal) | Ídem |
| `ticket_avg` | Ticket medio de compras anteriores (causal) | Ídem |
| `service_interval_mean_days` | Intervalo medio entre compras anteriores (causal) | Ídem |
| `service_interval_std_days` | Irregularidad del intervalo (causal) | Ídem |

### Features compartidas (con diferencias menores)

| Feature | Ruru | Mi proyecto |
|---|---|---|
| `pvp` | Incluida directamente | Incluida directamente |
| `edad` | `Edad` | `edad` |
| `renta_media_estimada` | Imputada con mediana | Incluida + flag `renta_desconocida` |
| `kw` | `Kw` | `kw` |
| `en_garantia` | `en_garantia_bin` — incluida en el modelo | Excluida del modelo prospectivo (constante en scoring: todos=1) |
| `mantenimiento_gratuito` | Incluida directamente | Incluida con bug fix (0/4 en vez de SI/NO) |
| `seguro_bateria_largo_plazo` | `seguro_bateria_bin` — flag creado | Incluida como flag |
| `forma_pago` | OHE con `pd.get_dummies` | OHE con `OneHotEncoder` dentro de pipeline |
| `modelo` | OHE con `pd.get_dummies` | OHE en LR; OrdinalEncoder en RF/XGB; category en LGBM |
| `equipamiento` | OHE con `pd.get_dummies` | OrdinalEncoder con orden explícito Low<Mid<Mid-High<High |
| `zona` | Incluida | Excluida (spread 0.48pp, ruido estadístico) |
| `encuesta_cliente_zona_taller` | Imputada con -999, incluida | Excluida (corr real ≈ 0, métrica de zona no individual) |
| `tiene_queja` | `(QUEJA == 'SI').astype(int)` | `tiene_queja` idéntico, pero excluido del modelo (corr=-0.008) |
| `days_last_service` | Excluida (leakage) | Excluida (leakage) |
| `km_medio_por_revision` | Excluida (derivada de revisiones, leakage) | Excluida (correlación global espuria, corr real ≈ 0) |
| `revisiones` | Excluida en Enfoque 2 (leakage lógico) | Excluida del modelo prospectivo (constante en scoring: todos=0) |

---

## 3. Preprocesado y pipeline

### Ruru — proceso manual, sin sklearn Pipeline

```
1. Merge de tablas con pandas
2. Creación de features derivadas
3. Imputación: fillna con mediana para numéricas, moda para categóricas
4. OneHotEncoding con pd.get_dummies(drop_first=True)
5. train_test_split(test_size=0.2, stratify=y, random_state=42)
6. Entrenamiento directo sobre los arrays resultantes
```

No hay ningún objeto `Pipeline` de sklearn. Todo el preprocesado se aplica al dataset completo antes del split, lo que significa que la imputación con mediana/moda usa información del test set (leakage menor de preprocesado, aunque en la práctica el impacto es pequeño).

### Mi proyecto — sklearn Pipeline con ColumnTransformer

```
1. Merge de tablas con SQL
2. Creación de features en feature_engineering()
3. GroupShuffleSplit por customer_id (train/val/test)
4. Dentro del Pipeline por modelo:
   - SimpleImputer(strategy='median') para numéricas
   - SimpleImputer(strategy='most_frequent') para categóricas
   - StandardScaler para numéricas (solo en Logistic Regression)
   - OneHotEncoder para categóricas OHE
   - OrdinalEncoder con orden explícito para equipamiento
   - OrdinalEncoder para categóricas de muchos valores en RF/XGB
   - dtype 'category' nativo para LightGBM
5. El imputer se ajusta SOLO con datos de train (fit_transform en train, transform en val/test)
```

La ventaja del Pipeline es que garantiza que no hay fuga de información del test al preprocesado.

---

## 4. Split y validación

| Aspecto | Ruru | Mi proyecto |
|---|---|---|
| Método de split | `train_test_split` aleatorio por fila | `GroupShuffleSplit` por `customer_id` |
| Leakage por multi-compra | Sí: ~25% de filas en test comparten cliente con train | No: overlap = 0 clientes |
| Proporciones | 80/20 | ~64/16/20 (train/val/test) |
| Conjunto de validación | No separado explícitamente | Separado para selección de umbral y modelo |
| Cross-validation | Sí: 5-fold StratifiedKFold para estimar varianza | No |
| Selección de umbral | 0.5 por defecto (implícito en `classification_report`) | Optimizado en validación (maximiza F1 en curva PR) |
| Selección del modelo ganador | El de mayor AUC en test | El de mayor PR-AUC en validación |

El split por `customer_id` en mi proyecto es metodológicamente más correcto porque hay 44.053 clientes únicos en 58.049 filas (algunos clientes tienen 2+ compras). Sin un split por cliente, un cliente puede aparecer en train y en test, lo que permite que el modelo aprenda patrones específicos de ese cliente que luego "reconoce" en el test set.

---

## 5. Modelos y configuración

### Ruru — Enfoque 2 (sin leakage, modelo final)

| Modelo | Hiperparámetros clave | Desbalanceo |
|---|---|---|
| Random Forest | `n_estimators=200`, `class_weight='balanced'` | `class_weight='balanced'` |
| XGBoost | `max_depth=3`, `lr=0.05`, `reg_alpha=1.0`, `reg_lambda=5.0`, `min_child_weight=10` | `scale_pos_weight=2.5` |
| LightGBM | `max_depth=3`, `lr=0.05`, `reg_alpha=1.0`, `reg_lambda=5.0`, `min_child_weight=10` | `scale_pos_weight=2.5` |

Ruru añade **regularización fuerte** en XGBoost y LightGBM: `reg_alpha=1.0` (L1), `reg_lambda=5.0` (L2), `max_depth=3`. Esto reduce el sobreajuste pero también la capacidad del modelo.

El `scale_pos_weight=2.5` refleja que el dataset redefinido tiene 33% churn (mucho menos desbalanceado que el original).

### Mi proyecto

| Modelo | Hiperparámetros clave | Desbalanceo |
|---|---|---|
| Logistic Regression | `C=0.1`, `max_iter=2000` | `class_weight='balanced'` |
| Random Forest | `n_estimators=300`, `max_depth=10`, `min_samples_leaf=10` | `class_weight='balanced'` |
| XGBoost | `n_estimators=300`, `max_depth=6`, `lr=0.05` | `scale_pos_weight=10.4` |
| LightGBM | `n_estimators=300`, `max_depth=6`, `lr=0.05` | `is_unbalance=True` |

Mi proyecto añade **Logistic Regression** como baseline interpretable (Ruru no la tiene). El `scale_pos_weight=10.4` refleja el 8.8% de churn (dataset mucho más desbalanceado).

---

## 6. Resultados finales

Los resultados no son directamente comparables porque el target es diferente (8.8% vs 33.3%), pero los registramos para referencia:

### Ruru — Enfoque 2 (target redefinido, 33.3% churn)

| Modelo | ROC-AUC | F1 (churn) |
|---|---|---|
| Random Forest | 0.8487 | 60.5% |
| **XGBoost** | **0.8527** | 67.4% |
| LightGBM | 0.8525 | 67.3% |

### Mi proyecto — versión `_claude` (target original, 8.8% churn)

Pendiente de ejecutar con las correcciones aplicadas. Referencia anterior (codex, sin `mantenimiento_gratuito`):

| Modelo | TEST ROC-AUC | TEST PR-AUC |
|---|---|---|
| LightGBM | 0.7880 | 0.2282 |
| XGBoost | 0.7849 | 0.2204 |
| Random Forest | 0.7802 | 0.2137 |

---

## 7. Síntesis de diferencias filosóficas

### 7.1 ¿Qué es el churn?

Ruru amplía la definición del problema capturando también a los clientes que nunca fueron al taller. Mi proyecto se ciñe al label original, que solo considera churn a quien fue y dejó de ir. Ambas son decisiones válidas pero modelan comportamientos distintos.

### 7.2 ¿Cómo se construyen las features?

Ruru usa las variables en bruto o con transformaciones simples (binarios, get_dummies). Mi proyecto hace un análisis feature por feature midiendo correlaciones reales, detectando correlaciones espurias y justificando cada decisión (ver `feature_creation.md`). También crea agregados cross-cliente (RFM) que Ruru no tiene.

### 7.3 ¿Cómo se garantiza la no contaminación?

Ruru aplica el preprocesado antes del split y hace el split por fila. Mi proyecto hace el split por cliente antes de cualquier imputación, y encapsula todo dentro de un sklearn Pipeline que se ajusta solo con datos de train.

### 7.4 ¿Cómo se elige el umbral de decisión?

Ruru usa el umbral por defecto (0.5) sin optimizarlo explícitamente. Mi proyecto busca el umbral que maximiza F1 en el conjunto de validación y reporta métricas finales en el test set con ese umbral.

---

## 8. Preguntas abiertas

1. **`TIPO_CARROCERIA` y `TRANSMISION_ID`**: Ruru usa estas features del mismo dataset. Es posible que existan en la tabla `products` pero no se seleccionaron en mi query SQL. Valdría la pena explorar si añadirlas aporta señal adicional.

2. **¿El target redefinido de Ruru es más correcto?** Es una decisión de negocio. Si el objetivo es identificar a todos los clientes "no comprometidos" con el servicio post-venta (incluyendo los que nunca fueron), el target de Ruru tiene sentido. Si el objetivo es predecir quién va a dejar de volver dado que ya empezó a ir, el target original tiene sentido.

3. **¿Por qué el AUC de Ruru (0.85) es mayor que el mío (0.79) con el target original?** Hay tres factores: (a) el target de Ruru es más "fácil" porque tiene señal más clara (33% vs 8.8%), (b) Ruru incluye `en_garantia` en el modelo que en mi caso es constante en scoring, (c) Ruru usa `perfil_cliente` (equivalente a `sin_revisiones`) que en el target original discrimina perfectamente pero en el redefinido sigue siendo informativa sin ser perfecta.
