# Data Understanding — Churn Prediction

## Contexto del problema

El objetivo es construir un modelo de clasificación binaria que prediga si un cliente es **churn** (`churn_400 = Y`) o no (`churn_400 = N`) tras una venta de vehículo.

El dataset principal (`sales`) tiene **58.049 registros** con **44.053 clientes únicos**, lo que implica que hay clientes con múltiples compras. La tasa de churn global es de aproximadamente **8.8 %** (5.093 Y vs 52.956 N) — dataset desequilibrado.

---

## Esquema de tablas

| Tabla | Filas | Rol |
|---|---|---|
| `sales` | 58.049 | Tabla principal de transacciones (train) |
| `new_sales` | 10.000 | Conjunto de validación |
| `customers` | 52.553 | Perfil demográfico del cliente |
| `products` | 504 | Catálogo de productos/vehículos |
| `stores` | 12 | Información de tiendas |
| `model_costs` | 11 | Costes por modelo de vehículo |

---

## Grupos de columnas

### Grupo 1 — Variable Objetivo

| Columna | Tabla | Tipo | Descripción |
|---|---|---|---|
| `churn_400` | sales | Categórica (Y/N) | Indica si el cliente lleva más de 400 días sin revisión en el taller. Es el **target** del modelo. |
| `base_date` | sales | Fecha | Fecha de corte del snapshot desde la que se miden los 400 días. |

**Distribución:** ~91 % N / 9 % Y (5.093 churn vs 52.956 no-churn). Dataset desequilibrado — usar métricas adecuadas (AUC-ROC, F1, Precision-Recall) y técnicas de balanceo.

**Análisis de consistencia del label (verificado en data_preparation):**

La regla del enunciado (`days_last_service > 400 → churn = Y`) no es perfectamente simétrica. La tabla de contingencia real es:

| Categoría | churn=N | churn=Y |
|---|---|---|
| Con servicio, ≤400 días | 24.548 | 0 |
| Con servicio, >400 días | 1.338 | 5.093 |
| Sin servicio (NULL), ≤400 días desde compra | 12.834 | 0 |
| Sin servicio (NULL), >400 días desde compra | 14.236 | 0 |

Dos anomalías relevantes:

**Anomalía 1 — 14.236 clientes nunca fueron al taller, llevan >400 días (media 665 días) y son `churn = N`.**
La regla del enunciado los clasificaría como churn, pero están etiquetados como no-churn. El label `churn_400` parece **excluir a clientes que nunca han tenido ninguna revisión**, posiblemente porque requieren al menos una visita al taller para entrar en el ciclo de fidelización medible. También puede estar relacionado con la columna `mantenimiento_gratuito`.

**Anomalía 2 — 1.338 clientes con `days_last_service > 400` son `churn = N`.**
Probablemente son clientes **multi-compra**: el `days_last_service` corresponde al servicio de un vehículo anterior, pero la venta en cuestión es reciente. La nueva compra "reinicia el reloj" y el cliente no se considera perdido.

**Conclusión sobre la regla real observada:**
`churn_400 = Y` ↔ el cliente tiene al menos una revisión registrada Y `days_last_service > 400`.

---

### Grupo 2 — Identificadores y claves técnicas

| Columna | Tabla | Tipo | Descripción |
|---|---|---|---|
| `code` | sales | String | Identificador único de la transacción (ej. `ID22371`). No aporta información predictiva directa. |
| `customer_id` | sales / customers | Integer | Clave del cliente. Permite join con `customers` y detectar clientes con múltiples compras. |
| `id_producto` | sales / products | String | Clave del producto. Permite join con `products`. |
| `tienda_desc` | sales / stores | String | Nombre de la tienda. Permite join con `stores`. |

**Nota:** Estos campos no se usarán como features directamente, pero son esenciales para construir agregaciones y joins.

---

### Grupo 3 — Perfil demográfico del cliente

> Proviene de la tabla `customers`. Describe quién es el cliente.

| Columna | Tabla | Tipo | Descripción |
|---|---|---|---|
| `edad` | customers | Numérica (entero) | Edad del cliente en años. Rango 20–78, media ~43. |
| `genero` | customers | Categórica (M/F/NaN) | Género del cliente. ~68 % hombres, ~30 % mujeres, ~1.5 % sin dato. |
| `renta_media_estimada` | customers | Numérica (float) | Estimación de la renta anual del cliente. Media ~22.000 €, con valores 0 que pueden ser nulos enmascarados. |
| `status_social` | customers | Categórica (A–K/NaN) | Segmentación socioeconómica del cliente. ~22 % sin dato. Distribución relativamente uniforme entre categorías. |

**Ideas de features:**
- Buckets de edad (joven < 35, adulto 35–55, senior > 55).
- Flag de `renta_media_estimada == 0` como posible dato faltante.
- Encoding ordinal de `status_social` si hay orden implícito, o one-hot si no.

---

### Grupo 4 — Información económica de la venta

> Describe el precio, el margen y la forma de financiación de la transacción.

| Columna | Tabla | Tipo | Descripción |
|---|---|---|---|
| `pvp` | sales | Numérica (float) | Precio de venta al público del vehículo. Rango 10.528–37.970 €, media ~23.563 €. |
| `coste_venta_no_impuestos` | sales | Numérica (float) | Coste asociado a la venta sin impuestos. Muchos valores a 0 (pueden indicar venta sin costes adicionales). |
| `margen_eur_bruto` | sales | Numérica (float) | Margen bruto de la venta en euros. Siempre positivo. |
| `margen_eur` | sales | Numérica (float) | Margen neto de la venta en euros. Puede ser negativo. Media ~1.630 €, con amplia varianza. |
| `forma_pago` | sales | Categórica | Método de pago: Contado, Financiera Marca, Otros, Préstamo Bancario. |
| `motivo_venta` | sales | Categórica | Razón de la venta: Particular / No Particular. |
| `fue_lead` | sales | Binaria (0/1) | Si el cliente llegó a través de un lead comercial. |
| `lead_compra` | sales | Numérica/Binaria | Indicador del canal de captación. |
| `sales_date` | sales | Fecha | Fecha de la transacción. Permite extraer temporalidad. |

**Ideas de features:**
- `margen_relativo = margen_eur / pvp` (rentabilidad normalizada).
- `descuento_implicito = margen_eur_bruto - margen_eur` (coste de descuentos y financiación).
- Encoding de `forma_pago` (binarias por categoría).
- Año y mes de `sales_date`; antigüedad de la compra respecto a hoy.
- Flag de compra con financiación (Financiera Marca o Préstamo Bancario).

---

### Grupo 5 — Características del producto

> Proviene de la tabla `products`. Describe qué vehículo compró el cliente.

| Columna | Tabla | Tipo | Descripción |
|---|---|---|---|
| `modelo` | products | Categórica (A–K) | Modelo del vehículo. Alta variabilidad en churn rate: desde 0 % (modelo H) hasta 24 % (modelo G). |
| `fuel` | products | Categórica | Tipo de combustible: HÍBRIDO (~91 %) o ELÉCTRICO (~9 %). |
| `equipamiento` | products | Categórica ordinal | Nivel de equipamiento: Low < Mid < Mid-High < High. |
| `kw` | products | Numérica (float) | Potencia del vehículo en kilovatios. Rango 48–193 kW, media ~94 kW. |

**Ideas de features:**
- Encoding ordinal de `equipamiento`.
- `es_electrico` flag binario.
- `churn_rate_modelo` como feature de target encoding (con cuidado de data leakage).
- Buckets de potencia (bajo < 80 kW, medio, alto > 120 kW).

---

### Grupo 6 — Geografía y tienda

> Proviene de la tabla `stores`. Describe dónde se realizó la venta.

| Columna | Tabla | Tipo | Descripción |
|---|---|---|---|
| `tienda_desc` | stores | Categórica | Nombre específico de la tienda (12 tiendas distintas). |
| `zona` | stores | Categórica (CENTRO, ESTE, NORTE, SUR) | Zona geográfica de la tienda. Churn rate ligeramente mayor en ESTE (9 %) y NORTE (8.9 %). |

**Ideas de features:**
- One-hot encoding de `zona`.
- `churn_rate_tienda` como target encoding.
- Flag de zona con mayor riesgo (ESTE / NORTE).

---

### Grupo 7 — Servicio post-venta

> Indica el comportamiento del cliente en el taller después de la compra.

> ⚠️ **RIESGO DE DATA LEAKAGE — leer antes de usar cualquier variable de este grupo.**

| Columna | Tabla | Tipo | Descripción |
|---|---|---|---|
| `days_last_service` | sales | Numérica (int) | Días desde el último servicio hasta `base_date`. ~47 % de NaN (clientes sin revisiones). Media 625 días en churn vs 183 en no-churn. |
| `revisiones` | sales | Numérica (entero) | Número total de revisiones realizadas. Media 1.6 en churn vs 0.9 en no-churn. |
| `km_medio_por_revision` | sales | Numérica (float) | Kilómetros medios entre revisiones. Media 34.860 km en churn vs 17.106 km en no-churn. |
| `km_ultima_revision` | sales | Numérica (float) | Kilómetros acumulados en la última revisión. |
| `en_garantia` | sales | Categórica (SI/NO) | Si el vehículo está en garantía en el momento del snapshot. |
| `extension_garantia` | sales | Categórica | Tipo de extensión contratada: NO, SI, SI-Financiera, SI-Campaña Regalo. |
| `mantenimiento_gratuito` | sales | Categórica | Si la venta incluye mantenimiento gratuito. Puede explicar por qué clientes sin revisiones son etiquetados como no-churn. |
| `fin_garantia` | sales | Fecha | Fecha de fin de garantía del vehículo. |

**Riesgo de data leakage:**

`days_last_service` es la variable que define directamente el target: todos los `churn=Y` tienen `days_last_service > 400` sin excepción. Incluirla como feature haría que el modelo aprendiera la etiqueta en lugar del comportamiento que la precede. **Debe excluirse del modelo.**

`revisiones`, `km_medio_por_revision` y `km_ultima_revision` también son métricas medidas en el momento del snapshot (`base_date`), por lo que incorporan información posterior a la venta que podría no estar disponible en producción. Usarlas requiere confirmar que estarán disponibles en el momento de scoring.

**Ideas de features (con cautela):**
- `sin_revisiones` flag binario (`revisiones == 0`) — señala clientes que nunca vinieron al taller.
- `tiene_extension_garantia` flag binario.
- `dias_hasta_fin_garantia = fin_garantia - sales_date` — proxy del compromiso de servicio contratado.
- `mantenimiento_gratuito` directamente como feature categórica.

---

### Grupo 8 — Engagement y satisfacción

> Mide la relación del cliente con la marca a través de encuestas y reclamaciones.

| Columna | Tabla | Tipo | Descripción |
|---|---|---|---|
| `encuesta_cliente_zona_taller` | sales | Numérica (float) | Puntuación de la encuesta del taller. Rango 0–276 (distribución anómala, posiblemente escala acumulada o distinta por zona). |
| `queja` | sales | Categórica (SI/NO/NaN) | Si el cliente presentó una queja formal. ~53 % de NaN. Sorprendentemente, clientes con queja tienen churn similar a los que no la tienen — puede indicar que quejas gestionadas bien fidelizan. |

**Ideas de features:**
- Flag de `queja == SI`.
- Imputar NaN de `queja` como categoría propia ("Sin_dato") — puede tener significado (clientes que no interactuaron).
- Normalizar `encuesta_cliente_zona_taller` o usar buckets (baja < 30, media 30–100, alta > 100).
- Interacción `queja * encuesta` para capturar insatisfacción combinada.

---

## Observaciones generales para feature engineering

### Clientes con múltiples compras
Hay **44.053 clientes únicos** en 58.049 ventas. Algunos clientes tienen compras con distinto valor de `churn_400` (ej. cliente 6840 tiene una venta Y y otra N). Esto implica que el modelo trabaja a **nivel de transacción**, no de cliente. Se pueden construir features que agreguen el **historial previo del cliente** (número de compras anteriores, si tuvo churn previo, etc.).

### Variables con alta proporción de NaN
| Columna | % NaN aprox. | Estrategia sugerida |
|---|---|---|
| `days_last_service` | ~47 % | **EXCLUIR del modelo** (data leakage). Si se usa, crear solo flag `sin_revisiones`. |
| `queja` | ~53 % | Tratar NaN como categoría propia ("Sin_dato") |
| `genero` | ~1.5 % | Imputar por moda o categoría "Desconocido" |
| `renta_media_estimada = 0` | variable | Flag de renta desconocida |
| `status_social` | ~22 % | Categoría "Sin dato" |

### Variables con data leakage confirmado — excluir del modelo
Estas variables se miden en el mismo snapshot (`base_date`) que el target. Incluirlas filtraría información del futuro al modelo:

| Variable | Motivo de exclusión |
|---|---|
| `days_last_service` | Define directamente el target: todos los churn=Y tienen >400, sin excepción |
| `km_medio_por_revision` | Métrica de comportamiento post-venta medida en el snapshot |
| `km_ultima_revision` | Ídem |
| `revisiones` | Conteo acumulado hasta el snapshot; correlacionado con el target |

### Variables más discriminantes para el modelo (sin leakage)
1. `modelo` — modelo H tiene 0 % churn; modelos C, F, G superan el 14 %.
2. `zona` — diferencias sistemáticas (ESTE 9 %, NORTE 8.9 %, CENTRO y SUR 8.6 %).
3. `forma_pago` — proxy de compromiso financiero con la marca.
4. `extension_garantia` — señala mayor vinculación con el servicio oficial.
5. `mantenimiento_gratuito` — puede explicar el comportamiento de clientes "sin revisiones, no-churn".
6. `pvp` / `margen_eur` — precio y rentabilidad como proxies del segmento de cliente.
7. `edad`, `renta_media_estimada` — perfil socioeconómico del comprador.
8. `sin_revisiones` (flag derivado) — clientes que nunca volvieron al taller.

### Variables posiblemente irrelevantes o ruidosas
- `lead_compra` — muchos valores a 0, poca varianza.
- `tienda_desc` — 12 categorías; mejor usar `zona` o target encoding.
- `code` — puramente técnico, descartar.
- `base_date` — fecha de corte del snapshot, no informativa para el modelo.

---

## Próximos pasos

1. **Construcción del dataset analítico:** JOIN de `sales` + `customers` + `products` + `stores`, trabajando sobre el schema `barrechee`.
2. **Excluir variables con leakage:** `days_last_service`, `km_medio_por_revision`, `km_ultima_revision`, `revisiones` (o usarlas solo como flag binario `sin_revisiones`).
3. **Tratamiento de NaN** según estrategias del cuadro anterior.
4. **Encoding** de variables categóricas (ordinal para `equipamiento`, one-hot para `zona`, target encoding para `modelo` y `tienda_desc`).
5. **Feature engineering** de temporalidad (`sales_date`), historial de cliente (multi-compra) y ratios económicos (`margen_relativo`, `descuento_implicito`).
6. **Análisis de correlación y VIF** para descartar multicolinealidad (e.g., `margen_eur_bruto` vs `pvp`).
7. **Baseline model** con Logistic Regression o Decision Tree para benchmarking.
8. **Modelo final:** Gradient Boosting (XGBoost / LightGBM) dado el desbalanceo y la mezcla de tipos de variable.