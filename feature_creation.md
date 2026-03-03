# Feature Engineering — Decisiones por bloque

---

## TABLA: sales — Bloque económico

**Pregunta:** ¿Están correlacionados los atributos económicos? ¿Cuáles mantenemos?

**Atributos analizados:** pvp, coste_venta_no_impuestos, margen_eur_bruto, margen_eur, margen_relativo, descuento_implicito, forma_pago, motivo_venta, origen, fue_lead, lead_compra

**Correlaciones relevantes encontradas:**
- margen_eur vs margen_relativo → r = 0.981 (redundantes, margen_relativo = margen_eur / pvp)
- margen_eur vs margen_eur_bruto → r = 0.718
- margen_eur_bruto vs pvp → r = 0.637
- descuento_implicito vs margen_relativo → r = -0.744
- descuento_implicito vs pvp → r = 0.524

**Churn rates destacados:**
- forma_pago Financiera Marca: 4.7% churn vs Contado: 10.5% — variable muy discriminante
- margen_eur negativo: churn 10.2% vs positivo: 8.3%
- fue_lead, lead_compra, motivo_venta, origen: diferencia de churn rate < 0.3pp — sin poder predictivo

**Decisiones:**

| Feature | Acción | Motivo |
|---|---|---|
| pvp | MANTENER | Mejor correlación con churn (-0.115), proxy del segmento |
| margen_relativo | MANTENER | Absorbe margen_eur y margen_eur_bruto de forma normalizada |
| forma_pago (one-hot) | MANTENER | Muy discriminante (Financiera Marca = mitad de churn) |
| coste_venta_no_impuestos | MANTENER | O flag coste_cero (7.9% zeros, posibles missing) |
| margen_eur | DESCARTAR | r=0.981 con margen_relativo |
| margen_eur_bruto | DESCARTAR | r=0.637 con pvp, r=0.718 con margen_eur |
| descuento_implicito | DESCARTAR | r=-0.744 con margen_relativo, r=0.524 con pvp |
| fue_lead | DESCARTAR | Churn rate idéntico (8.7% vs 8.9%) |
| lead_compra | DESCARTAR | Churn rate idéntico (8.8% vs 8.6%) |
| motivo_venta | DESCARTAR | Churn rate idéntico (8.5% vs 8.8%) |
| origen | DESCARTAR | Churn rate idéntico (8.7% vs 8.9%) |
| **margen_eur_negativo** (nuevo) | **CREAR** | Flag binario: 26.7% ventas con margen negativo → churn 10.2% vs 8.3%. Se añade pero se evaluará su utilidad en el modelo. |

---

## TABLA: sales — Bloque garantía y servicios contratados

**Pregunta:** ¿Qué aportan los atributos de garantía y servicios al modelo?

**Atributos analizados:** extension_garantia, mantenimiento_gratuito, seguro_bateria_largo_plazo, en_garantia, fin_garantia (→ dias_hasta_fin_garantia)

**Hallazgos críticos:**

BUG en dataset previo: tiene_mantenimiento_grat siempre vale 0 porque el código anterior comparaba mantenimiento_gratuito == 'SI', pero los valores reales son 0 y 4 (numéricos). Feature completamente rota en train.csv actual.

dias_hasta_fin_garantia tiene std=0, siempre vale 1461 días (fin_garantia = sales_date + 4 años sin excepción). Varianza cero → inútil.

**Churn rates relevantes:**
- en_garantia=SI: 4.15% churn | en_garantia=NO: 19.26% churn → r=-0.246, la más discriminante del dataset hasta ahora
- mantenimiento_gratuito=4: 0.46% churn | =0: 9.91% churn → protección extrema contra churn
- seguro_bateria=SI: 2.16% churn | =NO: 9.67% churn → muy discriminante
- extension_garantia SI-Financiera: 4.87% churn | NO: 9.78% | SI: 9.27% → solo la variante Financiera discrimina
- seguro_bateria y mantenimiento_gratuito capturan clientes distintos (apenas 9 registros con ambos) → información complementaria

**Decisiones:**

| Feature | Acción | Motivo |
|---|---|---|
| en_garantia | MANTENER | r=-0.246 con churn, la más discriminante del bloque |
| mantenimiento_gratuito | MANTENER (corregir bug) | Churn 0.46% vs 9.91%. Flag: mantenimiento_gratuito != 0 |
| seguro_bateria_largo_plazo | MANTENER | Churn 2.16% vs 9.67%, complementario a mantenimiento |
| ext_garantia_tiene (nuevo) | CREAR | Flag binario: tiene alguna extensión (SI / SI-Financiera / SI-Campaña → 1). Por sí solo no discrimina (9.27% ≈ 9.78%) |
| ext_garantia_financiera (nuevo) | CREAR | Flag binario: la extensión es Financiera → churn 4.87% vs resto. Aquí está todo el poder predictivo. Two flags elegido sobre ordinal porque el salto NO→SI es insignificante (0.5pp) mientras que SI→SI-Financiera es grande (4.4pp); el ordinal transmitiría pesos iguales entre escalones |
| fin_garantia | DESCARTAR | Fecha constante sin variabilidad |
| dias_hasta_fin_garantia | DESCARTAR | std=0, varianza cero, completamente inútil |

---

## TABLA: sales — Bloque post-venta y comportamiento en taller

**Pregunta:** ¿Qué variables de comportamiento post-venta son útiles sin incurrir en leakage? Análisis con perspectiva temporal (base_date = 2023-12-31).

**Atributos analizados:** days_last_service, revisiones, km_medio_por_revision, km_ultima_revision

**Hallazgo crítico — correlación espuria en km_medio_por_revision:**
La correlación global con churn era 0.259, pero era 100% artificial. El grupo sin revisiones tiene km_medio=0 y churn=0 siempre. Al filtrar solo los 30.979 clientes CON revisiones, la correlación cae a -0.004 (cero). Las medias de km entre churn=Y (34.860 km) y churn=N (34.994 km) son prácticamente idénticas. Variable completamente inútil.

**Correlaciones reales (solo clientes con revisiones):**
- km_medio vs churn: -0.004 (nulo — espurio a nivel global)
- km_ultima vs churn: -0.069 (pequeño, dirección contraria a la esperada: churn=N tiene más km totales)
- revisiones vs churn: -0.080 (pequeño pero real)

**Sobre sin_revisiones:**
sin_rev=1 implica churn=0 siempre (27.070 registros, 46.6%). Es la codificación directa de la regla de negocio: necesitas al menos una revisión para ser churn=Y. Útil para particionar el espacio, disponible en new_sales.

**Decisiones:**

| Feature | Acción | Motivo |
|---|---|---|
| days_last_service | EXCLUIR | Leakage directo: define el target al 100% (todos churn=Y tienen >400, sin excepción) |
| sin_revisiones | MANTENER | Codifica la regla de negocio. Particiona el espacio: sin_rev=1 → churn=0 siempre |
| revisiones | MANTENER con cautela | Corr real -0.080 entre quienes tienen revisiones. Disponible en new_sales |
| km_medio_por_revision | EXCLUIR | Correlación global 0.259 era 100% espuria. Corr real = -0.004. No aporta nada |
| km_ultima_revision | EXCLUIR | Corr real pequeña (-0.069), parcialmente redundante con revisiones, patrón contraintuitivo |

---

## TABLA: sales — Bloque engagement

**Pregunta:** ¿Aportan la encuesta de satisfacción y las quejas poder predictivo?

**Atributos analizados:** encuesta_cliente_zona_taller, queja

**Hallazgos:**

encuesta_cliente_zona_taller: corr global con churn = 0.023, corr entre clientes con revisiones = -0.018. Media churn=Y: 68 vs churn=N: 70 (diferencia de 2 puntos en escala 0-276). Además, la encuesta tiene valores para TODOS los registros incluyendo los 27.070 sin revisiones (que nunca fueron al taller), lo que indica que es una métrica agregada de zona, no individual. Si es así, su información ya está capturada por la variable zona.

queja: 57.4% NaN. Churn rates: NO=10.6%, SI=8.2%, Sin_dato=8.0%. Los clientes con queja tienen MENOS churn (efecto contraintuitivo: quien se queja sigue comprometido). Diferencial máximo de 2.4pp. Corr = -0.008.

**Decisiones:**

| Feature | Acción | Motivo |
|---|---|---|
| encuesta_cliente_zona_taller | EXCLUIR | Corr real ≈ 0, métrica de zona no individual, redundante con zona |
| queja | EXCLUIR | Corr = -0.008, 57% NaN, churn rates casi idénticos entre categorías, añadiría ruido |

---

## TABLA: customers

**Pregunta:** ¿Qué variables del perfil demográfico del cliente aportan poder predictivo?

**Atributos analizados:** edad, genero, renta_media_estimada, status_social

**Hallazgos:**

edad: relación monotónicamente decreciente con el churn (<35: 10.8%, 35-50: 9.2%, 50-65: 6.2%, >65: 1.9%). Sin NaN. Corr=-0.079. Relación suave y continua, no hace falta bucketizar.

genero: M=8.3% churn, F=9.7%, Desconocido=11.7%. Diferencia real aunque pequeña (1.4pp M vs F).

renta_media_estimada: 22.5% son zeros que encubren missing values. El poder predictivo está en el FLAG: renta=0 tiene churn 5.9% vs renta>0 tiene 9.6%. La correlación real del valor (excluyendo zeros) es -0.029, prácticamente nula. Los clientes con renta desconocida probablemente son empresas/flotas con patrones de mantenimiento distintos. Mismo patrón "Sin_dato = menor churn" que en status_social.

status_social: 22.1% NaN. Spread de 3.9pp entre categorías (U: 12.6% vs A: 8.7%). Sin_dato tiene el churn más bajo (5.95%), coherente con el patrón de renta=0. Sin orden socioeconómico claro entre letras → target encoding es lo más apropiado.

**Decisiones:**

| Feature | Acción | Motivo |
|---|---|---|
| edad | MANTENER numérico | Relación monotónica limpia, r=-0.079, sin NaN |
| genero | MANTENER (one-hot) | Diferencia M/F real, Desconocido como categoría propia |
| renta_media_estimada | MANTENER valor + flag renta_desconocida | Corr real del valor ≈ 0, pero el flag captura segmento con churn 5.9% |
| status_social | MANTENER (target encoding) | Spread 3.9pp entre categorías, sin orden implícito claro |

---

## TABLA: products

**Pregunta:** ¿Qué variables del catálogo de productos aportan poder predictivo? ¿Hay colinealidad?

**Atributos analizados:** modelo, fuel, equipamiento, kw

**Hallazgos:**

modelo: spread brutal 0.03% (H) a 24% (G). La variable más discriminante del dataset. H tiene solo 1 caso de churn en 2907 registros — prácticamente inmune al churn. G y F tienen n=25 y n=29 → target encoding será muy ruidoso, necesita smoothing (e.g. Laplace o media ponderada con la media global).

fuel: ELÉCTRICO=2.9% churn vs HÍBRIDO=9.3%. Señal grande pero parcialmente redundante con modelo: K es 100% eléctrico, F/G/J son 100% híbridos. Para modelos mixtos (A,B,D,H,I) sí aporta información adicional. Mantener como flag binario es_electrico.

equipamiento: Low=17%, Mid=7%, Mid-High=8.6%, High=3.2%. Corr=-0.115. Anomalía: Mid<Mid-High (7% vs 8.6%). El encoding ordinal asigna pesos iguales entre escalones, pero la anomalía es menor y para árboles funciona bien igualmente. Corr con kw=0.162 → no hay colinealidad real.

kw: rango 48-193kw, media 93.9. Corr=-0.098. <80kw=13.8%, 80-120kw=6.9%, >120kw=5.8%. Corr con equipamiento_ord=0.162 (baja, capturan dimensiones distintas: potencia vs nivel de acabado). Sin NaN.

**Decisiones:**

| Feature | Acción | Motivo |
|---|---|---|
| modelo | MANTENER (target encoding con smoothing) | Spread 0.03%-24%, la más discriminante. G y F con n<30 necesitan smoothing |
| es_electrico | MANTENER (flag binario) | Churn 2.9% vs 9.3%. Aporta info adicional en modelos mixtos |
| equipamiento | MANTENER (ordinal) | Spread 3.2%-17%, patrón mayormente decreciente. Anomalía Mid<Mid-High es menor |
| kw | MANTENER (numérico) | Corr -0.098, corr con equipamiento=0.162 (no colineal) |
| fuel (original) | DESCARTAR | Sustituida por es_electrico |

---

## TABLA: stores

**Pregunta:** ¿Aportan zona y tienda poder predictivo?

**Atributos analizados:** zona, tienda_desc

**Hallazgos:**
zona: ESTE=9.04%, NORTE=8.90%, CENTRO=8.63%, SUR=8.56%. Spread de 0.48pp. Con n>5.000 por zona el error estándar de la proporción es ~0.4% → las diferencias están dentro del ruido estadístico. No es un patrón real.

tienda_desc: Spread de 1.36pp entre MALAGA (7.88%) y SEVILLA (9.24%). Ligeramente mayor variabilidad pero sigue siendo mínima. Dentro de cada zona las tiendas son casi idénticas.

**Decisiones:**

| Feature | Acción | Motivo |
|---|---|---|
| zona | DESCARTAR | Spread 0.48pp, dentro del ruido estadístico |
| tienda_desc | DESCARTAR | Spread 1.36pp mínimo, capturado por mix de producto y perfil de cliente |

---

## TABLA: model_costs

**Pregunta:** ¿Aportan los costes por modelo información adicional?

**Atributos analizados:** margen, coste_transporte, margen_distribuidor, gastos_marketing, mantenimiento_medio, comision_marca

**Hallazgos:**
coste_transporte y margen_distribuidor son constantes en todos los modelos (250 y 5 respectivamente). Varianza cero, r=NaN.

El resto (margen, gastos_marketing, mantenimiento_medio, comision_marca) son funciones deterministas de modelo — la tabla tiene una fila por modelo. Si modelo ya es una feature, todas estas variables son completamente redundantes: no añaden ninguna información que el modelo no pueda inferir de modelo directamente.

**Decisiones:**

| Feature | Acción | Motivo |
|---|---|---|
| coste_transporte | DESCARTAR | Constante en todos los modelos, varianza cero |
| margen_distribuidor | DESCARTAR | Constante en todos los modelos, varianza cero |
| gastos_marketing | DESCARTAR | Función determinista de modelo → redundante |
| mantenimiento_medio | DESCARTAR | Función determinista de modelo → redundante |
| margen (costes) | DESCARTAR | Función determinista de modelo → redundante |
| comision_marca | DESCARTAR | Función determinista de modelo → redundante |

**Evidencia cuantitativa (Logistic Regression como test rápido):**
- AUC solo costes: 0.6315
- AUC solo modelo label: 0.6395
- AUC costes + modelo: 0.6402 → los costes añaden 0.0007 de AUC sobre modelo. Ruido absoluto.

**Caso concluyente F/G/H:** tres modelos con perfil de costes casi idéntico tienen churn rates de 17.2%, 24.0% y 0.03% respectivamente. Los costes no explican la variación de churn entre modelos. El label modelo sí.

---

**Nota sobre encoding:** el encoding (target encoding, one-hot, ordinal, etc.) se decide en el pipeline de modelado, no en el feature engineering. El feature engineering entrega las variables categóricas como strings. Cambiar de modelo no rompe el pipeline de features.

---

## RESUMEN FINAL — 22 features seleccionadas

### sales (13 features)

| Feature | Tipo | Nota |
|---|---|---|
| pvp | Numérico | |
| margen_relativo | Numérico derivado | margen_eur / pvp |
| margen_eur_negativo | Flag binario (NUEVO) | churn 10.2% vs 8.3% |
| coste_venta_no_impuestos | Numérico | 7.9% zeros posibles missing |
| forma_pago | Categórica | Encoding en pipeline |
| dias_desde_compra | Numérico derivado | r=0.287 con churn. Sustituye año_venta y mes_venta (r=-0.986 entre sí) |
| en_garantia | Binario | r=-0.246 con churn. La más discriminante del dataset |
| mantenimiento_gratuito | Flag binario (BUG FIX) | Comparar != 0, no == 'SI'. Churn 0.46% vs 9.91% |
| seguro_bateria_largo_plazo | Binario | Churn 2.16% vs 9.67% |
| ext_garantia_tiene | Flag binario (NUEVO) | Tiene alguna extensión de garantía |
| ext_garantia_financiera | Flag binario (NUEVO) | Extensión es Financiera → churn 4.87% |
| sin_revisiones | Flag binario | Codifica la regla de negocio: sin revisiones → churn=0 siempre |
| revisiones | Numérico | Corr real -0.080 entre quienes tienen revisiones |

### customers (5 features)

| Feature | Tipo | Nota |
|---|---|---|
| edad | Numérico | Relación monotónica: >65 años churn 1.9% |
| genero | Categórica | Encoding en pipeline |
| renta_media_estimada | Numérico | Corr real ≈ 0, útil junto al flag |
| renta_desconocida | Flag binario | renta=0 → churn 5.9% vs 9.6% |
| status_social | Categórica | Encoding en pipeline. Spread 3.9pp entre categorías |

### products (4 features)

| Feature | Tipo | Nota |
|---|---|---|
| modelo | Categórica | Encoding en pipeline. Spread 0.03%-24% — la más discriminante |
| equipamiento | Categórica con orden (Low < Mid < Mid-High < High) | Encoding en pipeline |
| es_electrico | Flag binario (NUEVO) | Churn 2.9% vs 9.3% |
| kw | Numérico | r=-0.098, complementario a equipamiento |

### Features descartadas — motivos

| Motivo | Features |
|---|---|
| Leakage directo | days_last_service |
| Correlación espuria | km_medio_por_revision (corr global 0.259 → real -0.004) |
| Redundantes por margen_relativo/pvp | margen_eur, margen_eur_bruto, descuento_implicito |
| Redundantes por dias_desde_compra | año_venta, mes_venta |
| Redundantes por modelo (AUC +0.0007) | todas las de model_costs |
| Sin señal (<1.4pp spread) | zona, tienda_desc |
| Sin señal (<0.3pp churn diff) | fue_lead, lead_compra, motivo_venta, origen |
| Sin señal (corr ≈ 0) | encuesta_cliente_zona_taller, queja |
| Varianza cero | coste_transporte, margen_distribuidor, fin_garantia, dias_hasta_fin_garantia |
| Identificadores técnicos | code, customer_id, sales_date, base_date, id_producto, tienda_desc |
| Sustituidas por flags | fuel (→ es_electrico), extension_garantia (→ dos flags) |
| Corr real pequeña y contraintuitiva | km_ultima_revision |

**Nota sobre modelo label vs características:** ambos se mantienen porque capturan dimensiones distintas. El label captura efectos latentes a nivel modelo (perfil del comprador, red de servicio, campañas). Las características (kw, equipamiento, es_electrico) capturan variabilidad dentro del mismo modelo — hay 504 productos en 11 modelos (~46 configuraciones por modelo), por lo que dentro de un mismo modelo puede haber versiones muy distintas.
