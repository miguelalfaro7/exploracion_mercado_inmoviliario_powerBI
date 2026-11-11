# Informe técnico — Exploración del mercado inmobiliario

## Fuente y unidad de análisis

Fuente oficial: `CSV procesado de viviendas.csv`. Se procesaron 21,613 ventas y 21,436 propiedades identificadas por `id`. El cargador conserva todas las filas y columnas, no elimina ventas por ID y convierte `date` a fecha y `zipcode` a categoría. Los requisitos 6–10 describen ventas; el ANOVA usa aparte una tabla con la última venta de cada propiedad.

## Requisito 6 — Precio según año de construcción

Se agruparon las ventas por `yr_built` y se exportaron media, mediana y cantidad. Los tres años con mayor media observada son:

- 1933: media US$772,050, mediana US$402,500, n=30 ventas.
- 2015: media US$759,785, mediana US$629,500, n=38 ventas.
- 1905: media US$752,978, mediana US$597,500, n=74 ventas.

La gráfica presenta medias y medianas en USD. Las medias fluctúan entre años y algunas cohortes tienen pocos registros; por ello los máximos no deben interpretarse como un efecto causal de la antigüedad. La distribución de ventas por año, ubicación y características también influye.

## Requisito 7 — Modelo predictivo

Se predijo `price` con RandomForestRegressor y un baseline de media de entrenamiento. La separación se hizo con GroupShuffleSplit usando `id` como grupo: 17,295 ventas de entrenamiento, 4,318 de prueba y 0 IDs compartidos. El preprocesamiento se ajusta dentro del Pipeline. `zipcode` se codificó como categoría.

Variables usadas: `bedrooms, bathrooms, sqft_living, sqft_lot, floors, waterfront, view, condition, grade, yr_built, yr_renovated, lat, long, zipcode`. Se excluyeron `id`, `registro_id`, `precio_por_m2` (derivada del precio), `superficie_m2` (redundante con `sqft_living`), la fecha y variables de área redundantes o de contexto (`sqft_above`, `sqft_basement`, `sqft_living15`, `sqft_lot15`).

| Modelo | Partición | MAE (USD) | RMSE (USD) | R² |
|---|---|---:|---:|---:|
| Random Forest | Entrenamiento | US$31,034 | US$60,476 | 0.9729 |
| Random Forest | Prueba | US$67,745 | US$127,797 | 0.8779 |
| Baseline de media | Prueba | US$230,957 | US$365,776 | -0.0001 |

El bosque supera al baseline en MAE, RMSE y R² de prueba. La brecha entre entrenamiento y prueba es evidencia de ajuste optimista en los datos de entrenamiento; la métrica de prueba agrupada es la estimación principal. La evaluación es aleatoria por propiedad, no una validación temporal para años futuros.

## Requisito 8 — Superficie y precio

- Pearson: r=0.702, p=<1e-300 (subdesbordamiento numérico).
- Spearman: r=0.644, p=<1e-300 (subdesbordamiento numérico).

Ambos coeficientes indican asociación positiva entre superficie habitable y precio. Los p-valores numéricamente menores al límite de representación se muestran como `<1e-300`; correlación no implica causalidad.

## Requisito 9 — Propiedades frente al agua

- No: n=21450, media US$531,564, mediana US$450,000.
- Sí: n=163, media US$1,661,876, mediana US$1,400,000.

`waterfront` identifica una propiedad frente al agua; no confirma necesariamente vista al mar. Las diferencias son descriptivas y no atribuyen causalidad a esta variable.

## Requisito 10 — Segmentos de precio

Los cuatro segmentos se definieron con cuartiles de las 21.613 ventas, no con límites oficiales del mercado:

- Económico: n=5404, media US$251,558, mediana US$259,000, rango US$75,000–US$321,950.
- Medio: n=5460, media US$385,689, mediana US$385,000, rango US$322,000–US$450,000.
- Alto: n=5376, media US$539,731, mediana US$537,000, rango US$450,500–US$645,000.
- Premium: n=5373, media US$987,540, mediana US$826,600, rango US$645,500–US$7,700,000.

## ANOVA por código postal

Se creó `propiedades_ultima_venta.csv` seleccionando, por `id`, la venta de fecha más reciente. Contiene 21,436 propiedades y 70 ZIP. H₀: el precio medio de la última venta es igual en todos los ZIP. H₁: al menos un ZIP tiene una media distinta.

Se aplicó Levene con centro mediano: estadístico 50.737, p=<1e-300 (subdesbordamiento numérico). Se seleccionó **ANOVA de Welch**, F=250.420, p=<1e-300 (subdesbordamiento numérico), α=0,05. Resultado: se rechaza H₀; existe evidencia de que al menos una media difiere. Post hoc: Games–Howell; 1,922 pares significativos de 2,415 evaluados.

Shapiro aplicado a una muestra reproducible de 5,000 residuos dio W=0.699, p=1.468e-69; la gráfica Q–Q muestra desviaciones de normalidad. Una propiedad por ID elimina la repetición de ventas dentro de la tabla del ANOVA, pero no elimina una posible dependencia espacial entre viviendas cercanas. ZIP es una aproximación geográfica, no una división oficial por barrio. Welch y Games–Howell no prueban causalidad.

## Archivos de salida

Las tablas y gráficas están en `salidas/`. El modelo serializado y el manifiesto de variables están en `salidas/modelos/` y `salidas/validacion_modelo.json`. El CSV recibido permanece intacto; las conversiones de tipos y columnas auxiliares se hacen en memoria.
