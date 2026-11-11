# Componente estadístico — Exploración del Mercado Inmobiliario

Este componente corresponde al Integrante 3. Lee como fuente oficial el archivo procesado por el Integrante 1, conserva su granularidad de venta y genera los análisis R6–R10, el modelo predictivo y la prueba ANOVA. No vuelve a ejecutar la limpieza ni elimina filas por `id`.

## Fuente oficial

El archivo de trabajo es `datos/CSV procesado de viviendas.csv`, copiado sin cambios desde el CSV recibido. Su SHA-256 se verificó al copiarlo. El análisis utiliza exclusivamente esta fuente; `kc_house_data.csv` y `datos/kc_house_clean.csv` son archivos anteriores y no alimentan los resultados nuevos.

El cargador preserva las columnas oficiales y transforma solo los tipos en memoria: `date` a fecha, `id` a etiqueta de texto y `zipcode` a categoría textual. Valida `registro_id`, que identifica cada venta, y no descarta ventas repetidas de una propiedad.

## Instalación y ejecución

Requiere Python 3.10 o posterior.

```powershell
py -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
python -m src.ejecutar_todo
```

El pipeline predetermina la fuente oficial dentro de `datos/`. Si se almacena en otra ubicación, configure la variable antes de ejecutar:

```powershell
$env:KC_HOUSE_SOURCE = 'C:\ruta\CSV procesado de viviendas.csv'
python -m src.ejecutar_todo
```

## Unidades de análisis

- **R6, R7, R8, R9 y R10:** ventas; se conservan las 21.613 filas.
- **R7:** `GroupShuffleSplit` agrupa por `id`; una propiedad no puede aparecer simultáneamente en entrenamiento y prueba.
- **ANOVA:** `salidas/propiedades_ultima_venta.csv` contiene una venta por propiedad: la de fecha más reciente. Se esperan 21.436 propiedades únicas. Welch se usa si Levene detecta varianzas desiguales; Games–Howell se aplica si el contraste resulta significativo y las varianzas son desiguales. La dependencia espacial entre viviendas cercanas sigue siendo una limitación.

## Modelo predictivo

El objetivo es `price`. Se excluyen `id`, `registro_id`, `precio_por_m2` (fuga directa del objetivo), `superficie_m2` (derivada de `sqft_living`), `date` y variables de superficie redundantes o de contexto. `zipcode` es categórica; `lat` y `long` se conservan. Se comparan Random Forest y un baseline de media con MAE, RMSE y R², tanto en entrenamiento como en prueba.

Para inferencia, el modelo serializado está en `salidas/modelos/random_forest_house_price.joblib`:

```powershell
python -m src.prediccion --bedrooms 3 --bathrooms 2 --sqft_living 1800 --sqft_lot 5000 --floors 1 --waterfront 0 --view 0 --condition 3 --grade 7 --yr_built 1990 --yr_renovated 0 --zipcode 98103 --lat 47.65 --long -122.34
```

## Resultados

- `salidas/precio_por_anio_construccion.csv`: media, mediana y número de ventas por año construido.
- `salidas/metricas_modelo.csv`: métricas de entrenamiento y prueba para Random Forest y baseline.
- `salidas/predicciones_test.csv`, `salidas/particion_modelo.csv` y `salidas/validacion_modelo.json`: predicciones y control de IDs compartidos.
- `salidas/correlacion_superficie_precio.csv`, `salidas/analisis_frente_agua.csv`, `salidas/ventas_segmentadas.csv` y `salidas/resumen_segmentos_precio.csv`.
- `salidas/anova_zipcode.csv`, `salidas/posthoc_games_howell_zipcode.csv` y `salidas/resumen_anova_zipcode.csv`.
- `salidas/graficas/`: gráficos PNG; `informes/informe_resultados.md`: interpretación de resultados reales.
- `salidas/diagnostico_fuente.json`: filas, columnas, nulos y duplicados observados en la fuente.

Las tablas CSV están codificadas en UTF-8 para importación en Power BI. Las medidas DAX del Integrante 2 no forman parte de este componente.

## Respaldo

Antes de adaptar el proyecto se guardaron los resultados anteriores, el informe, el README y el antiguo CSV procesado en `respaldo_resultados_20261111_150033/`. El CSV oficial recibido permanece intacto.
