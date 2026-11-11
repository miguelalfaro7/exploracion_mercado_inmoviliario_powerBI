"""Ejecuta el componente estadístico sobre la fuente oficial del Integrante 1."""

import pandas as pd

from .analisis import run_analyses
from .configuracion import OUTPUT, REPORTS, SOURCE, prepare_dirs
from .etl import load_official_data


def _usd(value):
    return f"US${value:,.0f}"


def _write_report(results):
    metrics = results["model"]["metrics"]
    rf_train = metrics.query("modelo == 'RandomForestRegressor' and particion == 'train'").iloc[0]
    rf_test = metrics.query("modelo == 'RandomForestRegressor' and particion == 'test'").iloc[0]
    base_test = metrics.query("modelo == 'BaselineMedia' and particion == 'test'").iloc[0]
    r6_top = results["r6"].nlargest(3, "precio_medio_usd")
    r6_lines = "\n".join(
        f"- {int(row.yr_built)}: media {_usd(row.precio_medio_usd)}, "
        f"mediana {_usd(row.precio_mediano_usd)}, n={int(row.cantidad_ventas)} ventas."
        for row in r6_top.itertuples()
    )
    corr_lines = "\n".join(
        f"- {row.metodo}: r={row.coeficiente:.3f}, p={row.p_presentacion}."
        for row in results["correlation"].itertuples()
    )
    water_lines = "\n".join(
        f"- {row.waterfront_label}: n={int(row.cantidad_ventas)}, "
        f"media {_usd(row.precio_medio_usd)}, mediana {_usd(row.precio_mediano_usd)}."
        for row in results["waterfront"].itertuples()
    )
    segment_lines = "\n".join(
        f"- {row.price_segment}: n={int(row.cantidad_ventas)}, "
        f"media {_usd(row.precio_medio_usd)}, mediana {_usd(row.precio_mediano_usd)}, "
        f"rango {_usd(row.precio_minimo_usd)}–{_usd(row.precio_maximo_usd)}."
        for row in results["segments"].itertuples()
    )
    anova = results["anova"]
    split = results["model"]["split"]
    report = f"""# Informe técnico — Exploración del mercado inmobiliario

## Fuente y unidad de análisis

Fuente oficial: `{SOURCE.name}`. Se procesaron {results['source_rows']:,} ventas y {results['unique_properties']:,} propiedades identificadas por `id`. El cargador conserva todas las filas y columnas, no elimina ventas por ID y convierte `date` a fecha y `zipcode` a categoría. Los requisitos 6–10 describen ventas; el ANOVA usa aparte una tabla con la última venta de cada propiedad.

## Requisito 6 — Precio según año de construcción

Se agruparon las ventas por `yr_built` y se exportaron media, mediana y cantidad. Los tres años con mayor media observada son:

{r6_lines}

La gráfica presenta medias y medianas en USD. Las medias fluctúan entre años y algunas cohortes tienen pocos registros; por ello los máximos no deben interpretarse como un efecto causal de la antigüedad. La distribución de ventas por año, ubicación y características también influye.

## Requisito 7 — Modelo predictivo

Se predijo `price` con RandomForestRegressor y un baseline de media de entrenamiento. La separación se hizo con GroupShuffleSplit usando `id` como grupo: {split['n_train']:,} ventas de entrenamiento, {split['n_test']:,} de prueba y {split['ids_compartidos']} IDs compartidos. El preprocesamiento se ajusta dentro del Pipeline. `zipcode` se codificó como categoría.

Variables usadas: `{', '.join(results['model']['features'])}`. Se excluyeron `id`, `registro_id`, `precio_por_m2` (derivada del precio), `superficie_m2` (redundante con `sqft_living`), la fecha y variables de área redundantes o de contexto (`sqft_above`, `sqft_basement`, `sqft_living15`, `sqft_lot15`).

| Modelo | Partición | MAE (USD) | RMSE (USD) | R² |
|---|---|---:|---:|---:|
| Random Forest | Entrenamiento | {_usd(rf_train.mae_usd)} | {_usd(rf_train.rmse_usd)} | {rf_train.r2:.4f} |
| Random Forest | Prueba | {_usd(rf_test.mae_usd)} | {_usd(rf_test.rmse_usd)} | {rf_test.r2:.4f} |
| Baseline de media | Prueba | {_usd(base_test.mae_usd)} | {_usd(base_test.rmse_usd)} | {base_test.r2:.4f} |

El bosque {"supera" if results['model']['beats_baseline'] else "no supera"} al baseline en MAE, RMSE y R² de prueba. La brecha entre entrenamiento y prueba es evidencia de ajuste optimista en los datos de entrenamiento; la métrica de prueba agrupada es la estimación principal. La evaluación es aleatoria por propiedad, no una validación temporal para años futuros.

## Requisito 8 — Superficie y precio

{corr_lines}

Ambos coeficientes indican asociación positiva entre superficie habitable y precio. Los p-valores numéricamente menores al límite de representación se muestran como `<1e-300`; correlación no implica causalidad.

## Requisito 9 — Propiedades frente al agua

{water_lines}

`waterfront` identifica una propiedad frente al agua; no confirma necesariamente vista al mar. Las diferencias son descriptivas y no atribuyen causalidad a esta variable.

## Requisito 10 — Segmentos de precio

Los cuatro segmentos se definieron con cuartiles de las 21.613 ventas, no con límites oficiales del mercado:

{segment_lines}

## ANOVA por código postal

Se creó `propiedades_ultima_venta.csv` seleccionando, por `id`, la venta de fecha más reciente. Contiene {anova['n_propiedades_ultima_venta']:,} propiedades y {anova['n_zipcodes']} ZIP. H₀: el precio medio de la última venta es igual en todos los ZIP. H₁: al menos un ZIP tiene una media distinta.

Se aplicó Levene con centro mediano: estadístico {anova['levene_statistico']:.3f}, p={anova['levene_p_presentacion']}. Se seleccionó **{anova['metodo']}**, F={anova['estadistico_F']:.3f}, p={anova['p_valor_presentacion']}, α=0,05. Resultado: {"se rechaza H₀; existe evidencia de que al menos una media difiere" if anova['rechaza_H0'] else "no se rechaza H₀"}. Post hoc: {anova['posthoc']}; {anova['pares_posthoc_significativos']:,} pares significativos de {anova['pares_posthoc_evaluados']:,} evaluados.

Shapiro aplicado a una muestra reproducible de {anova['shapiro_n_residuos']:,} residuos dio W={anova['shapiro_W']:.3f}, p={anova['shapiro_p_presentacion']}; la gráfica Q–Q muestra desviaciones de normalidad. Una propiedad por ID elimina la repetición de ventas dentro de la tabla del ANOVA, pero no elimina una posible dependencia espacial entre viviendas cercanas. ZIP es una aproximación geográfica, no una división oficial por barrio. Welch y Games–Howell no prueban causalidad.

## Archivos de salida

Las tablas y gráficas están en `salidas/`. El modelo serializado y el manifiesto de variables están en `salidas/modelos/` y `salidas/validacion_modelo.json`. El CSV recibido permanece intacto; las conversiones de tipos y columnas auxiliares se hacen en memoria.
"""
    (REPORTS / "informe_resultados.md").write_text(report, encoding="utf-8")


def main():
    prepare_dirs()
    print(f"Fuente oficial: {SOURCE}")
    df = load_official_data()
    print(f"Datos cargados sin descartar filas: {len(df):,} ventas")
    print("Ejecutando R6–R10, modelo agrupado por propiedad y ANOVA por última venta...")
    results = run_analyses(df)
    _write_report(results)
    print("Análisis completado.")
    print(f"Salidas: {OUTPUT}")
    print(f"Informe: {REPORTS / 'informe_resultados.md'}")


if __name__ == "__main__":
    main()
