"""Análisis estadístico y predicción sobre las ventas oficiales de King County."""

import json
import os
import warnings
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", str(Path(__file__).resolve().parents[1] / ".mplconfig"))

import joblib
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import statsmodels.api as sm
from scipy import stats
from scipy.stats import studentized_range
from sklearn.compose import ColumnTransformer
from sklearn.dummy import DummyRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.impute import SimpleImputer
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import GroupShuffleSplit
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder
from statsmodels.formula.api import ols
from statsmodels.stats.anova import anova_lm
from statsmodels.stats.multicomp import pairwise_tukeyhsd
from statsmodels.stats.oneway import anova_oneway

from .configuracion import FIGURES, MODEL_DIR, OUTPUT, prepare_dirs

TARGET = "price"
CATEGORICAL_FEATURES = ["zipcode"]
NUMERIC_FEATURES = [
    "bedrooms", "bathrooms", "sqft_living", "sqft_lot", "floors",
    "waterfront", "view", "condition", "grade", "yr_built",
    "yr_renovated", "lat", "long",
]
FEATURES = NUMERIC_FEATURES + CATEGORICAL_FEATURES
PRICE_SEGMENTS = ["Económico", "Medio", "Alto", "Premium"]
ALPHA = 0.05


def _p_display(value):
    if value == 0:
        return "<1e-300 (subdesbordamiento numérico)"
    return f"{value:.4g}"


def _savefig(filename):
    plt.tight_layout()
    plt.savefig(FIGURES / filename, dpi=160, bbox_inches="tight")
    plt.close()


def _metrics(actual, predicted):
    return {
        "mae_usd": float(mean_absolute_error(actual, predicted)),
        "rmse_usd": float(mean_squared_error(actual, predicted) ** 0.5),
        "r2": float(r2_score(actual, predicted)),
    }


def _fit_grouped_model(df):
    """Entrena el bosque y baseline con propiedades aisladas entre particiones."""
    x = df[FEATURES].copy()
    x["zipcode"] = x["zipcode"].astype("string")
    y = df[TARGET].astype(float)
    groups = df["id"].astype("string")

    splitter = GroupShuffleSplit(n_splits=1, test_size=0.20, random_state=42)
    train_idx, test_idx = next(splitter.split(x, y, groups=groups))
    x_train, x_test = x.iloc[train_idx], x.iloc[test_idx]
    y_train, y_test = y.iloc[train_idx], y.iloc[test_idx]
    ids_train, ids_test = groups.iloc[train_idx], groups.iloc[test_idx]
    overlap = set(ids_train) & set(ids_test)
    if overlap:
        raise AssertionError(f"Fuga de propiedades entre particiones: {len(overlap)} IDs")

    numeric_pipe = Pipeline([("imputer", SimpleImputer(strategy="median"))])
    categorical_pipe = Pipeline([
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("onehot", OneHotEncoder(handle_unknown="ignore")),
    ])
    preprocessor = ColumnTransformer([
        ("numeric", numeric_pipe, NUMERIC_FEATURES),
        ("categorical", categorical_pipe, CATEGORICAL_FEATURES),
    ])
    model = Pipeline([
        ("preprocess", preprocessor),
        ("random_forest", RandomForestRegressor(
            n_estimators=300,
            min_samples_leaf=2,
            random_state=42,
            n_jobs=1,
        )),
    ])
    model.fit(x_train, y_train)
    pred_train = model.predict(x_train)
    pred_test = model.predict(x_test)

    baseline = DummyRegressor(strategy="mean")
    baseline.fit(np.zeros((len(y_train), 1)), y_train)
    baseline_train = baseline.predict(np.zeros((len(y_train), 1)))
    baseline_test = baseline.predict(np.zeros((len(y_test), 1)))

    rows = []
    for name, split, actual, predicted in [
        ("RandomForestRegressor", "train", y_train, pred_train),
        ("RandomForestRegressor", "test", y_test, pred_test),
        ("BaselineMedia", "train", y_train, baseline_train),
        ("BaselineMedia", "test", y_test, baseline_test),
    ]:
        rows.append({"modelo": name, "particion": split, "n_ventas": len(actual), **_metrics(actual, predicted)})
    metrics = pd.DataFrame(rows)
    rf_test = metrics.query("modelo == 'RandomForestRegressor' and particion == 'test'").iloc[0]
    base_test = metrics.query("modelo == 'BaselineMedia' and particion == 'test'").iloc[0]
    beats_baseline = bool(
        rf_test.mae_usd < base_test.mae_usd
        and rf_test.rmse_usd < base_test.rmse_usd
        and rf_test.r2 > base_test.r2
    )

    split_data = pd.DataFrame({
        "registro_id": df["registro_id"].to_numpy(),
        "id": groups.to_numpy(),
        "particion": np.where(np.isin(np.arange(len(df)), train_idx), "train", "test"),
    })
    overlap_ids = sorted(overlap)
    split_summary = {
        "n_train": int(len(train_idx)),
        "n_test": int(len(test_idx)),
        "ids_train": int(ids_train.nunique()),
        "ids_test": int(ids_test.nunique()),
        "ids_compartidos": int(len(overlap_ids)),
        "random_state": 42,
        "test_size_target": 0.20,
        "modelo_supera_baseline_en_mae_rmse_r2": beats_baseline,
    }

    predictions = pd.DataFrame({
        "registro_id": df.iloc[test_idx]["registro_id"].to_numpy(),
        "id": groups.iloc[test_idx].to_numpy(),
        "actual_usd": y_test.to_numpy(),
        "predicho_usd": pred_test,
    })
    joblib.dump(model, MODEL_DIR / "random_forest_house_price.joblib")
    metrics.to_csv(OUTPUT / "metricas_modelo.csv", index=False, encoding="utf-8")
    predictions.to_csv(OUTPUT / "predicciones_test.csv", index=False, encoding="utf-8")
    split_data.to_csv(OUTPUT / "particion_modelo.csv", index=False, encoding="utf-8")
    (OUTPUT / "validacion_modelo.json").write_text(
        json.dumps({
            **split_summary,
            "variable_objetivo": TARGET,
            "variables_predictoras": FEATURES,
            "zipcode_categorica": True,
            "excluidas": ["id", "registro_id", "precio_por_m2", "superficie_m2", "date", "sqft_above", "sqft_basement", "sqft_living15", "sqft_lot15"],
        }, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    names = model.named_steps["preprocess"].get_feature_names_out()
    importances = model.named_steps["random_forest"].feature_importances_
    importance = pd.DataFrame({"variable": names, "importancia": importances}).sort_values("importancia", ascending=False)
    importance.to_csv(OUTPUT / "importancia_variables.csv", index=False, encoding="utf-8")

    plt.figure(figsize=(7, 7))
    sns.scatterplot(x=y_test, y=pred_test, alpha=0.35, s=18)
    limits = [min(y_test.min(), pred_test.min()), max(y_test.max(), pred_test.max())]
    plt.plot(limits, limits, "r--", label="Predicción perfecta")
    plt.xlabel("Precio real (USD)")
    plt.ylabel("Precio predicho (USD)")
    plt.title("Precios reales y predichos — conjunto de prueba")
    plt.legend()
    _savefig("real_vs_predicho.png")

    top = importance.head(15).sort_values("importancia")
    plt.figure(figsize=(9, 6))
    sns.barplot(data=top, x="importancia", y="variable")
    plt.xlabel("Importancia relativa")
    plt.ylabel("Variable")
    plt.title("Importancia de variables del Random Forest")
    _savefig("importancia_variables.png")

    return {
        "model": model,
        "metrics": metrics,
        "split": split_summary,
        "beats_baseline": beats_baseline,
        "features": FEATURES.copy(),
    }


def _run_anova_last_sale(df):
    """ANOVA de precios por ZIP sobre una venta reciente por cada propiedad."""
    properties = (
        df.sort_values(["id", "date", "registro_id"])
        .groupby("id", sort=False, as_index=False)
        .tail(1)
        .sort_values("id")
        .reset_index(drop=True)
    )
    if properties["id"].nunique() != 21436 or len(properties) != 21436:
        raise AssertionError(
            f"Se esperaban 21.436 propiedades únicas; se obtuvieron {len(properties):,} filas "
            f"y {properties['id'].nunique():,} IDs."
        )
    properties.to_csv(OUTPUT / "propiedades_ultima_venta.csv", index=False, encoding="utf-8", date_format="%Y-%m-%d")

    groups_with_zip = list(properties.groupby("zipcode", sort=True, observed=True))
    groups = [group["price"].to_numpy(dtype=float) for _, group in groups_with_zip if len(group) >= 2]
    if len(groups) < 2:
        raise ValueError("ANOVA requiere al menos dos ZIP con dos o más propiedades.")

    ols_fit = ols("price ~ C(zipcode)", data=properties).fit()
    residuals = ols_fit.resid
    normality_sample = residuals.sample(min(5000, len(residuals)), random_state=42)
    shapiro = stats.shapiro(normality_sample)
    levene = stats.levene(*groups, center="median")

    if levene.pvalue >= ALPHA:
        anova_table = anova_lm(ols_fit, typ=2)
        f_stat = float(anova_table.loc["C(zipcode)", "F"])
        p_value = float(anova_table.loc["C(zipcode)", "PR(>F)"])
        method = "ANOVA clásico"
    else:
        welch = anova_oneway(groups, use_var="unequal")
        f_stat = float(welch.statistic)
        p_value = float(welch.pvalue)
        method = "ANOVA de Welch"

    posthoc = pd.DataFrame()
    if p_value < ALPHA:
        if method == "ANOVA clásico":
            tukey = pairwise_tukeyhsd(properties["price"], properties["zipcode"].astype(str), alpha=ALPHA)
            posthoc = pd.DataFrame(tukey.summary().data[1:], columns=tukey.summary().data[0])
            posthoc.to_csv(OUTPUT / "posthoc_tukey_zipcode.csv", index=False, encoding="utf-8")
            posthoc_name = "Tukey HSD"
            significant_pairs = int(posthoc["reject"].sum())
        else:
            pairs = []
            k = len(groups_with_zip)
            for index, (zip_a, group_a) in enumerate(groups_with_zip):
                a = group_a["price"].to_numpy(dtype=float)
                for zip_b, group_b in groups_with_zip[index + 1:]:
                    b = group_b["price"].to_numpy(dtype=float)
                    if len(a) < 2 or len(b) < 2:
                        continue
                    variance_a, variance_b = a.var(ddof=1), b.var(ddof=1)
                    component_a, component_b = variance_a / len(a), variance_b / len(b)
                    se_squared = component_a + component_b
                    dof = se_squared**2 / (
                        component_a**2 / (len(a) - 1) + component_b**2 / (len(b) - 1)
                    )
                    q_stat = abs(a.mean() - b.mean()) / np.sqrt(se_squared / 2)
                    adjusted_p = float(studentized_range.sf(q_stat, k, dof))
                    pairs.append({
                        "zipcode_1": str(zip_a),
                        "zipcode_2": str(zip_b),
                        "diferencia_media_usd": float(a.mean() - b.mean()),
                        "q": float(q_stat),
                        "gl_welch": float(dof),
                        "p_ajustado": adjusted_p,
                        "significativo": bool(adjusted_p < ALPHA),
                    })
            posthoc = pd.DataFrame(pairs)
            posthoc.to_csv(OUTPUT / "posthoc_games_howell_zipcode.csv", index=False, encoding="utf-8")
            posthoc_name = "Games–Howell"
            significant_pairs = int(posthoc["significativo"].sum())
    else:
        posthoc_name = "No aplica: ANOVA no significativo"
        significant_pairs = 0

    result = {
        "metodo": method,
        "posthoc": posthoc_name,
        "n_ventas_por_propiedad": int(len(df)),
        "n_propiedades_ultima_venta": int(len(properties)),
        "n_zipcodes": int(properties["zipcode"].nunique()),
        "estadistico_F": f_stat,
        "p_valor": p_value,
        "p_valor_presentacion": _p_display(p_value),
        "alpha": ALPHA,
        "levene_statistico": float(levene.statistic),
        "levene_p": float(levene.pvalue),
        "levene_p_presentacion": _p_display(float(levene.pvalue)),
        "shapiro_n_residuos": int(len(normality_sample)),
        "shapiro_W": float(shapiro.statistic),
        "shapiro_p": float(shapiro.pvalue),
        "shapiro_p_presentacion": _p_display(float(shapiro.pvalue)),
        "rechaza_H0": bool(p_value < ALPHA),
        "pares_posthoc_significativos": significant_pairs,
        "pares_posthoc_evaluados": int(len(posthoc)),
        "H0": "El precio medio de la última venta es igual en todos los códigos postales.",
        "H1": "Al menos un código postal tiene un precio medio distinto.",
        "unidad_analisis": "Una última venta por propiedad, agrupada por id.",
        "independencia": "No hay IDs repetidos en la tabla ANOVA; puede persistir dependencia espacial entre viviendas cercanas.",
    }
    pd.DataFrame([result]).to_csv(OUTPUT / "anova_zipcode.csv", index=False, encoding="utf-8")

    summary_by_zip = properties.groupby("zipcode", as_index=False, observed=True).agg(
        cantidad_propiedades=("id", "count"),
        precio_medio_usd=("price", "mean"),
        precio_mediano_usd=("price", "median"),
    )
    summary_by_zip.to_csv(OUTPUT / "resumen_anova_zipcode.csv", index=False, encoding="utf-8")

    plt.figure(figsize=(14, 6))
    sns.boxplot(data=properties, x="zipcode", y="price", showfliers=False, color="#5478A8")
    plt.title("Distribución de precios por código postal — última venta por propiedad")
    plt.xlabel("Código postal")
    plt.ylabel("Precio (USD)")
    plt.xticks(rotation=90, fontsize=7)
    _savefig("anova_precios_zipcode.png")

    sm.qqplot(normality_sample, line="45", fit=True)
    plt.title("Gráfica Q–Q de residuos del modelo por código postal")
    plt.xlabel("Cuantiles teóricos")
    plt.ylabel("Cuantiles de residuos")
    _savefig("anova_qq_residuos.png")

    return result


def run_analyses(df):
    """Genera análisis R6–R10 y ANOVA, con ventas como grano principal."""
    prepare_dirs()
    sns.set_theme(style="whitegrid")
    work = df.copy()
    work["zipcode"] = work["zipcode"].astype("string")
    work["waterfront_label"] = work["waterfront"].map({0: "No", 1: "Sí"}).fillna("Desconocido")
    work["sale_year"] = work["date"].dt.year.astype("int64")
    work["price_segment"] = pd.qcut(work["price"], q=4, labels=PRICE_SEGMENTS)

    # Indicadores auxiliares para Power BI, siempre calculados sobre las 21.613 ventas.
    work.groupby("sale_year", as_index=False).agg(
        cantidad_ventas=("registro_id", "count"),
        precio_medio_usd=("price", "mean"),
        precio_mediano_usd=("price", "median"),
    ).to_csv(OUTPUT / "ventas_por_anio.csv", index=False, encoding="utf-8")
    work.groupby("zipcode", as_index=False, observed=True).agg(
        cantidad_ventas=("registro_id", "count"),
        precio_medio_usd=("price", "mean"),
        precio_mediano_usd=("price", "median"),
        precio_medio_m2_usd=("precio_por_m2", "mean"),
        latitud=("lat", "mean"),
        longitud=("long", "mean"),
    ).to_csv(OUTPUT / "mapa_precios_zipcode.csv", index=False, encoding="utf-8")
    work.groupby("zipcode", as_index=False, observed=True).agg(
        cantidad_ventas=("registro_id", "count"),
        precio_medio_usd=("price", "mean"),
    ).sort_values("precio_medio_usd", ascending=False).head(10).to_csv(
        OUTPUT / "top_10_zipcodes.csv", index=False, encoding="utf-8"
    )
    work.groupby(["bedrooms", "bathrooms"], as_index=False).agg(
        cantidad_ventas=("registro_id", "count"),
        precio_medio_usd=("price", "mean"),
    ).to_csv(OUTPUT / "precio_habitaciones_banos.csv", index=False, encoding="utf-8")

    # R6: años de construcción agrupan ventas, no viviendas únicas.
    by_year = work.groupby("yr_built", as_index=False).agg(
        cantidad_ventas=("registro_id", "count"),
        precio_medio_usd=("price", "mean"),
        precio_mediano_usd=("price", "median"),
    )
    by_year.to_csv(OUTPUT / "precio_por_anio_construccion.csv", index=False, encoding="utf-8")
    plt.figure(figsize=(11, 5))
    plt.plot(by_year.yr_built, by_year.precio_medio_usd, label="Media", linewidth=1.8)
    plt.plot(by_year.yr_built, by_year.precio_mediano_usd, label="Mediana", linewidth=1.5, linestyle="--")
    plt.title("Precio de venta por año de construcción")
    plt.xlabel("Año de construcción")
    plt.ylabel("Precio (USD)")
    plt.legend()
    _savefig("precio_por_anio_construccion.png")

    model_results = _fit_grouped_model(work)

    # R8: correlación de ventas, sin interpretación causal.
    pearson = stats.pearsonr(work["sqft_living"], work["price"])
    spearman = stats.spearmanr(work["sqft_living"], work["price"])
    correlation = pd.DataFrame([
        {"metodo": "Pearson", "coeficiente": float(pearson.statistic), "p_valor": float(pearson.pvalue), "p_presentacion": _p_display(float(pearson.pvalue)), "n_ventas": len(work)},
        {"metodo": "Spearman", "coeficiente": float(spearman.statistic), "p_valor": float(spearman.pvalue), "p_presentacion": _p_display(float(spearman.pvalue)), "n_ventas": len(work)},
    ])
    correlation.to_csv(OUTPUT / "correlacion_superficie_precio.csv", index=False, encoding="utf-8")
    plt.figure(figsize=(9, 6))
    sns.regplot(data=work, x="sqft_living", y="price", scatter_kws={"alpha": 0.2, "s": 12}, line_kws={"color": "red"})
    plt.title("Superficie habitable y precio de venta")
    plt.xlabel("Superficie habitable (ft²)")
    plt.ylabel("Precio (USD)")
    _savefig("correlacion_superficie_precio.png")

    # R9: waterfront describe propiedades frente al agua, no vista al mar.
    waterfront = work.groupby("waterfront_label", as_index=False, observed=True).agg(
        cantidad_ventas=("registro_id", "count"),
        precio_medio_usd=("price", "mean"),
        precio_mediano_usd=("price", "median"),
        desviacion_estandar_usd=("price", "std"),
    )
    waterfront.to_csv(OUTPUT / "analisis_frente_agua.csv", index=False, encoding="utf-8")
    plt.figure(figsize=(7, 5))
    sns.boxplot(data=work, x="waterfront_label", y="price", showfliers=False)
    plt.title("Precio de venta según propiedad frente al agua")
    plt.xlabel("Propiedad frente al agua")
    plt.ylabel("Precio (USD)")
    _savefig("frente_agua_boxplot.png")

    # R10: bandas estadísticas definidas por cuartiles de las ventas.
    work[["registro_id", "id", "price", "price_segment"]].rename(
        columns={"price": "precio_usd", "price_segment": "segmento_precio"}
    ).to_csv(OUTPUT / "ventas_segmentadas.csv", index=False, encoding="utf-8")
    segments = work.groupby("price_segment", as_index=False, observed=False).agg(
        cantidad_ventas=("registro_id", "count"),
        precio_medio_usd=("price", "mean"),
        precio_mediano_usd=("price", "median"),
        precio_minimo_usd=("price", "min"),
        precio_maximo_usd=("price", "max"),
    )
    segments.to_csv(OUTPUT / "resumen_segmentos_precio.csv", index=False, encoding="utf-8")
    plt.figure(figsize=(8, 5))
    sns.countplot(data=work, x="price_segment", order=PRICE_SEGMENTS)
    plt.title("Cantidad de ventas por segmento de precio")
    plt.xlabel("Segmento por cuartil")
    plt.ylabel("Cantidad de ventas")
    _savefig("segmentos_precio.png")

    # ANOVA: una última venta por cada propiedad para evitar pseudorreplicación.
    anova = _run_anova_last_sale(work)

    # Verificación de que los principales análisis conservan la granularidad de venta.
    expected_rows = len(work)
    row_counts = {
        "ventas_fuente": int(expected_rows),
        "ventas_R6": int(by_year.cantidad_ventas.sum()),
        "ventas_R8": int(correlation.n_ventas.iloc[0]),
        "ventas_R9": int(waterfront.cantidad_ventas.sum()),
        "ventas_R10": int(segments.cantidad_ventas.sum()),
        "propiedades_ANOVA": int(anova["n_propiedades_ultima_venta"]),
    }
    if any(row_counts[key] != expected_rows for key in ["ventas_R6", "ventas_R8", "ventas_R9", "ventas_R10"]):
        raise AssertionError(f"Algún análisis no utilizó todas las ventas: {row_counts}")
    (OUTPUT / "conteos_analisis.json").write_text(
        json.dumps(row_counts, ensure_ascii=False, indent=2), encoding="utf-8"
    )

    return {
        "source_rows": expected_rows,
        "unique_properties": int(work.id.nunique()),
        "row_counts": row_counts,
        "r6": by_year,
        "model": model_results,
        "correlation": correlation,
        "waterfront": waterfront,
        "segments": segments,
        "anova": anova,
    }
