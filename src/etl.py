"""Carga y validación del CSV oficial ya preparado por el Integrante 1.

Este módulo no vuelve a limpiar el archivo ni elimina ventas repetidas por ID.
Solo convierte tipos en memoria para que los análisis usen fecha y ZIP con la
semántica adecuada. Las columnas originales y adicionales se conservan.
"""

import json
from pathlib import Path

import pandas as pd

from .configuracion import OUTPUT, SOURCE, prepare_dirs

REQUIRED_COLUMNS = {
    "id", "date", "price", "bedrooms", "bathrooms", "sqft_living",
    "sqft_lot", "floors", "waterfront", "view", "condition", "grade",
    "yr_built", "yr_renovated", "zipcode", "lat", "long",
    "precio_por_m2", "superficie_m2", "registro_id",
}
NUMERIC_COLUMNS = {
    "price", "bedrooms", "bathrooms", "sqft_living", "sqft_lot",
    "floors", "waterfront", "view", "condition", "grade", "sqft_above",
    "sqft_basement", "yr_built", "yr_renovated", "lat", "long",
    "sqft_living15", "sqft_lot15", "precio_por_m2", "superficie_m2",
    "registro_id",
}


def load_official_data(source: Path = SOURCE) -> pd.DataFrame:
    """Carga el archivo oficial sin descartar filas ni columnas."""
    source = Path(source)
    if not source.is_file():
        raise FileNotFoundError(
            f"No se encuentra el CSV oficial: {source}. "
            "Configure KC_HOUSE_SOURCE o coloque el archivo en datos/."
        )

    df = pd.read_csv(source, low_memory=False)
    missing = sorted(REQUIRED_COLUMNS - set(df.columns))
    if missing:
        raise ValueError(f"El CSV oficial no contiene columnas requeridas: {missing}")

    source_rows = len(df)
    source_columns = list(df.columns)
    null_counts = df.isna().sum()

    # El ID identifica la propiedad; se conserva como etiqueta y no se usa como
    # predictor. ZIP también se fuerza a texto para tratarlo como categoría.
    df["id"] = df["id"].astype("string")
    df["zipcode"] = df["zipcode"].astype("string")
    df["date"] = pd.to_datetime(df["date"], errors="coerce")

    for column in NUMERIC_COLUMNS.intersection(df.columns):
        df[column] = pd.to_numeric(df[column], errors="raise")

    if df["date"].isna().any():
        raise ValueError("Hay fechas nulas o que no se pudieron convertir.")
    if df["id"].isna().any() or df["price"].isna().any():
        raise ValueError("Hay IDs o precios nulos en el CSV oficial.")
    if df["registro_id"].isna().any() or df["registro_id"].duplicated().any():
        raise ValueError("registro_id debe identificar cada fila del CSV oficial.")
    if df.duplicated(["id", "date"]).any():
        raise ValueError("El CSV contiene claves id-date repetidas; requieren revisión.")

    duplicate_id_rows = int(df["id"].duplicated(keep="first").sum())
    repeated_id_groups = int(df.loc[df["id"].duplicated(keep=False), "id"].nunique())
    diagnostic = {
        "fuente_oficial": str(source),
        "filas_leidas": int(source_rows),
        "filas_devuelta_por_carga": int(len(df)),
        "columnas_conservadas": source_columns,
        "columnas": int(len(source_columns)),
        "nulos_por_columna": {str(k): int(v) for k, v in null_counts.items()},
        "filas_completamente_duplicadas": int(df.duplicated().sum()),
        "filas_adicionales_por_id_repetido": duplicate_id_rows,
        "ids_de_propiedad_repetidos": repeated_id_groups,
        "ids_unicos": int(df["id"].nunique()),
        "registro_id_unicos": int(df["registro_id"].nunique()),
        "registros_eliminados": 0,
        "transformaciones_en_memoria": [
            "date convertida a datetime",
            "id conservado como texto",
            "zipcode convertido a texto categórico",
            "columnas originales y adicionales conservadas",
        ],
    }
    prepare_dirs()
    (OUTPUT / "diagnostico_fuente.json").write_text(
        json.dumps(diagnostic, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    return df
