"""Rutas del componente estadístico del proyecto inmobiliario."""

import os
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "datos"
SOURCE = Path(
    os.environ.get(
        "KC_HOUSE_SOURCE",
        DATA / "CSV procesado de viviendas.csv",
    )
)
OUTPUT = ROOT / "salidas"
FIGURES = OUTPUT / "graficas"
MODEL_DIR = OUTPUT / "modelos"
REPORTS = ROOT / "informes"


def prepare_dirs():
    for path in (DATA, OUTPUT, FIGURES, MODEL_DIR, REPORTS):
        path.mkdir(parents=True, exist_ok=True)
