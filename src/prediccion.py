"""CLI sencilla para inferencia con el pipeline serializado."""
import argparse
import pandas as pd
import joblib
from .configuracion import MODEL_DIR
from .analisis import FEATURES

def main():
    parser = argparse.ArgumentParser(description="Predice precio estimado para una vivienda King County.")
    for feature in FEATURES:
        parser.add_argument(f"--{feature}", required=True, type=str if feature == "zipcode" else float)
    args = parser.parse_args()
    model = joblib.load(MODEL_DIR / "random_forest_house_price.joblib")
    row = {name: getattr(args, name) for name in FEATURES}
    print(f"Precio estimado: ${model.predict(pd.DataFrame([row]))[0]:,.2f} USD")

if __name__ == "__main__":
    main()
