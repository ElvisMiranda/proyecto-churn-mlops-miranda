from pathlib import Path

import joblib
import pandas as pd
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

BASE_DIR = Path(__file__).resolve().parents[1]
DATA_DIR = BASE_DIR / "data"
MODELS_DIR = BASE_DIR / "models"

TRAIN_DATA = DATA_DIR / "train.csv"
MODELO_LR = MODELS_DIR / "modelo_churn_lr.pkl"
MODELO_GB = MODELS_DIR / "modelo_churn_gb.pkl"

ALGORITMOS = {
    "lr": ("Regresión Logística", LogisticRegression()),
    "gb": ("Gradient Boosting", GradientBoostingClassifier(random_state=42)),
}

def entrenar_modelo(algoritmo: str = "lr"):
    if not TRAIN_DATA.exists():
        raise FileNotFoundError(
            "No se encontró data/train.csv. Primero ejecuta src/preparar_datos.py"
        )

    MODELS_DIR.mkdir(exist_ok=True)

    df = pd.read_csv(TRAIN_DATA)

    X = df.drop(columns=["churn"])
    y = df["churn"]

    nombre, clasificador = ALGORITMOS[algoritmo]

    modelo = Pipeline(
        steps=[
            ("escalado", StandardScaler()),
            ("clasificador", clasificador),
        ]
    )

    modelo.fit(X, y)

    ruta = MODELO_LR if algoritmo == "lr" else MODELO_GB
    joblib.dump(modelo, ruta)

    print(f"Modelo [{nombre}] entrenado correctamente.")
    print(f"Guardado en: {ruta}")

def entrenar_todos():
    for alg in ALGORITMOS:
        entrenar_modelo(alg)
    print("\nAmbos modelos entrenados y guardados.")

if __name__ == "__main__":
    entrenar_todos()
