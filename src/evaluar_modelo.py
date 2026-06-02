from pathlib import Path

import joblib
import pandas as pd
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score

BASE_DIR = Path(__file__).resolve().parents[1]
DATA_DIR = BASE_DIR / "data"
MODELS_DIR = BASE_DIR / "models"
DOCS_DIR = BASE_DIR / "docs"

TEST_DATA = DATA_DIR / "test.csv"
MODELOS = {
    "lr": ("Regresión Logística", MODELS_DIR / "modelo_churn_lr.pkl"),
    "gb": ("Gradient Boosting", MODELS_DIR / "modelo_churn_gb.pkl"),
}
METRICS_FILE = DOCS_DIR / "metricas_modelo.md"

def evaluar_modelo(algoritmo: str):
    nombre, ruta = MODELOS[algoritmo]

    if not ruta.exists():
        raise FileNotFoundError(
            f"No se encontró {ruta}. Primero ejecuta src/entrenar_modelo.py"
        )

    modelo = joblib.load(ruta)
    y_pred = modelo.predict(X_test)

    return {
        "modelo": nombre,
        "accuracy": accuracy_score(y_test, y_pred),
        "precision": precision_score(y_test, y_pred, zero_division=0),
        "recall": recall_score(y_test, y_pred, zero_division=0),
        "f1": f1_score(y_test, y_pred, zero_division=0),
    }

def evaluar_todos():
    if not TEST_DATA.exists():
        raise FileNotFoundError(
            "No se encontró data/test.csv. Primero ejecuta src/preparar_datos.py"
        )

    DOCS_DIR.mkdir(exist_ok=True)

    global X_test, y_test
    df = pd.read_csv(TEST_DATA)
    X_test = df.drop(columns=["churn"])
    y_test = df["churn"]

    resultados = [evaluar_modelo(alg) for alg in MODELOS]

    filas = ""
    for r in resultados:
        filas += f"| {r['modelo']} | {r['accuracy']:.4f} | {r['precision']:.4f} | {r['recall']:.4f} | {r['f1']:.4f} |\n"

    contenido = f"""# Métricas de modelos de churn

## Comparación de modelos

| Modelo | Accuracy | Precision | Recall | F1-score |
|---|---:|---:|---:|---:|
{filas}
## Interpretación

- **Accuracy**: porcentaje general de aciertos.
- **Precision**: qué tan confiables son las predicciones positivas.
- **Recall**: qué proporción de clientes con churn fueron identificados.
- **F1-score**: resumen de precision y recall en una sola métrica.
"""

    METRICS_FILE.write_text(contenido, encoding="utf-8")

    print("Modelos evaluados correctamente.")
    print(f"Métricas guardadas en: {METRICS_FILE}")

if __name__ == "__main__":
    evaluar_todos()
