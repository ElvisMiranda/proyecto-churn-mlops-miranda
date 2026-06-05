from pathlib import Path

import joblib
import pandas as pd
from fastapi import FastAPI, HTTPException, Query
from pydantic import BaseModel

BASE_DIR = Path(__file__).resolve().parents[1]
MODELS_DIR = BASE_DIR / "models"
RUTAS_MODELOS = {
    "lr": MODELS_DIR / "modelo_churn_lr.pkl",
    "gb": MODELS_DIR / "modelo_churn_gb.pkl",
}
NOMBRES_MODELOS = {
    "lr": "Regresión Logística",
    "gb": "Gradient Boosting",
}

app = FastAPI(
    title="API de Predicción de Churn",
    version="0.2.0",
    description="API para consumir modelos de Machine Learning (LR y GB)."
)

class Cliente(BaseModel):
    edad: int
    antiguedad_meses: int
    saldo_promedio: float
    reclamos: int
    usa_app: int

def cargar_modelo(algoritmo: str):
    ruta = RUTAS_MODELOS[algoritmo]
    if not ruta.exists():
        return None
    return joblib.load(ruta)

@app.get("/")
def inicio():
    return {
        "mensaje": "Servicio ML-Ops activo",
        "estado": "ok",
        "author": "Elvis Miranda Aramayo",
        "modelos_disponibles": list(NOMBRES_MODELOS.keys()),
    }

@app.get("/health")
def health():
    disponibles = {k: v.exists() for k, v in RUTAS_MODELOS.items()}
    return {
        "estado": "ok",
        "modelos": disponibles,
    }

@app.post("/predict")
def predict(
    cliente: Cliente,
    modelo: str = Query("lr", description="Algoritmo: lr (Logistic Regression) o gb (Gradient Boosting)"),
):
    if modelo not in RUTAS_MODELOS:
        raise HTTPException(
            status_code=400,
            detail=f"Modelo '{modelo}' no válido. Opciones: {list(RUTAS_MODELOS.keys())}",
        )

    model = cargar_modelo(modelo)

    if model is None:
        raise HTTPException(
            status_code=503,
            detail=f"Modelo '{NOMBRES_MODELOS[modelo]}' no disponible. Entrénalo primero con src/entrenar_modelo.py"
        )

    datos = pd.DataFrame([cliente.model_dump()])

    prediccion = int(model.predict(datos)[0])

    probabilidad = None
    if hasattr(model, "predict_proba"):
        probabilidad = float(model.predict_proba(datos)[0][1])

    return {
        "modelo": modelo,
        "algoritmo": NOMBRES_MODELOS[modelo],
        "churn_predicho": prediccion,
        "probabilidad_churn": probabilidad,
    }
