from fastapi.testclient import TestClient
from api.main import app

client = TestClient(app)


def test_inicio():
    response = client.get("/")
    assert response.status_code == 200
    data = response.json()
    assert data["mensaje"] == "Servicio ML-Ops activo"
    assert data["estado"] == "ok"
    assert "autor" in data


def test_health():
    response = client.get("/health")
    assert response.status_code == 200
    data = response.json()
    assert data["estado"] == "ok"
    assert data["modelo"] == "modelo_churn_v1"
    assert data["monitoreo"] == "activo"


def test_info():
    response = client.get("/info")
    assert response.status_code == 200
    data = response.json()
    assert data["version_modelo"] == "modelo_churn_v1"
    assert "autor" in data
    assert "variables_utilizadas" in data
    assert isinstance(data["variables_utilizadas"], list)
    assert len(data["variables_utilizadas"]) > 0


def test_predict_valido_alto_riesgo():
    response = client.post("/predict", json={
        "antiguedad": 2,
        "cargo_mensual": 140.0,
        "reclamos": 7,
    })
    assert response.status_code == 200
    data = response.json()
    assert data["prediccion"] == "alto_riesgo"
    assert 0.0 <= data["probabilidad"] <= 1.0


def test_predict_valido_bajo_riesgo():
    response = client.post("/predict", json={
        "antiguedad": 70,
        "cargo_mensual": 25.0,
        "reclamos": 0,
    })
    assert response.status_code == 200
    data = response.json()
    assert data["prediccion"] == "bajo_riesgo"
    assert 0.0 <= data["probabilidad"] <= 1.0


def test_predict_antiguedad_negativa():
    response = client.post("/predict", json={
        "antiguedad": -1,
        "cargo_mensual": 50.0,
        "reclamos": 1,
    })
    assert response.status_code == 422


def test_predict_reclamos_excesivos():
    response = client.post("/predict", json={
        "antiguedad": 12,
        "cargo_mensual": 50.0,
        "reclamos": 200,
    })
    assert response.status_code == 422


def test_predict_campo_faltante():
    response = client.post("/predict", json={
        "antiguedad": 12,
        "reclamos": 1,
    })
    assert response.status_code == 422


def test_metrics_estructura():
    response = client.get("/metrics")
    assert response.status_code == 200
    data = response.json()
    keys = [
        "solicitudes_totales", "errores_validacion", "errores_internos",
        "predicciones_validas", "predicciones_alto_riesgo",
        "predicciones_bajo_riesgo", "solicitudes_con_anomalias",
        "latencia_promedio_ms", "latencia_maxima_ms",
    ]
    for key in keys:
        assert key in data, f"Falta clave: {key}"
