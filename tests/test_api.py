from fastapi.testclient import TestClient

from api.main import app

client = TestClient(app)

def test_inicio():
    response = client.get("/")

    assert response.status_code == 200
    data = response.json()
    assert "mensaje" in data
    assert "modelos_disponibles" in data

def test_health():
    response = client.get("/health")

    assert response.status_code == 200
    data = response.json()
    assert "estado" in data
    assert "modelos" in data

def test_info():
    response = client.get("/info")

    assert response.status_code == 200
    data = response.json()
    assert data["version_modelo"] == "modelo_churn_v1"
    assert "autor" in data
    assert "variables_utilizadas" in data
    assert isinstance(data["variables_utilizadas"], list)
    assert len(data["variables_utilizadas"]) > 0

def test_predict_modelo_invalido():
    response = client.post("/predict?modelo=xgboost", json={
        "edad": 30,
        "antiguedad_meses": 12,
        "saldo_promedio": 2500.0,
        "reclamos": 1,
        "usa_app": 1,
    })

    assert response.status_code == 400
    assert "no válido" in response.json()["detail"]
