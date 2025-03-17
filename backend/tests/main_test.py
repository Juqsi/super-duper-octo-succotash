from fastapi.testclient import TestClient

from api.main import app

client = TestClient(app)


def test_hello_world():
    response = client.get("/")
    assert response.status_code == 200  # Statuscode sollte 200 OK sein
    # Antwort sollte das erwartete JSON sein
    assert response.json() == {"message": "Hello World 🌍"}
