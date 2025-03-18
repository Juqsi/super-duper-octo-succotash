import base64
from io import BytesIO

from PIL import Image
from fastapi.testclient import TestClient

from main import MAX_IMAGE_SIZE
from main import app

client = TestClient(app)


# Funktion zum Erzeugen und Base64 Kodieren eines 1x1 Testbildes
def generate_base64_image():
    # Erstelle ein 1x1 Bild (weiß)
    image = Image.new("RGB", (1, 1), color=(255, 255, 255))

    # Speichere das Bild in einem BytesIO-Objekt
    buffer = BytesIO()
    image.save(buffer, format="JPEG")
    buffer.seek(0)

    # Base64-Kodierung des Bildes
    base64_image = base64.b64encode(buffer.read()).decode("utf-8")

    return f"data:image/jpeg;base64,{base64_image}"


def test_hello_world():
    response = client.get("/")
    assert response.status_code == 200
    assert response.json() == {"message": "Hello World 🌍"}


def test_classify_plant_valid_image():
    # Test für den "/uploads" Endpoint mit einem gültigen Base64-Bild
    valid_base64_image = generate_base64_image()

    response = client.post(
        "/uploads", json={"image_base64": valid_base64_image}
    )
    assert response.status_code == 200
    # Überprüft, ob die Vorhersage im JSON enthalten ist
    assert "prediction" in response.json()


def test_classify_plant_invalid_image():
    # Test für den "/uploads" Endpoint mit ungültigem Base64-Format
    invalid_base64_image = "data:image/jpeg;base64,invalid-}base64"

    response = client.post(
        "/uploads", json={"image_base64": invalid_base64_image}
    )
    assert response.status_code == 400
    assert response.json()["detail"] == "Ungültiges Base64-Format."


def test_image_too_large():
    # Test, wenn die Bildgröße zu groß ist
    large_base64_image = "data:image/jpeg;base64," + "A" * (MAX_IMAGE_SIZE + 1)

    response = client.post(
        "/uploads", json={"image_base64": large_base64_image}
    )
    assert response.status_code == 413
    assert response.json()[
        "detail"] == "Das Bild überschreitet die maximale Dateigröße von 5 MB."
