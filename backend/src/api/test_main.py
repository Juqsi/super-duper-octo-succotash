import base64
from io import BytesIO

from PIL import Image
from fastapi.testclient import TestClient

from .main import MAX_IMAGE_SIZE
from .main import app

client = TestClient(app)


def generate_base64_image():
    """
    Generates a 1x1 pixel white test image and encodes it in Base64 format.

    Returns:
        str: Base64-encoded string of the test image in JPEG format.
    """
    image = Image.new("RGB", (1, 1), color=(255, 255, 255))

    buffer = BytesIO()
    image.save(buffer, format="JPEG")
    buffer.seek(0)

    base64_image = base64.b64encode(buffer.read()).decode("utf-8")

    return f"data:image/jpeg;base64,{base64_image}"


def test_hello_world():
    """
    Tests the root endpoint ("/") and verifies that the expected
    "Hello World 🌍" message is returned.

    Assertions:
        - The HTTP status code must be 200.
        - The JSON response must contain the expected message.
    """
    response = client.get("/")
    assert response.status_code == 200
    assert response.json() == {"message": "Hello World 🌍"}


def test_classify_plant_valid_image():
    """
    Tests the "/uploads" endpoint with a valid Base64 image.

    Assertions:
        - The HTTP status code must be 200.
        - The JSON response must contain the key "prediction".
    """
    valid_base64_image = generate_base64_image()

    response = client.post(
        "/uploads", json={"images": [valid_base64_image]}
    )
    assert response.status_code == 200
    assert "results" in response.json()


def test_classify_plant_multiple_images():
    """
    Tests the "/uploads" endpoint with multiple Base64 images.

    Assertions:
        - The HTTP status code must be 200.
        - The JSON response must contain predictions for each uploaded image.
    """
    valid_base64_image1 = generate_base64_image()
    valid_base64_image2 = generate_base64_image()

    response = client.post(
        "/uploads", json={"images": [valid_base64_image1, valid_base64_image2]}
    )
    assert response.status_code == 200


def test_classify_plant_invalid_image():
    """
    Tests the "/uploads" endpoint with an invalid Base64 string.

    Assertions:
       - The HTTP status code must be 400.
       - The JSON response must contain the error "Invalid Base64 format.".
    """
    invalid_base64_image = "data:image/jpeg;base64,invalid-}base64"

    response = client.post(
        "/uploads", json={"images": [invalid_base64_image]}
    )
    assert response.status_code == 400
    assert response.json()["detail"] == "Ungültiges Base64-Format."


def test_image_too_large():
    """
    Tests the "/uploads" endpoint with an oversized image.

    Assertions:
        - The HTTP status code must be 413.
        - The JSON response must contain the appropriate error message indicating size limit exceeded.
    """
    large_base64_image = "data:image/jpeg;base64," + "A" * (MAX_IMAGE_SIZE + 1)

    response = client.post(
        "/uploads", json={"images": [large_base64_image]}
    )
    assert response.status_code == 413
    assert response.json()["detail"] == "Das Bild überschreitet die maximale Dateigröße von 5 MB."


def test_classify_plant_empty_image_list():
    """
    Tests the "/uploads" endpoint with an empty image list.

    Assertions:
        - The HTTP status code must be 400.
        - The JSON response must contain the appropriate error message for missing images.
    """
    response = client.post(
        "/uploads", json={}
    )
    assert response.status_code == 400
    assert response.json()["detail"] == "Fehlende Bilddaten."
