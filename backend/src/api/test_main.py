import base64
from pathlib import Path

from fastapi.testclient import TestClient

from .main import MAX_IMAGE_SIZE
from .main import app

client = TestClient(app)


def generate_base64_image(image_filename="test_img.jpg"):
    """
    Reads an existing image from the same folder as this script,
    encodes it in Base64 format, and returns it as a string.

    Args:
        image_filename (str): Name of the image file to encode.

    Returns:
        str: Base64-encoded string of the image in JPEG format.
    """
    script_dir = Path(__file__).resolve().parent
    image_path = script_dir / image_filename

    if not image_path.exists():
        raise FileNotFoundError(f"Das Bild '{image_filename}' wurde nicht im Ordner gefunden.")

    with open(image_path, "rb") as image_file:
        base64_image = base64.b64encode(image_file.read()).decode("utf-8")

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
    assert "results" in response.json()


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
