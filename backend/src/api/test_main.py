import base64
from pathlib import Path
from unittest.mock import patch

import pytest
from fastapi.testclient import TestClient

from plantapi.plant_api import PlantGetter
from .main import MAX_IMAGE_SIZE
from .main import app

client = TestClient(app)


@pytest.fixture()
def client_mock():
    """Fixture to create a test client for the FastAPI app."""
    client_mock = TestClient(app)
    yield client_mock


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
        raise FileNotFoundError(f"The image '{image_filename}' was not found in the folder.")

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


@patch.object(PlantGetter, 'get_plant_list_data')
def test_classify_plant_valid_image(mock_get_plant_list_data, client_mock):
    """
    Tests the "/uploads" endpoint with a valid Base64 image.

    Assertions:
        - The HTTP status code must be 200.
        - The JSON response must contain the key "results".
    """
    valid_base64_image = generate_base64_image()

    # Mocking the response from PlantGetter's get_plant_list_data method
    mock_get_plant_list_data.return_value = [
        {
            "name": "Rosa indica",
            "plant": {"scientific_name": "Rosa indica", "family": "Rosaceae"},
            "wikipedia": "https://de.wikipedia.org/wiki/Rosa_indica"
        }
    ]

    response = client.post(
        "/uploads", json={"images": [valid_base64_image]}
    )
    print(response.json())
    assert response.status_code == 200
    assert "results" in response.json()
    assert len(response.json()["results"]) == 1
    assert "recognitions" in response.json()["results"][0]


@patch.object(PlantGetter, 'get_plant_list_data')
def test_classify_plant_multiple_images(mock_get_plant_list_data, client_mock):
    """
    Tests the "/uploads" endpoint with multiple Base64 images.

    Assertions:
        - The HTTP status code must be 200.
        - The JSON response must contain predictions for each uploaded image.
    """
    valid_base64_image1 = generate_base64_image()
    valid_base64_image2 = generate_base64_image()

    # Mocking the response from PlantGetter's get_plant_list_data method
    mock_get_plant_list_data.side_effect = [
        [
            {
                "name": "Rosa indica",
                "plant": {"scientific_name": "Rosa indica", "family": "Rosaceae"},
                "wikipedia": "https://de.wikipedia.org/wiki/Rosa_indica"
            }
        ],
        [
            {
                "name": "Tulipa gesneriana",
                "plant": {"scientific_name": "Tulipa gesneriana", "family": "Liliaceae"},
                "wikipedia": "https://de.wikipedia.org/wiki/Tulipa_gesneriana"
            }
        ]
    ]

    response = client.post(
        "/uploads", json={"images": [valid_base64_image1, valid_base64_image2]}
    )
    print(response.json())
    assert response.status_code == 200
    assert "results" in response.json()
    assert len(response.json()["results"]) == 2
    assert len(response.json()["results"][0]["recognitions"]) == 5
    assert len(response.json()["results"][1]["recognitions"]) == 5


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
    assert response.json()["detail"] == "Invalid Base64 format."


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
    assert response.json()["detail"] == "The image exceeds the maximum file size of 5 MB."


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
    assert response.json()["detail"] == "Missing image data."
