import base64
import binascii
import os
import subprocess
import uuid
from io import BytesIO

from PIL import Image, UnidentifiedImageError
from fastapi import FastAPI, HTTPException

# Create image upload folder
UPLOAD_FOLDER = os.getenv("UPLOAD_FOLDER", "classify")
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

# Load maximum image size from .env file (default: 5 MB)
MAX_IMAGE_SIZE = int(os.getenv("MAX_IMAGE_SIZE", "5242880"))

app = FastAPI()


def decode_and_save_image(image_base64: str) -> str:
    """
    Decodes a Base64-encoded image and saves it as a JPG file.

    Args:
        image_base64 (str): Base64-encoded image string.

    Returns:
        str: The file path of the saved JPG image.

    Raises:
        HTTPException: If the image size exceeds the limit,
                        if the Base64 format is invalid,
                        or if the image format is unsupported.
    """
    if "," in image_base64:
        image_base64 = image_base64.split(",")[1]

    if len(image_base64) > MAX_IMAGE_SIZE:
        raise HTTPException(
            status_code=413, detail="Das Bild überschreitet die maximale Dateigröße von 5 MB.")

    try:
        image_data = base64.b64decode(image_base64)

    except binascii.Error:
        raise HTTPException(
            status_code=400, detail="Ungültiges Base64-Format.")

    try:
        image = Image.open(BytesIO(image_data))

        unique_filename = f"{uuid.uuid4().hex}.jpg"
        save_path = os.path.join(UPLOAD_FOLDER, unique_filename)
        image.convert("RGB").save(save_path, "JPEG")

        return save_path

    except UnidentifiedImageError:
        raise HTTPException(
            status_code=400, detail="Das Bildformat wird nicht unterstützt.")

    except Exception as e:
        raise HTTPException(
            status_code=500, detail=f"Unbekannter Fehler: {str(e)}")


def run_plant_classifier(image_path: str) -> str:
    """
    Executes the plant classifier script with the given image path.

    Args:
        image_path (str): Path to the image file to be classified.

    Returns:
        str: The classification result from the script.

    Raises:
        HTTPException: If the classification script encounters an error.
    """
    try:
        result = subprocess.run(
            ["python", "plant_classifier.py", image_path],
            capture_output=True, text=True, check=True
        )
        return result.stdout.strip()
    except subprocess.CalledProcessError as e:
        raise HTTPException(
            status_code=500, detail=f"Fehler im KI-Skript: {e}")
    finally:
        if os.path.exists(image_path):
            os.remove(image_path)


@app.post("/uploads")
async def classify_plant(image_data: dict):
    """
    Endpoint for classifying multiple images.

    Args:
        image_data (dict): JSON data containing a list of Base64-encoded images.

    Returns:
        dict: A dictionary with classification predictions for each uploaded image.

    Raises:
        HTTPException: If the 'images' field is missing in the request.
    """
    if "images" not in image_data:
        raise HTTPException(status_code=400, detail="Fehlende Bilddaten.")

    # predictions = []
    for image_base64 in image_data["images"]:
        # image_path = (
        decode_and_save_image(image_base64)  # )
        # prediction = run_plant_classifier(image_path)
        # predictions.append(prediction)

    results = []  # run_plant_data_api(predictions)

    return {"results": results}


@app.get("/")
async def hello_world():
    """
    A simple endpoint to verify that the API is running.

    Returns:
        dict: A message confirming the API is online.
    """
    return {"message": "Hello World 🌍"}


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="localhost", port=8000)
