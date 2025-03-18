import base64
import binascii
import os
import subprocess
import uuid
from io import BytesIO

from PIL import Image, UnidentifiedImageError
from fastapi import FastAPI, HTTPException

# Bild-Upload-Ordner erstellen
UPLOAD_FOLDER = os.getenv("UPLOAD_FOLDER", "classify")
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

# Maximale Bildgröße aus der .env-Datei laden (Standard: 5 MB)
MAX_IMAGE_SIZE = int(os.getenv("MAX_IMAGE_SIZE", "5242880"))

app = FastAPI()


# Base64-Dekodierung und JPG-Speicherung
def decode_and_save_image(image_base64: str) -> str:
    # Optionalen Header entfernen
    if "," in image_base64:
        image_base64 = image_base64.split(",")[1]

    # Dateigrößen-Prüfung
    if len(image_base64) > MAX_IMAGE_SIZE:
        raise HTTPException(
            status_code=413, detail="Das Bild überschreitet die maximale Dateigröße von 5 MB.")

    # Überprüfen, ob das Base64-Format gültig ist
    try:
        image_data = base64.b64decode(image_base64)

    except binascii.Error:
        raise HTTPException(
            status_code=400, detail="Ungültiges Base64-Format.")

    try:
        image = Image.open(BytesIO(image_data))

        # Bild in JPG konvertieren und speichern
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


# Bild an KI-Skript weitergeben
def run_plant_classifier(image_path: str) -> str:
    try:
        result = subprocess.run(
            ["python", "plant_classifier.py", image_path],  # Aufruf des KI-Skripts
            capture_output=True, text=True, check=True
        )
        return result.stdout.strip()  # Ausgabe des Skripts als Antwort zurückgeben
    except subprocess.CalledProcessError as e:
        raise HTTPException(
            status_code=500, detail=f"Fehler im KI-Skript: {e}")
    finally:
        # Bild nach erfolgreicher oder fehlgeschlagener Verarbeitung löschen
        if os.path.exists(image_path):
            os.remove(image_path)


@app.post("/uploads")
async def classify_plant(image_data: dict):
    # image_path =
    decode_and_save_image(image_data["image_base64"])
    prediction = ""  # run_plant_classifier(image_path)
    return {"prediction": prediction}


@app.get("/")
async def hello_world():
    return {"message": "Hello World 🌍"}


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="localhost", port=8000)
