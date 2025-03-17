import base64
import os
import uuid
from io import BytesIO
from pydantic import BaseModel, constr
from fastapi import FastAPI, HTTPException
from PIL import Image, UnidentifiedImageError
import binascii
import subprocess


# Bild-Upload-Ordner erstellen
UPLOAD_FOLDER = os.getenv("UPLOAD_FOLDER", "uploads")
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

# Maximale Bildgröße aus der .env-Datei laden (Standard: 5 MB)
MAX_IMAGE_SIZE = int(os.getenv("MAX_IMAGE_SIZE", "5242880"))

app = FastAPI()

# Bild-Upload-Ordner erstellen
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

# Base64-Dekodierung und JPG-Speicherung
def decode_and_save_image(image_base64: str) -> str:
    try:
        # Optionalen Header entfernen
        if "," in image_base64:
            image_base64 = image_base64.split(",")[1]

        # Base64 dekodieren und Bild öffnen
        image_data = base64.b64decode(image_base64)

        # Dateigrößen-Prüfung
        if len(image_data) > MAX_IMAGE_SIZE:
            raise HTTPException(status_code=413, detail="Das Bild überschreitet die maximale Dateigröße von 5 MB.")

        image = Image.open(BytesIO(image_data))

        # Bild in JPG konvertieren und speichern
        unique_filename = f"{uuid.uuid4().hex}.jpg"
        save_path = os.path.join(UPLOAD_FOLDER, unique_filename)
        image.convert("RGB").save(save_path, "JPEG")

        return save_path

    except (binascii.Error, ValueError):
        raise HTTPException(status_code=400, detail="Ungültiges Base64-Format.")
    except UnidentifiedImageError:
        raise HTTPException(status_code=400, detail="Das Bildformat wird nicht unterstützt.")

# Bild an KI-Skript weitergeben
def run_plant_classifier(image_path: str) -> str:
    try:
        result = subprocess.run(
            ["python", "plant_classifier.py", image_path],  # Aufruf des KI-Skripts
            capture_output=True, text=True, check=True
        )
        return result.stdout.strip()  # Ausgabe des Skripts als Antwort zurückgeben
    except subprocess.CalledProcessError as e:
        raise HTTPException(status_code=500, detail=f"Fehler im KI-Skript: {e}")
    finally:
        # Bild nach erfolgreicher oder fehlgeschlagener Verarbeitung löschen
        if os.path.exists(image_path):
            os.remove(image_path)

@app.post("/classify")
async def classify_plant(image_data: dict):
    try:
        image_path = decode_and_save_image(image_data["image_base64"])
        prediction = run_plant_classifier(image_path)
        return {"prediction": prediction}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/")
async def hello_world():
    return {"message": "Hello World 🌍"}

