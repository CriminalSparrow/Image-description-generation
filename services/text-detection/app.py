# services/text-detection/app.py
from fastapi import FastAPI, UploadFile
import cv2
import numpy as np

app = FastAPI()

@app.post("/detect")
async def detect(file: UploadFile):
    # Самостоятельная предобработка под конкретную модель
    image_bytes = await file.read()
    img = cv2.imdecode(np.frombuffer(image_bytes, np.uint8), cv2.IMREAD_COLOR)
    img = cv2.resize(img, (640, 640))  # Ресайз под YOLOv8
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)  # Конвертация цветов
    
    # ... логика модели ...
    return {"text": "распознанный текст"}