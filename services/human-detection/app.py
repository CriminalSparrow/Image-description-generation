# services/human-detection/app.py
from fastapi import FastAPI, UploadFile, HTTPException
import cv2
import numpy as np
from deepface import DeepFace  # Пример для анализа людей

app = FastAPI()

@app.post("/process")
async def detect_humans(file: UploadFile):
    try:
        # Предобработка под DeepFace (512x512, RGB)
        image_bytes = await file.read()
        img = cv2.imdecode(np.frombuffer(image_bytes, np.uint8), cv2.IMREAD_COLOR)
        img = cv2.resize(img, (512, 512))
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        
        # Анализ людей
        analysis = DeepFace.analyze(img, actions=["gender", "age"])
        return {
            "count": len(analysis),
            "details": [{"gender": r["gender"], "age": r["age"]} for r in analysis]
        }
    except Exception as e:
        raise HTTPException(500, detail=str(e))