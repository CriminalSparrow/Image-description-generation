# services/image-captioning/app.py
from fastapi import FastAPI, UploadFile, HTTPException
import numpy as np
import cv2
from transformers import pipeline

app = FastAPI()
model = pipeline("image-to-text", model="nlpconnect/vit-gpt2-image-captioning")

@app.post("/predict")
async def predict(file: UploadFile):
    try:
        image_bytes = await file.read()
        img = cv2.imdecode(np.frombuffer(image_bytes, np.uint8), cv2.IMREAD_COLOR)
        img = cv2.resize(img, (224, 224))
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        
        # Генерация описания
        caption = model(img)
        return {"caption": caption[0]["generated_text"]}
    except Exception as e:
        raise HTTPException(500, detail=str(e))