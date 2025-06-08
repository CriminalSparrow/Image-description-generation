# services/ui-detection/app.py
from fastapi import FastAPI, UploadFile, HTTPException
import cv2
import numpy as np
import torch  # Для YOLO

app = FastAPI()
model = torch.hub.load("ultralytics/yolov5", "yolov5s")  # Пример

@app.post("/analyze")
async def detect_ui(file: UploadFile):
    try:
        # Предобработка под YOLO (640x640, нормализация)
        image_bytes = await file.read()
        img = cv2.imdecode(np.frombuffer(image_bytes, np.uint8), cv2.IMREAD_COLOR)
        img = cv2.resize(img, (640, 640))
        img = img / 255.0  # Нормализация для YOLO
        
        # Детекция
        results = model(img)
        ui_elements = ["button" if cls == 0 else "input" for cls in results.pred[0][:, -1]]
        return {"ui_elements": ui_elements}
    except Exception as e:
        raise HTTPException(500, detail=str(e))