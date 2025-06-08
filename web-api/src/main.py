from fastapi import FastAPI, UploadFile, HTTPException  
import httpx  
import cv2  
import numpy as np  

app = FastAPI()  

# Конфиг сервисов  
SERVICES = {  
    "caption": "http://image-captioning:5000/predict",  
    "text": "http://text-detection:5000/detect",  
    "ui": "http://ui-detection:5000/analyze",  
    "human": "http://human-detection:5000/process"  
}  

@app.post("/analyze")
async def analyze_image(file: UploadFile):
    try:
        image_bytes = await file.read()  # Получаем сырые байты

        # Отправка исходного изображения во все сервисы
        async with httpx.AsyncClient() as client:
            tasks = {
                name: client.post(url, files={"image": image_bytes})
                for name, url in SERVICES.items()
            }
            responses = await asyncio.gather(*tasks.values())

        return {
            name: response.json()
            for name, response in zip(SERVICES.keys(), responses)
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))