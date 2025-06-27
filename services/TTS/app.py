from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import FileResponse
from pydantic import BaseModel
import edge_tts
import uuid
import asyncio
import os

app = FastAPI(title="Text to Speech API")

# Папка для временных файлов
OUTPUT_DIR = "./tts_audio"
os.makedirs(OUTPUT_DIR, exist_ok=True)

class TTSRequest(BaseModel):
    text: str
    lang: str = "ru-RU"  # По умолчанию русский
    voice: str = "ru-RU-DmitryNeural"  # Можно поменять голос

@app.post("/tts")
async def text_to_speech(request: TTSRequest):
    try:
        text = request.text
        voice = request.voice
        output_path = os.path.join(OUTPUT_DIR, f"{uuid.uuid4()}.mp3")

        communicate = edge_tts.Communicate(text=text, voice=voice)
        await communicate.save(output_path)

        return FileResponse(output_path, media_type="audio/mpeg", filename="speech.mp3")

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
