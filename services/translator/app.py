from fastapi import FastAPI, Request, HTTPException
import pickle
import pandas as pd
from enum import Enum
from pydantic import BaseModel
import requests
import os
from dotenv import load_dotenv
from typing import Optional

load_dotenv()
app = FastAPI()

all_LLM_Modles = [
    "google/gemini-2.0-flash-exp:free",
    "deepseek/deepseek-chat-v3-0324:free",    
    "qwen/qwen3-235b-a22b:free",
    "meta-llama/llama-4-maverick:free"
]

def translate_text(
    text: str,
    orig_lang: str = 'английский',
    target_lang: str = 'русский',
    llm_model: str = "google/gemini-2.0-flash-exp:free",
    llm_key: Optional[str] = None
    ) -> tuple[str, bool]:
    """
    Переводит текст между языками через OpenRouter API (используя `requests`).

    Параметры:
        text (str): Текст для перевода.
        orig_lang (Language): Исходный язык.
        target_lang (Language): Целевой язык.
        llm_model (str): Модель LLM (по умолчанию: deepseek-v3).
        llm_key (str, optional): API-ключ OpenRouter. Если None, берётся из переменной окружения.

    Возвращает:
        str: Переведённый текст или сообщение об ошибке.
        bool: False - ошибка, True - перевод сделан. 
    """
    
    if not llm_key:
        llm_key = os.getenv("OPENROUTER_API_KEY")
    if not llm_key:
        return "Ошибка: Не указан API-ключ OpenRouter.", False

    url = "https://openrouter.ai/api/v1/chat/completions"
    
    headers = {
        "Authorization": f"Bearer {llm_key}"
    }

    translation_prompt = f"""
        Strictly translate this {orig_lang} text to {target_lang}.
        Rules:
        1. Output ONLY the translation, NOTHING else
        2. Preserve original meaning 100%
        3. Keep names/terms unchanged
        4. For slang/idioms use closest natural equivalent
        5. Maintain original formatting (quotes, line breaks etc.)
        
        Text to translate:
        {text}
    """

    data = {
        "model": llm_model,
        "messages": [{"role": "user", "content": translation_prompt}],
        "temperature": 0.1,
    }

    try:
        response = requests.post(url, headers=headers, json=data, timeout=30)
        response.raise_for_status() 

        result = response.json()
        translated_text = result["choices"][0]["message"]["content"].strip()

        return str(translated_text), True

    except requests.exceptions.RequestException as e:
        return f"Ошибка сети: {str(e)}", False
    
    except (KeyError, IndexError) as e:
        return f"Ошибка обработки ответа API: {str(e)}", False


@app.get("/health")
def health():
    return {"status": "OK"}

@app.get("/models")
def get_models():
    return {"models list": all_LLM_Modles}

@app.post("/translate")
def translate(request: dict):

    llm_api_key = os.getenv("OPENROUTER_API_KEY")

    text_orig = request['text']
    lang_orig = request['lang_orig']
    lang_target = request['lang_target']

    if (lang_orig == lang_target):
        return { "translated_text": text_orig }
    
    for llm_model in all_LLM_Modles:
        text_target, status = translate_text(text_orig, lang_orig, lang_target, llm_model, llm_api_key)

        print(text_target)

        if status:
            return { "translated_text": text_target }
        
    return { "translated_text": "None of APIs worked"}

if __name__ == '__main__':
    import uvicorn
    #uvicorn.run(app, host="0.0.0.0", port=5000) # docker compose 
    uvicorn.run(app, host="127.0.0.1", port=5000) # test in python