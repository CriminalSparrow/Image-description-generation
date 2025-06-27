from fastapi import FastAPI, UploadFile, File, Query
from fastapi.responses import JSONResponse
from PIL import Image
import torch
import httpx
from transformers import AutoProcessor, AutoModelForCausalLM

app = FastAPI(title="Florence 2 Image Captioning API")

# === Загрузка модели и процессора ===
MODEL_ID = "microsoft/Florence-2-large"
model = None
processor = None
device = "cuda:0" if torch.cuda.is_available() else "cpu"
SERVICE_TOKEN_IDS = {0, 2}  # <pad> и </s>


@app.on_event("startup")
async def load_model():
    global model, processor
    print(f"Torch version: {torch.__version__}")
    print(f"CUDA available: {torch.cuda.is_available()}")

    print("Loading Florence 2 model...")
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_ID,
        trust_remote_code=True,
        torch_dtype="auto"
    ).to(device).eval()

    print(f"Model loaded on {next(model.parameters()).device}")

    processor = AutoProcessor.from_pretrained(MODEL_ID, trust_remote_code=True)
    print("Processor loaded")


def prepare_image(image):
    """Универсальная конвертация изображения"""
    if isinstance(image, str):
        img = Image.open(image).convert("RGB")
    elif isinstance(image, UploadFile):
        img = Image.open(image.file).convert("RGB")
    else:
        img = image.convert("RGB")
    
    # Удаление EXIF-метаданных для JPG
    img.info = {}
    return img

def generate_caption(image, prompt='<DETAILED_CAPTION>', num_beams=1):
    try:

        image = prepare_image(image)
        
        # Нормализация промпта
        prompt = prompt.upper().strip()
        if prompt not in ['<CAPTION>', '<DETAILED_CAPTION>', '<MORE_DETAILED_CAPTION>']:
            prompt = '<DETAILED_CAPTION>'
        
        inputs = processor(
            text=prompt,
            images=image,
            return_tensors="pt"
        ).to(device)

        if 'pixel_values' in inputs:
            inputs['pixel_values'] = inputs['pixel_values'].to(dtype=next(model.parameters()).dtype)


        print(f"Image format: {image.format}, mode: {image.mode}")
        print(f"Input prompt: {prompt}")
        print(f"Input_ids: {inputs['input_ids']}")

        with torch.no_grad():
            outputs = model.generate(
                pixel_values=inputs['pixel_values'].to(dtype=next(model.parameters()).dtype),
                input_ids=inputs['input_ids'],
                max_new_tokens=300,
                num_beams=num_beams,
                output_scores=True,
                return_dict_in_generate=True
            )

        # Обработка вероятностей
        logits = torch.stack(outputs.scores, dim=1)
        generated_tokens = outputs.sequences[:, 1:]

        probs = torch.softmax(logits, dim=-1)
        selected_probs = probs.gather(-1, generated_tokens.unsqueeze(-1)).squeeze(-1)

        # Маска для неслужебных токенов
        mask = ~torch.isin(generated_tokens, torch.tensor(list(SERVICE_TOKEN_IDS), device=generated_tokens.device))
        filtered_probs = selected_probs[mask]

        mean_confidence = filtered_probs.mean().item() if len(filtered_probs) > 0 else 0.0

        # Декодируем текст
        caption = processor.batch_decode(outputs.sequences, skip_special_tokens=True)[0]

        return caption, mean_confidence
    
    except Exception as e:
        print(f"Generation error: {str(e)}")
        return "", 0.0

async def translate_text(caption: str, target_language: str = "русский"):
    """Асинхронный запрос ко второй API для перевода"""
    url = "http://translation_api:8001/translate" # Порт и endpoint второй API
    payload = {
        "text": caption,
        "lang_orig": "английский",
        "lang_target": target_language
    }

    async with httpx.AsyncClient(timeout=20.0) as client:
        response = await client.post(url, json=payload)
        response.raise_for_status()
        return response.json()["translated_text"]


@app.post("/caption")
async def describe_image(
    file: UploadFile = File(...),
    prompt: str = Query(default="<Caption>", description="Prompt type: <CAPTION>, <DETAILED_CAPTION> или <MORE_DETAILED_CAPTION>"),
    num_beams: int = Query(default=1, description="Number of beams for beam search")
):
    try:
        image = Image.open(file.file).convert("RGB")

        caption, confidence = generate_caption(image, prompt=prompt, num_beams=num_beams)

        translated_caption = await translate_text(caption)

        # Для неуверенных предсказаний добавим варнинг/заменим ответ модели на сообщение о неудаче
        if confidence < 0.64:
            if prompt == '<CAPTION>':
                translated_caption = 'Модель не уверена в своём ответе. Вероятность ошибки высока.\n' + translated_caption
            else:
                translated_caption = 'Распознать содержимое изображения не удалось'

        return {
            "caption": translated_caption,
            "mean_confidence": confidence
        }

    except Exception as e:
        return JSONResponse(status_code=500, content={"error": str(e)})
