from fastapi import FastAPI, UploadFile, File
from fastapi.responses import JSONResponse
from PIL import Image
import torch
from transformers import AutoTokenizer, AutoProcessor, AutoModelForImageTextToText

app = FastAPI(title="Image Captioning API")

# === Подгружаем модель и процессор ===
MODEL_PATH = "nanonets/Nanonets-OCR-s"
model = None
tokenizer = None
processor = None


@app.on_event("startup")
async def load_model():
    global model, tokenizer, processor
    print(f'{torch.__version__}')
    print(f'{torch.version.cuda}')
    print("Starting model load...")

    model = AutoModelForImageTextToText.from_pretrained(
        MODEL_PATH,
        torch_dtype="auto",
        device_map="cpu"
    ).eval()
    print(f"Model loaded on {next(model.parameters()).device}")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH)
    print("Tokenizer loaded")
    processor = AutoProcessor.from_pretrained(MODEL_PATH, use_fast=True)
    print("Processor loaded")


@app.post("/caption")
async def describe_image(file: UploadFile = File(...)):
    try:
        image = Image.open(file.file)
        prompt = """Write full and detailized image caption for blind people. Include overall mood. Caption must be in russian."""
        
        messages = [
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": [
                {"type": "image", "image": image},
                {"type": "text", "text": prompt},
            ]},
        ]
        
        text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        inputs = processor(text=[text], images=[image], padding=True, return_tensors="pt").to(model.device)
        
        with torch.no_grad():
            output_ids = model.generate(**inputs, max_new_tokens=300, do_sample=False)
        
        generated_ids = [output_ids[len(input_ids):] for input_ids, output_ids in zip(inputs.input_ids, output_ids)]
        description = processor.batch_decode(generated_ids, skip_special_tokens=True, clean_up_tokenization_spaces=True)[0]
        return {"description": description}

    except Exception as e:
        return JSONResponse(status_code=500, content={"error": str(e)})
