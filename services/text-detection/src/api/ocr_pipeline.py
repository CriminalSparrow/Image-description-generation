import os
from PIL import Image
import numpy as np
import cv2

# We will import the evaluator classes you've already built
from qwen_evaluator import QwenEvaluator
from easyocr_evaluator import EasyOCREvaluator
from unicode import draw_results_unicode
# A helper function to draw results on the image
def draw_results(image: Image.Image, predictions: list):
    """Draws bounding boxes and text on a PIL image."""
    img_cv = np.array(image.convert("RGB"))
    img_cv = cv2.cvtColor(img_cv, cv2.COLOR_RGB2BGR) # Convert to OpenCV format

    for pred in predictions:
        points = np.array(pred['points'], dtype=np.int32).reshape(-1, 1, 2)
        text = pred['text']
        cv2.polylines(img_cv, [points], isClosed=True, color=(0, 255, 0), thickness=2)
        # Put text above the top-left corner of the box
        cv2.putText(img_cv, text, (points[0][0][0], points[0][0][1] - 10), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)

    img_pil = Image.fromarray(cv2.cvtColor(img_cv, cv2.COLOR_BGR2RGB))
    return img_pil


class OCRPipeline:
    """
    A lightweight wrapper that hides all the complexity of model loading
    and provides a simple interface for performing OCR.
    """
    def __init__(self):
        # Lazy load models to avoid loading everything at once if not needed
        self.models = {
            "Qwen-VL (VLM)": None,
            "EasyOCR (Fine-Tuned)": None
        }
        self.active_model = None

    def _load_model(self, model_name: str):
        """Loads a model only when it's first requested."""
        if self.models.get(model_name) is None:
            print(f"Loading {model_name}...")
            if model_name == "Qwen-VL (VLM)":
                self.models[model_name] = QwenEvaluator('Qwen/Qwen2.5-VL-7B-Instruct')
            elif model_name == "EasyOCR (Fine-Tuned)":
                # Ensure the model path is correct
                self.models[model_name] = EasyOCREvaluator(model_path='models/easyocr_finetuned/')
            print(f"{model_name} loaded.")
        
        self.active_model = self.models[model_name]

    def predict(self, input_image: Image.Image, model_name: str):
        """
        The main public method. Takes a PIL image and a model name,
        and returns the annotated image and the structured text data.
        """
        # Load the selected model if it's not already loaded
        self._load_model(model_name)
        
        # We need to save the image temporarily because both evaluators
        # currently expect a file path.
        os.makedirs('./pictures/', exist_ok=True)
        temp_image_path = "./pictures/tmp.jpg"
        input_image.save(temp_image_path)

        # Run prediction using the active model's predict method
        if isinstance(self.active_model, QwenEvaluator):
            # Qwen needs a batch, so we wrap it in a list
            predictions = self.active_model.inference(input_image, temp_image_path)
        elif isinstance(self.active_model, EasyOCREvaluator):
            predictions = self.active_model.predict(temp_image_path)
        else:
            predictions = []
        
        # Clean up the temporary file
        os.remove(temp_image_path)

        # Draw the results on the image
        # output_image = draw_results(input_image, predictions)
        output_image = draw_results_unicode(input_image, predictions)
        
        # Format the text output nicely
        formatted_text = "\n".join([p['text'] for p in predictions])
        
        return output_image, formatted_text