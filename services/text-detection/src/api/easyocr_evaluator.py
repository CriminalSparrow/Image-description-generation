import easyocr
import numpy as np

class EasyOCREvaluator:
    def __init__(self, model_path='../models/easyocr_finetuned/'):
        """
        Initializes the EasyOCR reader with our fine-tuned recognition model.
        """
        print("Initializing EasyOCR with fine-tuned model...")
        # We use the standard detector but specify our own recognizer
        self.reader = easyocr.Reader(lang_list=['ru', 'en'], # Languages
            gpu=True,
            model_storage_directory=model_path,
            user_network_directory=model_path,
            recog_network='cyrillic_g2' # Tells EasyOCR to look for a custom model
        )

    def predict(self, image_path: str):
        """
        Takes an image path and returns structured predictions.
        """
        # EasyOCR's readtext performs both detection and recognition
        results = self.reader.readtext(image_path)
        
        predictions = []
        for (bbox, text, prob) in results:
            # bbox is [[x1,y1], [x2,y1], [x2,y2], [x1,y2]]
            # Convert it to the 8-point format our evaluator expects
            points = np.array(bbox).flatten().tolist()
            predictions.append({
                'text': text,
                'points': points
            })
        return predictions