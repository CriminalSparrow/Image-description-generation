import torch, gc
from transformers import AutoModelForCausalLM, AutoTokenizer
from transformers import Qwen2_5_VLForConditionalGeneration, AutoProcessor, BitsAndBytesConfig
from transformers.generation import GenerationConfig
import re


class QwenEvaluator:
    def __init__(self, model_name="Qwen/Qwen-VL-Chat"):
        self.model = Qwen2_5_VLForConditionalGeneration.from_pretrained(model_name, 
                torch_dtype=torch.float16, 
                # attn_implementation="flash_attention_2",
                device_map="auto")
        self.processor = AutoProcessor.from_pretrained(model_name)
    
    def _parse_response(self, response_text):
        """Parses the model's string output into structured data."""
        predictions = []
        pattern = r'\{\((\d+),(\d+),(\d+),(\d+)\)\}\s*--\s*(.*)'
        # Use re.findall to get a list of all matching groups
        matches = re.findall(pattern, response_text)
        
        for match in matches:
            try:
                # The first 4 elements are the coordinate strings
                coords = match[:4]
                # The 5th element is the text string
                text = match[4]
        
                # Convert coordinate strings to integers
                x1, y1, x2, y2 = map(int, coords)
                
                # Clean up the extracted text
                cleaned_text = text.strip().replace('"', '')
                
                # Convert to 4-point polygon format for your evaluator
                # (top-left, top-right, bottom-right, bottom-left)
                points = [x1, y1, x2, y1, x2, y2, x1, y2]
                
                predictions.append({'text': cleaned_text, 'points': points})
            except (ValueError, IndexError) as e:
                print(f"Skipping a malformed match: {match} due to error: {e}")
        return predictions
    def inference(self, image, image_path, prompt ='Read all the text in the image. For each section of text, print its bounding box and text in this bounding box in the format: json\n{(x1,y1),(x2,y2)} -- text',
                  sys_prompt="You are a helpful assistant.", max_new_tokens=1024, return_input=False):
        image_local_path = image_path
        img_width, img_height = image.size
        # prompt = 'Read all the text in the image. For each section of text, print its bounding box and text in this bounding box in the format: {(x1,y1),(x2,y2)} -- text'
        # prompt = "Read all the text in the image."
#         prompt ="You are a precise Optical Character Recognition (OCR) engine. Your only function is to extract text and bounding boxes from the image.\
#                 Follow these instructions exactly:\
#                 1. Identify every piece of distinct text in the image.\
#                 2. For each piece of text, determine its bounding box as top-left (x1, y1) and bottom-right (x2, y2) coordinates.\
#                 3. Format EACH finding on a SEPARATE line using this EXACT template: `{(x1,y1,x2,y2)} -- text`\
# \
#                 **CRITICAL RULES:** \
#                 - DO NOT use JSON or markdown code blocks. \
#                 - DO NOT add any introductory text, explanations, or summaries. \
#                 - Your entire response must ONLY consist of lines matching the specified template.\
#                 -Make sure that the coordinates that you write out do not go beyond the boundaries of the picture.\
# \
#                 **EXAMPLE OUTPUT FORMAT:**\
#                 {(143,89,406,135)} -- вконтакте\
#                 {(487,90,766,135)} -- Мемередиа\
#                 {(73,232,830,316)} -- МЕМЫ ГОДА"
        prompt = """Your task is to act as a literal OCR engine.
        - Identify every individual line of text.
        - For each line, provide its tightest possible bounding box.
        - Use the format: `{(x1,y1,x2,y2)} -- text`
        - Do not group multiple lines of text into a single box.
        - Do not add any other text.

        EXAMPLE:
        {(240,240,730,300)} -- Лечебный кабинет
        {(240,330,450,400)} -- № 4
        """
        messages = [
            {"role": "system", "content": sys_prompt},
            {"role": "user", "content": [
                    {"type": "text", "text": prompt},
                    {"image": image_local_path},
                ]
            },
        ]
        text = self.processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        print("text:", text)
        # image_inputs, video_inputs = process_vision_info([messages])
        inputs = self.processor(text=[text], images=[image], padding=False, return_tensors="pt")
        inputs = inputs.to('cuda')
        with torch.no_grad():
            output_ids =self.model.generate(**inputs, max_new_tokens=max_new_tokens)
        generated_ids = [output_ids[len(input_ids):] for input_ids, output_ids in zip(inputs.input_ids, output_ids)]
        
        response = self.processor.batch_decode(generated_ids, skip_special_tokens=True, clean_up_tokenization_spaces=True)
        print(img_width, img_height)
        print(response)
        return self._parse_response(response[0])
