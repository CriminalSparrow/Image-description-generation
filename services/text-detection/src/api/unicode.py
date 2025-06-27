# Place this in a utils file or at the top of src/ocr_pipeline.py

from PIL import Image, ImageDraw, ImageFont
import numpy as np
import cv2
import os

def find_font(font_name="NotoSans-Regular.ttf"):
    """
    Tries to find a standard TTF font on a Linux system.
    Falls back to a default if not found.
    """
    font_paths = [
        "./NotoSans-Regular.ttf"
    ]
    for path in font_paths:
        if os.path.exists(path):
            return path
    # If no standard font is found, tell Gradio to use its default
    # which often has better Unicode support than OpenCV.
    print(f"Warning: Could not find a standard TTF font. Text rendering may be limited.")
    return None

def draw_results_unicode(image: Image.Image, predictions: list, font_size=20):
    """
    Draws bounding boxes and Unicode text (like Cyrillic) on a PIL image.
    """
    # Create a copy to draw on
    img_to_draw_on = image.copy()
    draw = ImageDraw.Draw(img_to_draw_on)
    
    # --- KEY CHANGE: Load a TTF font that supports Cyrillic ---
    font_path = find_font()
    try:
        # If a font is found, use it.
        if font_path:
            font = ImageFont.truetype(font_path, font_size)
            # font = ImageFont.truetype(font_path)

        else:
            # Fallback to Pillow's default font if no system font is found
            font = ImageFont.load_default()
    except IOError:
        print("Error: Font file could not be loaded. Using default font.")
        font = ImageFont.load_default()

    for pred in predictions:
        # The points are for the polygon outline
        points = pred['points']
        text = pred['text']
        
        # Unpack points for drawing the polygon
        poly_points = [(points[i], points[i+1]) for i in range(0, len(points), 2)]
        
        # --- Draw the polygon outline ---
        draw.polygon(poly_points, outline="lime", width=3)
        
        # Determine the position for the text (above the top-left point)
        text_x = min(p[0] for p in poly_points)
        text_y = min(p[1] for p in poly_points) - (font_size + 2) # Position text above the box

        # --- Draw the text with a background rectangle for readability ---
        if text:
            # Get the size of the text to draw a background
            try:
                # Use getbbox for modern Pillow versions
                text_bbox = draw.textbbox((text_x, text_y), text, font=font)
            except AttributeError:
                # Fallback for older Pillow versions
                text_width, text_height = draw.textsize(text, font=font)
                text_bbox = (text_x, text_y, text_x + text_width, text_y + text_height)

            # Draw a black rectangle behind the text
            draw.rectangle(text_bbox, fill="black")
            # Draw the actual text in white on top of the rectangle
            draw.text((text_x, text_y), text, font=font, fill="red")
            
    return img_to_draw_on