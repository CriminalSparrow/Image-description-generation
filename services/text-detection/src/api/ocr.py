# app.py

import gradio as gr
from ocr_pipeline import OCRPipeline

# --- 1. Initialize our OCR Pipeline Wrapper ---
# This single object will manage all our models.
pipeline = OCRPipeline()

# --- 2. Define the main function for Gradio ---
# This function takes inputs from the UI and returns outputs to the UI.
def ocr_demo(input_image, model_choice):
    """
    This function is the bridge between the Gradio UI and our pipeline.
    """
    if input_image is None:
        return None, "Please upload an image first."
    
    # Call our pipeline's predict method
    annotated_image, extracted_text = pipeline.predict(input_image, model_choice)
    
    return annotated_image, extracted_text

# --- 3. Build the Gradio Interface ---
# We define the UI components and layout here.
with gr.Blocks(title="OCR Project Demo", theme=gr.themes.Soft()) as demo:
    gr.Markdown("# OCR Pipeline: VLM vs. Fine-Tuned Model")
    gr.Markdown("Upload an image and select a model to perform Optical Character Recognition.")

    with gr.Row():
        with gr.Column():
            # Input Components
            input_image = gr.Image(type="pil", label="Upload Image")
            model_choice = gr.Dropdown(
                choices=["Qwen-VL (VLM)", "EasyOCR (Fine-Tuned)"],
                label="Choose Model",
                value="EasyOCR (Fine-Tuned)" # Default choice
            )
            submit_btn = gr.Button("Recognize Text", variant="primary")

        with gr.Column():
            # Output Components
            output_image = gr.Image(type="pil", label="Annotated Image")
            extracted_text = gr.Textbox(label="Extracted Text", lines=10)

    # --- 4. Connect the UI to the function ---
    # This tells Gradio what to do when the button is clicked.
    submit_btn.click(
        fn=ocr_demo,
        inputs=[input_image, model_choice],
        outputs=[output_image, extracted_text]
    )
    
    # gr.Markdown("## Example Images")
    # gr.Examples(
    #     examples=["/kaggle/input/your-dataset/images/example1.jpg", "/kaggle/input/your-dataset/images/example2.jpg"], # <-- CHANGE THESE PATHS
    #     inputs=input_image
    # )

# --- 5. Launch the App ---
if __name__ == "__main__":
    # share=True creates a public link, useful for sharing or embedding in presentations
    demo.launch(share=True)