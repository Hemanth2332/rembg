import gradio as gr
from src.inference_utils import load_model, process_image

# Initialize Model
model = load_model()

def gradio_wrapper(input_img):
    """Wrapper for process_image to be used in Gradio interface."""
    return process_image(model, input_img)

# Gradio Interface
with gr.Blocks(title="Background Removal UI") as demo:
    gr.Markdown("# 🎨 Background Removal with ResNet50-UNet")
    gr.Markdown("Upload an image to see the predicted mask and the resulting foreground.")
    
    with gr.Row():
        input_image = gr.Image(label="Input Image", height=320)
        with gr.Column():
            output_mask = gr.Image(label="Predicted Mask", image_mode="L", height=150)
            output_result = gr.Image(label="Background Removed (Overlapped)", height=150)
            
    submit_btn = gr.Button("Remove Background", variant="primary")
    submit_btn.click(
        fn=gradio_wrapper,
        inputs=input_image,
        outputs=[output_mask, output_result]
    )
    
    gr.Examples(
        examples=["test_imgs/lion.jpg"],
        inputs=input_image
    )

if __name__ == "__main__":
    demo.launch()
