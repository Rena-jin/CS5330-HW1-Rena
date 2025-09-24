import gradio as gr
import numpy as np
import sys
import os

# Add src directory to path
sys.path.append('src')

from image_processor import ImagePreprocessor
from grid_analyzer import GridAnalyzer
from tile_mapper import TileMapper

class MosaicGeneratorApp:
    """
    Simple mosaic generator app with clean interface.
    """
    
    def __init__(self):
        """Initialize all components."""
        self.preprocessor = ImagePreprocessor(target_size=(512, 512))
        self.analyzer = GridAnalyzer(base_grid_size=32)
        self.mapper = TileMapper()
    
    def generate_mosaic(self, image, grid_size, use_quantization, complexity_threshold):
        """
        Generate mosaic and return only the mosaic result (no concatenation).
        """
        try:
            # Step 1: Image Preprocessing
            processed_image = self.preprocessor.preprocess_image(
                image, 
                apply_quantization=use_quantization, 
                n_colors=8
            )
            
            # Step 2: Grid Analysis and Color Classification
            self.analyzer.base_grid_size = grid_size
            grid_info, color_analysis, dynamic_palette = self.analyzer.analyze_image_grid(
                processed_image, 
                complexity_threshold=complexity_threshold
            )
            
            # Step 3: Tile Mapping
            mosaic = self.mapper.generate_mosaic(
                processed_image, 
                grid_info, 
                color_analysis, 
                dynamic_palette
            )
            
            # Return just the mosaic (no white borders, no concatenation)
            return mosaic
            
        except Exception as e:
            # Return error message
            error_image = np.full((512, 512, 3), [255, 0, 0], dtype=np.uint8)  # Red error image
            return error_image

def create_interface():
    """
    Create simple, clean interface with large image display areas.
    """
    app = MosaicGeneratorApp()
    
    with gr.Blocks(
        title="Interactive Image Mosaic Generator",
        theme=gr.themes.Soft(),
        css="""
        .gradio-container {
            max-width: 1200px !important;
            margin: 0 auto !important;
        }
        .image-container {
            border: none !important;
            padding: 0 !important;
        }
        .image-container img {
            border: none !important;
            max-width: 100% !important;
            height: auto !important;
            object-fit: contain !important;
        }
        /* Fixed upload area styling */
        .upload-container {
            min-height: 400px !important;
            max-height: 400px !important;
            height: 400px !important;
            width: 100% !important;
            border: 2px dashed #d1d5db !important;
            border-radius: 12px !important;
            background: linear-gradient(to bottom, #f9fafb, #f3f4f6) !important;
            position: relative !important;
            display: flex !important;
            flex-direction: column !important;
            transition: all 0.3s ease !important;
        }
        .upload-container:hover {
            border-color: #6366f1 !important;
            background: linear-gradient(to bottom, #f0f9ff, #e0f2fe) !important;
        }
        .upload-container .wrap {
            height: 100% !important;
            width: 100% !important;
            display: flex !important;
            flex-direction: column !important;
            align-items: center !important;
            justify-content: center !important;
            padding: 15px 15px 70px 15px !important;
            box-sizing: border-box !important;
        }
        .upload-container img {
            max-width: 90% !important;
            max-height: 85% !important;
            object-fit: contain !important;
            border-radius: 8px !important;
            box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1) !important;
        }
        /* Position icons outside the main frame */
        .upload-container .source-selection {
            position: absolute !important;
            bottom: 0px !important;
            left: 0px !important;
            right: 0px !important;
            height: 60px !important;
            background: rgba(255, 255, 255, 0.9) !important;
            border-top: 1px solid #e5e7eb !important;
            border-radius: 0 0 12px 12px !important;
            display: flex !important;
            justify-content: center !important;
            align-items: center !important;
            gap: 20px !important;
            padding: 10px !important;
        }
        /* Ensure column layout stays fixed */
        .gradio-column {
            flex-direction: column !important;
        }
        """
    ) as interface:
        
        # Simple Header
        gr.HTML("""
        <div style="text-align: center; padding: 20px;">
            <h1>Interactive Image Mosaic Generator</h1>
            <h3>CS5330 Computer Vision - Homework 1</h3>
        </div>
        """)
        
        # Main Layout
        with gr.Row():
            # Left Column - Controls (Narrow)
            with gr.Column(scale=1, min_width=250):
                
                # Large Upload Area
                gr.HTML("<h3>Upload Image</h3>")
                image_input = gr.Image(
                    type="numpy",
                    label="",
                    height=400,
                    show_label=False,
                    show_download_button=False,
                    show_share_button=False,
                    container=True,
                    elem_classes=["upload-container"]
                )
                
                # Manual Settings
                gr.HTML("<h3>Manual Settings</h3>")
                
                grid_size = gr.Slider(
                    minimum=16,
                    maximum=64,
                    value=32,
                    step=8,
                    label="Grid Size"
                )
                
                complexity_threshold = gr.Slider(
                    minimum=20,
                    maximum=80,
                    value=50,
                    step=5,
                    label="Complexity Threshold"
                )
                
                use_quantization = gr.Checkbox(
                    label="Color Quantization",
                    value=True
                )
                
                # Generate Button
                generate_btn = gr.Button(
                    "Generate Mosaic",
                    variant="primary",
                    size="lg"
                )
            
            # Right Column - Large Display Area
            with gr.Column(scale=2, min_width=600):
                
                gr.HTML("<h3>Mosaic Result</h3>")
                
                # Large Mosaic Display Area
                mosaic_output = gr.Image(
                    label="",
                    height=500,
                    show_label=False,
                    show_download_button=True,
                    show_share_button=False
                )
        
        # Instructions Section (ADDED ONLY THIS)
        gr.HTML("""
        <div style="margin-top: 30px; padding: 20px; background-color: #f8f9fa; border-radius: 10px;">
            <h3>How to Upload Images:</h3>
            <ul>
                <li><strong>Click the upload area</strong> above and select an image file</li>
                <li><strong>Drag and drop</strong> an image directly into the upload box</li>
                <li><strong>Use example images</strong> from the examples section below</li>
                <li><strong>Supported formats:</strong> JPG, PNG, JPEG</li>
                <li><strong>Best results:</strong> Images with clear colors and good contrast</li>
            </ul>
            <h3>How to Download:</h3>
            <ul>
                <li><strong>After generating a mosaic</strong>, look for the download button in the top-right corner of the result image</li>
                <li><strong>Click the download icon</strong> to save your mosaic</li>
            </ul>
        </div>
        """)
        
        # Examples Section (Full Width)
        gr.HTML("<h3>Examples</h3>")
        
        # Create examples from your test images
        example_images = []
        if os.path.exists('examples'):
            for file in sorted(os.listdir('examples')):
                if file.lower().endswith(('.png', '.jpg', '.jpeg')):
                    example_images.append([
                        os.path.join('examples', file),
                        32,   # grid size
                        True, # quantization
                        50    # threshold
                    ])
        
        if example_images:
            gr.Examples(
                examples=example_images,
                inputs=[image_input, grid_size, use_quantization, complexity_threshold],
                label=""
            )
        
        # Bind Generate Function
        generate_btn.click(
            fn=app.generate_mosaic,
            inputs=[image_input, grid_size, use_quantization, complexity_threshold],
            outputs=[mosaic_output]
        )
    
    return interface

if __name__ == "__main__":
    demo = create_interface()
    demo.launch(
        share=False,
        server_port=7860,
        show_error=True,
        inbrowser=True
    )