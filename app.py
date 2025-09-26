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
    Enhanced mosaic generator app with three artistic tile styles.
    """
    
    def __init__(self):
        """Initialize all components."""
        self.preprocessor = ImagePreprocessor(target_size=(512, 512))
        self.analyzer = GridAnalyzer(base_grid_size=32)
        self.mapper = TileMapper()
    
    def generate_mosaic(self, image, grid_size, use_quantization, complexity_threshold, tile_style):
        """
        Generate mosaic with selected tile style.
        """
        try:
            if image is None:
                return None
                
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
            
            # Step 3: Tile Mapping with selected style
            tile_style_map = {
                "Classic Solid": "solid",
                "Smart Geometric": "pattern",
                "Oil Painting": "oil_painting"
            }
            
            selected_style = tile_style_map.get(tile_style, "solid")
            
            # Use the method that exists in your current tile_mapper
            if hasattr(self.mapper, 'generate_mosaic_with_bounds'):
                image_bounds = self.preprocessor.get_image_bounds()
                mosaic = self.mapper.generate_mosaic_with_bounds(
                    processed_image, 
                    grid_info, 
                    color_analysis, 
                    dynamic_palette,
                    tile_style=selected_style,
                    image_bounds=image_bounds
                )
            else:
                # Fallback to basic method
                mosaic = self.mapper.generate_mosaic(
                    processed_image, 
                    grid_info, 
                    color_analysis, 
                    dynamic_palette,
                    tile_style=selected_style
                )
            
            return mosaic
            
        except Exception as e:
            print(f"Error generating mosaic: {e}")
            import traceback
            traceback.print_exc()
            # Return error image
            error_image = np.full((512, 512, 3), [255, 50, 50], dtype=np.uint8)
            return error_image

def create_interface():
    """
    Create clean interface with three artistic tile style selection.
    """
    app = MosaicGeneratorApp()
    
    with gr.Blocks(
        title="Enhanced Interactive Image Mosaic Generator",
        theme=gr.themes.Soft(),
        css="""
        .gradio-container {
            max-width: 1300px !important;
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
        .upload-container .source-selection {
            display: none !important;
        }
        .tile-style-group {
            border: 2px solid #e5e7eb !important;
            border-radius: 8px !important;
            padding: 15px !important;
            margin: 10px 0 !important;
            background: linear-gradient(to bottom, #f8fafc, #f1f5f9) !important;
        }
        """
    ) as interface:
        
        # Clean Header
        gr.HTML("""
        <div style="text-align: center; padding: 20px; background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); color: white; border-radius: 10px; margin-bottom: 20px;">
            <h1 style="margin: 0; font-size: 2.5em;">Enhanced Interactive Image Mosaic Generator</h1>
            <h3 style="margin: 10px 0; opacity: 0.9;">CS5330 Computer Vision - Three Artistic Tile Styles</h3>
            <p style="margin: 5px 0; opacity: 0.8;"><strong>Features:</strong> Classic Solid • Smart Geometric • Oil Painting</p>
        </div>
        """)
        
        # Main Layout
        with gr.Row():
            # Left Column - Controls
            with gr.Column(scale=1, min_width=320):
                
                # Upload Area
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
                
                # Artistic Style Selection
                gr.HTML("<h3>Artistic Tile Styles</h3>")
                with gr.Group(elem_classes=["tile-style-group"]):
                    tile_style = gr.Radio(
                        choices=["Classic Solid", "Smart Geometric", "Oil Painting"],
                        value="Classic Solid",
                        label="Choose Your Artistic Style",
                        info="Each style uses different computer vision techniques"
                    )
                
                # Style Descriptions
                gr.HTML("""
                <div style="font-size: 0.85em; color: #555; line-height: 1.5; background: #f8f9fa; padding: 15px; border-radius: 8px; margin: 10px 0;">
                    <strong>Style Guide:</strong><br>
                    <strong>• Classic Solid:</strong> Traditional mosaic blocks with solid colors<br>
                    <strong>• Smart Geometric:</strong> Gradient-driven patterns using Sobel edge detection<br>
                    <strong>• Oil Painting:</strong> Textured brush strokes with artistic color variations
                </div>
                """)
                
                # Technical Settings
                gr.HTML("<h3>Technical Parameters</h3>")
                
                grid_size = gr.Slider(
                    minimum=16,
                    maximum=64,
                    value=16,  # Changed to smaller default for better quality
                    step=8,
                    label="Grid Size",
                    info="Smaller = more detail, Larger = more artistic"
                )
                
                complexity_threshold = gr.Slider(
                    minimum=20,
                    maximum=80,
                    value=50,  # Back to 50 as requested
                    step=5,
                    label="Complexity Threshold",
                    info="Controls subdivision in complex areas"
                )
                
                use_quantization = gr.Checkbox(
                    label="Color Quantization",
                    value=True,
                    info="Reduce colors for enhanced artistic effect"
                )
                
                # Generate Button
                generate_btn = gr.Button(
                    "Generate Artistic Mosaic",
                    variant="primary",
                    size="lg"
                )
            
            # Right Column - Results
            with gr.Column(scale=2, min_width=600):
                
                gr.HTML("<h3>Your Artistic Mosaic</h3>")
                
                # Main Mosaic Display
                mosaic_output = gr.Image(
                    label="",
                    height=500,
                    show_label=False,
                    show_download_button=True,
                    show_share_button=False
                )
        
        # Enhanced Instructions
        gr.HTML("""
        <div style="margin-top: 30px; padding: 25px; background: linear-gradient(to right, #f8f9fa, #e9ecef); border-radius: 15px; border: 1px solid #dee2e6;">
            <h3 style="color: #495057; margin-bottom: 20px;">How to Create Artistic Mosaics</h3>
            
            <div style="display: grid; grid-template-columns: 1fr 1fr; gap: 25px; margin-bottom: 20px;">
                <div style="background: white; padding: 20px; border-radius: 10px; border: 1px solid #e9ecef;">
                    <h4 style="color: #6f42c1; margin-bottom: 15px;">Quick Start Guide</h4>
                    <ol style="margin: 0; padding-left: 20px; line-height: 1.6;">
                        <li><strong>Upload</strong> your favorite image</li>
                        <li><strong>Choose</strong> an artistic tile style</li>
                        <li><strong>Adjust</strong> parameters to taste</li>
                        <li><strong>Generate</strong> your unique mosaic</li>
                        <li><strong>Download</strong> and share your art!</li>
                    </ol>
                </div>
                
                <div style="background: white; padding: 20px; border-radius: 10px; border: 1px solid #e9ecef;">
                    <h4 style="color: #e83e8c; margin-bottom: 15px;">Artistic Features</h4>
                    <ul style="margin: 0; padding-left: 20px; line-height: 1.6;">
                        <li><strong>Three unique styles</strong> with distinct visual effects</li>
                        <li><strong>Smart adaptation</strong> based on image content</li>
                        <li><strong>Advanced algorithms</strong> using computer vision techniques</li>
                        <li><strong>Professional quality</strong> downloadable results</li>
                        <li><strong>Boundary detection</strong> for clean image edges</li>
                    </ul>
                </div>
            </div>
            
            <div style="background: #e7f3ff; padding: 15px; border-radius: 8px; border-left: 4px solid #0066cc;">
                <strong>Note:</strong>
                <span style="margin-left: 10px;">Upload your own images or try the examples below to create unique artistic mosaics with three different visual styles.</span>
            </div>
        </div>
        """)
        
        # Examples Section
        gr.HTML("<h3>Try These Example Images</h3>")
        
        # Create examples with better settings for quality
        example_images = []
        if os.path.exists('examples'):
            for file in sorted(os.listdir('examples')):
                if file.lower().endswith(('.png', '.jpg', '.jpeg')):
                    example_images.append([
                        os.path.join('examples', file),
                        16,   # Smaller grid for better detail
                        50,   # Back to 50 threshold as requested
                        True, # Color Quantization enabled
                        "Classic Solid"  # Default style
                    ])
        
        if example_images:
            gr.Examples(
                examples=example_images,
                inputs=[image_input, grid_size, complexity_threshold, use_quantization, tile_style],
                label="Click any example to load with optimized settings"
            )
        
        # Bind Generate Function
        generate_btn.click(
            fn=app.generate_mosaic,
            inputs=[image_input, grid_size, use_quantization, complexity_threshold, tile_style],
            outputs=[mosaic_output]
        )
    
    return interface

if __name__ == "__main__":
    demo = create_interface()
    demo.launch(share=True)
