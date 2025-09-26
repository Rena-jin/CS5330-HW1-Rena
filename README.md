# CS5330 Interactive Image Mosaic Generator

An interactive mosaic generator that transforms images into artistic mosaics using computer vision techniques.

## Features

- **Adaptive Grid System**: Non-fixed grid sizes (32×32 → 16×16 → 8×8)
- **Three Tile Styles**: Classic Solid, Smart Geometric, Oil Painting
- **Computer Vision Algorithms**: Canny edge detection, Sobel gradients, K-means clustering, LAB color space
- **Interactive Interface**: Gradio web interface with real-time generation
- **Performance Metrics**: MSE and SSIM evaluation

## Installation & Usage

### Prerequisites
```bash
pip install -r requirements.txt
```

### Running the Application
```bash
python app.py
```
Then open your browser to: `http://localhost:7860`

### Testing the System
```bash
# Test all three tile styles with performance metrics
python test_performance_metrics.py

# Test preprocessing functionality  
python test_image_preprocessing.py

# Test grid analysis algorithms
python test_grid_analysis.py

# Test tile generation methods
python test_tile_mapping.py
```

## File Structure

```
├── app.py                     # Main Gradio application
├── examples/                  # Test images (geometric.jpg, landscape.jpg, portrait.jpg)
├── src/                      # Core modules
│   ├── grid_analyzer.py      # Adaptive grid analysis with Canny/Sobel
│   ├── image_processor.py    # Image preprocessing and quantization
│   ├── performance_metrics.py # MSE/SSIM evaluation
│   └── tile_mapper.py       # Three tile generation algorithms
├── test_*.py                 # Testing modules
└── test results/            # Generated test images
```

## Usage Instructions

1. Upload an image (PNG, JPG, JPEG)
2. Choose tile style (Classic Solid, Smart Geometric, Oil Painting)
3. Adjust parameters:
   - Grid Size (16-64): Smaller = more detail
   - Complexity Threshold (20-80): Controls subdivision
   - Color Quantization: Simplifies colors
4. Click "Generate Artistic Mosaic"
5. Download your result

## Requirements Met

- Step 1: Image preprocessing with quantization
- Step 2: Non-fixed grid division using image processing techniques
- Step 3: Tile mapping with color classification
- Step 4: Interactive Gradio interface
- Step 5: MSE and SSIM performance metrics

## Author

Ruiling (Rena) Jin  
CS5330 Computer Vision - Fall 2025