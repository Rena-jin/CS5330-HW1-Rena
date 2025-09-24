import sys
sys.path.append('src')

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
import os

from image_processor import ImagePreprocessor
from grid_analyzer import GridAnalyzer

def test_all_example_images():
    """
    Test grid analysis with all three example images to evaluate algorithm performance.
    PURE STEP 2 TESTING - No mosaic generation, only grid analysis and color classification.
    """
    print("Testing Step 2: Grid Analysis and Color Classification...")
    print("="*65)
    
    # Initialize modules
    preprocessor = ImagePreprocessor(target_size=(512, 512))
    analyzer = GridAnalyzer(base_grid_size=32)
    
    # Get all image files
    image_files = []
    for file in os.listdir('examples'):
        if file.lower().endswith(('.png', '.jpg', '.jpeg')):
            image_files.append(os.path.join('examples', file))
    
    if len(image_files) < 3:
        print(f"Expected 3 images, found {len(image_files)}")
        return False
    
    # Sort to ensure consistent order
    image_files.sort()
    
    results = []
    
    # Test each image with same parameters
    complexity_threshold = 50
    
    for i, image_path in enumerate(image_files):
        image_name = os.path.basename(image_path).split('.')[0]
        print(f"\n{i+1}. Testing {image_name}...")
        
        try:
            # Step 1 result: Preprocess image
            processed_image = preprocessor.preprocess_image(
                image_path, apply_quantization=True, n_colors=8
            )
            
            # STEP 2 FUNCTIONALITY: Grid analysis and color classification
            grid_info, color_analysis, dynamic_palette = analyzer.analyze_image_grid(
                processed_image, complexity_threshold
            )
            
            # Get statistics
            stats = analyzer.get_grid_statistics(grid_info, color_analysis)
            
            # Store results for Step 2 analysis only
            result = {
                'name': image_name,
                'path': image_path,
                'processed_image': processed_image,
                'grid_info': grid_info,
                'color_analysis': color_analysis,
                'dynamic_palette': dynamic_palette,
                'stats': stats
            }
            results.append(result)
            
            # Print Step 2 analysis results
            print(f"   Image type detected: {analyzer._detect_image_type(processed_image)}")
            print(f"   Grid division: {stats['total_cells']} total cells")
            print(f"   Large cells: {stats['large_cells']} | Small cells: {stats['subdivided_cells']}")
            print(f"   Subdivision ratio: {stats['subdivision_ratio']:.1%}")
            print(f"   Dynamic palette generated: {len(dynamic_palette)} colors")
            print(f"   Color categories identified: {stats['unique_colors']}")
            print(f"   Average cell size: {stats['avg_cell_size']:.0f} pixels")
            
            # Show color classification results
            categories = [analysis[1] for analysis in color_analysis]
            unique_categories = list(set(categories))
            print(f"   Color categories: {', '.join(unique_categories[:5])}")
            
        except Exception as e:
            print(f"   Error processing {image_path}: {e}")
            continue
    
    if not results:
        print("No images processed successfully!")
        return False
    
    # Create Step 2 specific visualization (NO MOSAIC)
    create_step2_visualization(results)
    
    # Analyze Step 2 performance
    analyze_step2_performance(results)
    
    return True

def create_step2_visualization(results):
    """
    Create visualization for STEP 2 ONLY: Grid division, color analysis, and palette generation.
    NO MOSAIC GENERATION - that's Step 3. Fixed layout to prevent overflow.
    """
    num_images = len(results)
    fig, axes = plt.subplots(num_images, 4, figsize=(16, 4*num_images))  # Reduced figure size
    
    if num_images == 1:
        axes = axes.reshape(1, -1)
    
    for i, result in enumerate(results):
        # 1. Original processed image (from Step 1)
        axes[i, 0].imshow(result['processed_image'])
        axes[i, 0].set_title(f"{result['name'].title()}\nProcessed Image\n(Step 1 Result)")
        axes[i, 0].axis('off')
        
        # 2. Grid division visualization (Step 2 Part 1)
        axes[i, 1].imshow(result['processed_image'])
        
        # Draw grid lines to show division
        for y1, y2, x1, x2 in result['grid_info']:
            rect = Rectangle((x1, y1), x2-x1, y2-y1, 
                           linewidth=1, edgecolor='white', facecolor='none', alpha=0.8)
            axes[i, 1].add_patch(rect)
        
        stats = result['stats']
        axes[i, 1].set_title(f"Grid Division\n{stats['total_cells']} cells\n(Step 2: Grid Analysis)")
        axes[i, 1].axis('off')
        
        # 3. Dynamic color palette (Step 2 Part 2)
        palette_img = create_palette_preview(result['dynamic_palette'])
        axes[i, 2].imshow(palette_img)
        axes[i, 2].set_title(f"Dynamic Palette\n{len(result['dynamic_palette'])} colors\n(Step 2: Color Analysis)")
        axes[i, 2].axis('off')
        
        # 4. Color classification visualization (Step 2 Result) - NO CONCATENATION
        classification_viz = create_color_classification_visualization(result)
        axes[i, 3].imshow(classification_viz)
        axes[i, 3].set_title(f"Color Classification\n{stats['unique_colors']} categories\n(Step 2: Classification)")
        axes[i, 3].axis('off')
    
    plt.tight_layout()
    plt.subplots_adjust(bottom=0.05, top=0.95)  # Adjust margins to prevent overflow
    plt.savefig('step2_grid_analysis_results.png', dpi=150, bbox_inches='tight')
    plt.show()
    
    print(f"\nStep 2 analysis results saved as 'step2_grid_analysis_results.png'")

def create_palette_preview(palette, tile_size=30):
    """
    Create a visual preview of the dynamic color palette.
    """
    colors = list(palette.values())
    n_colors = len(colors)
    
    # Arrange colors in a grid
    cols = min(4, n_colors)
    rows = (n_colors + cols - 1) // cols
    
    preview = np.zeros((rows * tile_size, cols * tile_size, 3), dtype=np.uint8)
    
    for i, color in enumerate(colors):
        row = i // cols
        col = i % cols
        
        y1 = row * tile_size
        y2 = y1 + tile_size
        x1 = col * tile_size
        x2 = x1 + tile_size
        
        preview[y1:y2, x1:x2] = color
    
    return preview

def create_color_classification_visualization(result):
    """
    Create visualization showing color classification by filling cells completely.
    """
    processed_image = result['processed_image']
    grid_info = result['grid_info']
    color_analysis = result['color_analysis']
    
    # Create completely new image showing just the color categories
    h, w = processed_image.shape[:2]
    classification_image = np.zeros((h, w, 3), dtype=np.uint8)
    
    # Fill each cell completely with its category color
    for (y1, y2, x1, x2), (dominant_color, category, category_color) in zip(grid_info, color_analysis):
        # Ensure coordinates are safe
        y1, y2 = max(0, y1), min(h, y2)
        x1, x2 = max(0, x1), min(w, x2)
        
        # Fill entire cell with category color
        classification_image[y1:y2, x1:x2] = category_color
    
    return classification_image

def analyze_step2_performance(results):
    """
    Analyze Step 2 specific performance: grid division and color classification.
    """
    print(f"\nStep 2 Performance Analysis:")
    print("="*50)
    
    print(f"{'Image':<12} {'Cells':<6} {'Large%':<7} {'Small%':<7} {'Palette':<8} {'Categories':<11} {'Classification'}")
    print("-" * 75)
    
    for result in results:
        stats = result['stats']
        name = result['name']
        palette_size = len(result['dynamic_palette'])
        
        # Calculate Step 2 specific metrics
        total_cells = stats['total_cells']
        large_pct = (stats['large_cells'] / total_cells) * 100
        small_pct = stats['subdivision_ratio'] * 100
        categories = stats['unique_colors']
        
        # Step 2 assessment
        if small_pct > 70:
            classification = "Over-subdivided"
        elif small_pct < 20:
            classification = "Under-subdivided"  
        elif 30 <= small_pct <= 60:
            classification = "Well-balanced"
        else:
            classification = "Good"
        
        print(f"{name:<12} {total_cells:<6} {large_pct:<7.1f} {small_pct:<7.1f} {palette_size:<8} {categories:<11} {classification}")
    
    print(f"\nStep 2 Detailed Analysis:")
    print("-" * 30)
    
    for result in results:
        name = result['name']
        stats = result['stats']
        palette = result['dynamic_palette']
        
        print(f"\n{name.upper()} - Step 2 Results:")
        print(f"  Grid Division: {stats['total_cells']} cells ({stats['large_cells']} large, {stats['subdivided_cells']} subdivided)")
        print(f"  Color Analysis: {len(palette)} palette colors → {stats['unique_colors']} categories used")
        print(f"  Classification Quality: {stats['unique_colors']/len(palette):.1%} palette utilization")
        
        # Show intensity/color analysis results
        complexities = []
        dominant_colors = []
        for dominant_color, category, category_color in result['color_analysis']:
            complexities.append(np.mean(dominant_color))
            dominant_colors.append(dominant_color)
        
        if complexities:
            print(f"  Intensity Analysis: Avg={np.mean(complexities):.1f}, Range={min(complexities):.0f}-{max(complexities):.0f}")
        
        print(f"  Step 2 Status: COMPLETED - Ready for Step 3 tile mapping")

if __name__ == "__main__":
    success = test_all_example_images()
    
    if success:
        print(f"\nStep 2 (Grid Analysis & Color Classification) testing completed!")
        print("Results show:")
        print("  • Grid division with adaptive subdivision")
        print("  • Color intensity/value analysis for each cell") 
        print("  • Color classification into categories")
        print("  • Dynamic palette generation")
        print("\nReady for Step 3: Tile Mapping!")
    else:
        print(f"\nStep 2 testing failed!")