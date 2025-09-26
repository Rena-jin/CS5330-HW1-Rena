import sys
import os
import numpy as np
import matplotlib.pyplot as plt

# Add src directory to path
sys.path.append('src')

from image_processor import ImagePreprocessor
from grid_analyzer import GridAnalyzer
from tile_mapper import TileMapper
from performance_metrics import PerformanceMetrics

def test_three_tile_styles():
    """
    Test and compare all three tile styles for assignment demonstration.
    """
    print("="*70)
    print("MOSAIC STYLE PERFORMANCE COMPARISON TEST")
    print("="*70)
    
    # Initialize components
    preprocessor = ImagePreprocessor(target_size=(512, 512))
    analyzer = GridAnalyzer(base_grid_size=32)
    mapper = TileMapper()
    metrics = PerformanceMetrics()
    
    # Test settings
    test_images = []
    if os.path.exists('examples'):
        for file in sorted(os.listdir('examples'))[:3]:  # Test first 3 images
            if file.lower().endswith(('.png', '.jpg', '.jpeg')):
                test_images.append(os.path.join('examples', file))
    
    if not test_images:
        print("No test images found in 'examples' folder!")
        return
    
    # Test parameters
    grid_size = 24
    complexity_threshold = 40
    use_quantization = True
    
    # Three tile styles to test
    tile_styles = [
        ("Classic Solid", "solid"),
        ("Smart Geometric", "pattern"), 
        ("Oil Painting", "oil_painting")
    ]
    
    overall_results = {}
    
    for img_path in test_images:
        print(f"\n{'='*50}")
        print(f"Testing Image: {os.path.basename(img_path)}")
        print(f"{'='*50}")
        
        # Step 1: Preprocess image
        processed_image = preprocessor.preprocess_image(
            img_path, 
            apply_quantization=use_quantization, 
            n_colors=8
        )
        
        # Step 2: Analyze grid (same for all styles)
        analyzer.base_grid_size = grid_size
        grid_info, color_analysis, dynamic_palette = analyzer.analyze_image_grid(
            processed_image, 
            complexity_threshold=complexity_threshold
        )
        
        print(f"Grid Analysis Results:")
        print(f"  Total cells: {len(grid_info)}")
        print(f"  Palette colors: {len(dynamic_palette)}")
        
        image_results = {}
        
        # Step 3: Test each tile style
        for style_name, style_code in tile_styles:
            print(f"\n--- Testing {style_name} ---")
            
            # Generate mosaic
            if hasattr(mapper, 'generate_mosaic_with_bounds'):
                image_bounds = preprocessor.get_image_bounds()
                mosaic = mapper.generate_mosaic_with_bounds(
                    processed_image, grid_info, color_analysis, 
                    dynamic_palette, tile_style=style_code, image_bounds=image_bounds
                )
            else:
                mosaic = mapper.generate_mosaic(
                    processed_image, grid_info, color_analysis, 
                    dynamic_palette, tile_style=style_code
                )
            
            # Evaluate performance
            performance_results = metrics.evaluate_comprehensive_quality(processed_image, mosaic)
            
            # Print results
            print(f"MSE: {performance_results['mse']:.2f}")
            print(f"SSIM: {performance_results['ssim']:.4f}")
            print(f"Color Fidelity: {performance_results['color_fidelity']:.3f}")
            print(f"Edge Preservation: {performance_results['edge_preservation']:.3f}")
            print(f"Overall Score: {performance_results['overall_score']:.1f}/100")
            
            # Get quality assessment
            assessment = metrics.get_quality_assessment(performance_results)
            print(f"Assessment: {assessment.split('(')[0].strip()}")
            
            image_results[style_name] = performance_results
        
        overall_results[os.path.basename(img_path)] = image_results
        
        # Create visual comparison for this image
        create_visual_comparison(
            processed_image, 
            grid_info, 
            color_analysis, 
            dynamic_palette, 
            mapper, 
            preprocessor,
            os.path.basename(img_path)
        )
    
    # Create summary report
    create_summary_report(overall_results)
    
    print(f"\n{'='*70}")
    print("PERFORMANCE COMPARISON COMPLETE")
    print("Check generated images and summary report!")
    print(f"{'='*70}")

def create_visual_comparison(processed_image, grid_info, color_analysis, 
                           dynamic_palette, mapper, preprocessor, image_name):
    """
    Create side-by-side visual comparison of all three styles.
    """
    tile_styles = [
        ("Original", None),
        ("Classic Solid", "solid"),
        ("Smart Geometric", "pattern"), 
        ("Oil Painting", "oil_painting")
    ]
    
    fig, axes = plt.subplots(1, 4, figsize=(16, 4))
    fig.suptitle(f'Tile Style Comparison - {image_name}', fontsize=14, fontweight='bold')
    
    for i, (style_name, style_code) in enumerate(tile_styles):
        if style_code is None:
            # Show original
            axes[i].imshow(processed_image)
        else:
            # Generate mosaic
            if hasattr(mapper, 'generate_mosaic_with_bounds'):
                image_bounds = preprocessor.get_image_bounds()
                mosaic = mapper.generate_mosaic_with_bounds(
                    processed_image, grid_info, color_analysis, 
                    dynamic_palette, tile_style=style_code, image_bounds=image_bounds
                )
            else:
                mosaic = mapper.generate_mosaic(
                    processed_image, grid_info, color_analysis, 
                    dynamic_palette, tile_style=style_code
                )
            axes[i].imshow(mosaic)
        
        axes[i].set_title(style_name, fontweight='bold')
        axes[i].axis('off')
    
    plt.tight_layout()
    plt.savefig(f'comparison_{image_name.split(".")[0]}.png', dpi=150, bbox_inches='tight')
    plt.show()

def create_summary_report(results):
    """
    Create a summary report comparing all styles across all images.
    """
    print(f"\n{'='*70}")
    print("SUMMARY PERFORMANCE REPORT")
    print(f"{'='*70}")
    
    # Calculate average scores for each style
    style_averages = {}
    
    for image_name, image_results in results.items():
        for style_name, metrics in image_results.items():
            if style_name not in style_averages:
                style_averages[style_name] = {
                    'mse': [], 'ssim': [], 'color_fidelity': [], 
                    'edge_preservation': [], 'overall_score': []
                }
            
            style_averages[style_name]['mse'].append(metrics['mse'])
            style_averages[style_name]['ssim'].append(metrics['ssim'])
            style_averages[style_name]['color_fidelity'].append(metrics['color_fidelity'])
            style_averages[style_name]['edge_preservation'].append(metrics['edge_preservation'])
            style_averages[style_name]['overall_score'].append(metrics['overall_score'])
    
    # Print average results
    print(f"\nAVERAGE PERFORMANCE ACROSS ALL TEST IMAGES:")
    print(f"{'-'*70}")
    print(f"{'Style':<18} {'MSE':<8} {'SSIM':<8} {'Color':<8} {'Edge':<8} {'Overall':<8}")
    print(f"{'-'*70}")
    
    for style_name, metrics in style_averages.items():
        avg_mse = np.mean(metrics['mse'])
        avg_ssim = np.mean(metrics['ssim'])
        avg_color = np.mean(metrics['color_fidelity'])
        avg_edge = np.mean(metrics['edge_preservation'])
        avg_overall = np.mean(metrics['overall_score'])
        
        print(f"{style_name:<18} {avg_mse:<8.1f} {avg_ssim:<8.3f} {avg_color:<8.3f} {avg_edge:<8.3f} {avg_overall:<8.1f}")
    
    # Best performing style
    best_style = max(style_averages.keys(), 
                    key=lambda style: np.mean(style_averages[style]['overall_score']))
    
    print(f"\nBEST PERFORMING STYLE: {best_style}")
    print(f"Average Overall Score: {np.mean(style_averages[best_style]['overall_score']):.1f}/100")

if __name__ == "__main__":
    test_three_tile_styles()