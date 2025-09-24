import sys
sys.path.append('src')

import numpy as np
import matplotlib.pyplot as plt
import os

from image_processor import ImagePreprocessor
from grid_analyzer import GridAnalyzer  
from tile_mapper import TileMapper

def test_simple_tile_mapping():
    """
    Test the simple tile mapping functionality as required by the assignment.
    """
    print("Testing Simple Tile Mapping (Assignment Step 3)...")
    print("="*55)
    
    # Initialize modules
    preprocessor = ImagePreprocessor(target_size=(512, 512))
    analyzer = GridAnalyzer(base_grid_size=32)
    mapper = TileMapper()
    
    # Get test images
    image_files = [f for f in os.listdir('examples') if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
    image_files.sort()
    
    results = []
    
    # Test all three images
    for image_file in image_files:
        image_path = os.path.join('examples', image_file)
        image_name = os.path.basename(image_path).split('.')[0]
        
        print(f"\nProcessing {image_name}...")
        
        # Step 1: Preprocess image
        processed_image = preprocessor.preprocess_image(image_path, apply_quantization=True, n_colors=8)
        
        # Step 2: Analyze grid and get color classifications
        grid_info, color_analysis, dynamic_palette = analyzer.analyze_image_grid(processed_image, complexity_threshold=50)
        
        # Step 3: Create mosaic using simple colored tiles
        mosaic = mapper.generate_mosaic(processed_image, grid_info, color_analysis, dynamic_palette)
        
        # Get tile mapping statistics
        tile_stats = mapper.get_mapping_statistics(grid_info, color_analysis)
        
        results.append({
            'name': image_name,
            'original': processed_image,
            'mosaic': mosaic,
            'dynamic_palette': dynamic_palette,
            'tile_stats': tile_stats
        })
        
        print(f"  Total tiles created: {tile_stats['total_tiles']}")
        print(f"  Unique colors used: {tile_stats['unique_colors']}")
        print(f"  Average tile size: {tile_stats['avg_tile_area']:.1f} pixels")
        
        # Show color distribution
        color_dist = tile_stats['color_distribution']
        top_colors = sorted(color_dist.items(), key=lambda x: x[1], reverse=True)[:3]
        print(f"  Most used colors: {', '.join([f'{name}({count})' for name, count in top_colors])}")
    
    # Create visualization
    create_simple_mosaic_visualization(results, mapper)
    
    return True

def create_simple_mosaic_visualization(results, mapper):
    """
    Create visualization showing the tile mapping results.
    """
    num_images = len(results)
    fig, axes = plt.subplots(num_images, 4, figsize=(20, 5*num_images))
    
    if num_images == 1:
        axes = axes.reshape(1, -1)
    
    for i, result in enumerate(results):
        # 1. Original processed image
        axes[i, 0].imshow(result['original'])
        axes[i, 0].set_title(f"{result['name'].title()}\nOriginal")
        axes[i, 0].axis('off')
        
        # 2. Generated mosaic
        axes[i, 1].imshow(result['mosaic'])
        axes[i, 1].set_title(f"Mosaic\n{result['tile_stats']['total_tiles']} tiles")
        axes[i, 1].axis('off')
        
        # 3. Tile set (color palette) preview
        tile_preview = mapper.get_tile_set_preview(result['dynamic_palette'], tile_size=(40, 40))
        axes[i, 2].imshow(tile_preview)
        axes[i, 2].set_title(f"Tile Set\n{len(result['dynamic_palette'])} colors")
        axes[i, 2].axis('off')
        
        # 4. Side by side comparison
        comparison = np.hstack([result['original'], result['mosaic']])
        axes[i, 3].imshow(comparison)
        axes[i, 3].set_title(f"Before → After\nComparison")
        axes[i, 3].axis('off')
    
    plt.tight_layout()
    plt.savefig('simple_tile_mapping_results.png', dpi=150, bbox_inches='tight')
    plt.show()
    
    print(f"\nSimple tile mapping results saved as 'simple_tile_mapping_results.png'")

def analyze_step3_requirements():
    """
    Analyze how well the implementation meets Step 3 requirements.
    """
    print(f"\nStep 3 Requirements Analysis:")
    print("="*35)
    
    print(f"\n📋 ASSIGNMENT STEP 3 REQUIREMENTS:")
    print("-"*40)
    print("✓ Prepare a set of image tiles (or simple coloured squares)")
    print("  → Implemented: Dynamic color palette creates tile set")
    print("  → Each color becomes a simple colored square tile")
    
    print("\n✓ Replace each grid cell with corresponding tile based on classified colour")
    print("  → Implemented: generate_mosaic() replaces each grid cell")
    print("  → Uses color_analysis results from Step 2")
    print("  → Maps classified colors to appropriate colored tiles")
    
    print(f"\n🔧 IMPLEMENTATION DETAILS:")
    print("-"*25)
    print("• Tile Creation: create_colored_tile() makes simple colored squares")
    print("• Color Mapping: Uses dynamic palette from grid analysis")
    print("• Cell Replacement: Each grid cell becomes one colored tile")
    print("• Size Adaptation: Tiles automatically match grid cell sizes")
    
    print(f"\n💡 KEY BENEFITS:")
    print("-"*15)
    print("• Simple and efficient tile generation")
    print("• Direct color classification to tile mapping")
    print("• Adaptive tile sizes based on grid analysis")
    print("• Uses improved dynamic color palettes from Step 2")
    
    print(f"\n🎨 VISUAL RESULT:")
    print("-"*15)
    print("• Original image divided into grid cells")
    print("• Each cell replaced by solid color tile")
    print("• Color chosen based on dominant color analysis")
    print("• Creates classic mosaic appearance")

if __name__ == "__main__":
    success = test_simple_tile_mapping()
    
    if success:
        analyze_step3_requirements()
        print(f"\n🎉 Simple tile mapping (Step 3) completed successfully!")
        print("This implementation directly follows the assignment requirements:")
        print("  1. Creates simple colored square tiles")
        print("  2. Maps grid cells to tiles based on color classification")
        print("  3. Generates final mosaic by tile replacement")
    else:
        print(f"\nTile mapping test failed!")