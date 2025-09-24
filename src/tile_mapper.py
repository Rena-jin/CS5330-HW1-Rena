import numpy as np

class TileMapper:
    """
    Enhanced tile mapping module for the mosaic generator.
    Creates simple colored tiles based on classified colors from grid analysis with debugging support.
    """
    
    def __init__(self):
        """Initialize the tile mapper."""
        self.debug_mode = False
        
    def set_debug_mode(self, debug=True):
        """Enable or disable debug mode for troubleshooting."""
        self.debug_mode = debug
    
    def create_colored_tile(self, color, size):
        """
        Create a simple colored square tile.
        
        Args:
            color (tuple): RGB color values
            size (tuple): Tile size (width, height)
            
        Returns:
            numpy.ndarray: Solid color tile
        """
        # Ensure color values are valid
        color = np.clip(color, 0, 255).astype(np.uint8)
        tile = np.full((size[1], size[0], 3), color, dtype=np.uint8)
        return tile
    
    def generate_mosaic(self, original_image, grid_info, color_analysis, dynamic_palette=None):
        """
        Generate the complete mosaic by replacing each grid cell with a colored tile.
        Enhanced with debugging information to help identify color classification issues.
        
        Args:
            original_image (numpy.ndarray): Original processed image
            grid_info (list): List of grid cell coordinates (y1, y2, x1, x2)
            color_analysis (list): Color analysis results for each cell
            dynamic_palette (dict): Dynamic color palette (optional, for reference)
            
        Returns:
            numpy.ndarray: Complete mosaic image
        """
        # Create output mosaic with same dimensions as original
        mosaic = np.zeros_like(original_image)
        
        # Debug information
        if self.debug_mode:
            print(f"\nGenerating mosaic with {len(grid_info)} cells...")
            print(f"Using dynamic palette: {dynamic_palette is not None}")
            if dynamic_palette:
                print(f"Palette contains {len(dynamic_palette)} colors")
        
        # Keep track of color usage for debugging
        color_usage = {}
        problematic_cells = []
        
        # Replace each grid cell with corresponding colored tile
        for i, ((y1, y2, x1, x2), (dominant_color, category, category_color)) in enumerate(zip(grid_info, color_analysis)):
            # Calculate tile size for this cell
            tile_size = (x2 - x1, y2 - y1)
            
            # Extract original cell for comparison if debugging
            if self.debug_mode and i < 10:  # Debug first 10 cells
                original_cell = original_image[y1:y2, x1:x2]
                original_avg = np.mean(original_cell, axis=(0, 1)).astype(int)
                
                # Check if there's a significant color mismatch
                color_diff = np.linalg.norm(original_avg - np.array(category_color))
                
                print(f"Cell {i:2d}: Pos({x1:3d},{y1:3d})-({x2:3d},{y2:3d})")
                print(f"         Original avg RGB: {tuple(original_avg)}")
                print(f"         Dominant RGB:     {tuple(dominant_color)}")
                print(f"         Category: {category}")
                print(f"         Category RGB:     {category_color}")
                print(f"         Color difference: {color_diff:.1f}")
                
                if color_diff > 80:  # Significant difference threshold
                    problematic_cells.append({
                        'index': i,
                        'position': (x1, y1, x2, y2),
                        'original_avg': original_avg,
                        'dominant': dominant_color,
                        'category': category,
                        'category_color': category_color,
                        'difference': color_diff
                    })
                    print(f"         ⚠️  Large color difference detected!")
                print()
            
            # Track color usage
            if category in color_usage:
                color_usage[category] += 1
            else:
                color_usage[category] = 1
            
            # Create colored tile using the classified category color
            colored_tile = self.create_colored_tile(category_color, tile_size)
            
            # Place tile in mosaic at the grid cell position
            mosaic[y1:y2, x1:x2] = colored_tile
        
        # Debug summary
        if self.debug_mode:
            print(f"\nMosaic generation completed!")
            print(f"Color usage statistics:")
            sorted_colors = sorted(color_usage.items(), key=lambda x: x[1], reverse=True)
            for category, count in sorted_colors:
                print(f"  {category}: {count} cells")
            
            if problematic_cells:
                print(f"\n⚠️  Found {len(problematic_cells)} cells with significant color differences:")
                for cell in problematic_cells[:3]:  # Show first 3 problematic cells
                    print(f"  Cell {cell['index']}: {cell['category']} "
                          f"(diff: {cell['difference']:.1f})")
        
        return mosaic
    
    def generate_debug_comparison(self, original_image, grid_info, color_analysis):
        """
        Generate a side-by-side comparison showing original cells vs their dominant colors.
        Useful for debugging color classification issues.
        
        Args:
            original_image (numpy.ndarray): Original image
            grid_info (list): Grid cell coordinates
            color_analysis (list): Color analysis results
            
        Returns:
            numpy.ndarray: Debug comparison image
        """
        h, w = original_image.shape[:2]
        
        # Create two images: original with grid lines, and dominant colors
        original_with_grid = original_image.copy()
        dominant_color_image = np.zeros_like(original_image)
        
        for (y1, y2, x1, x2), (dominant_color, category, category_color) in zip(grid_info, color_analysis):
            # Draw grid lines on original
            original_with_grid[y1:y2, x1, :] = [255, 255, 255]  # Vertical lines
            original_with_grid[y1:y2, x2-1, :] = [255, 255, 255] 
            original_with_grid[y1, x1:x2, :] = [255, 255, 255]  # Horizontal lines
            original_with_grid[y2-1, x1:x2, :] = [255, 255, 255]
            
            # Fill dominant color image
            tile_size = (x2 - x1, y2 - y1)
            colored_tile = self.create_colored_tile(dominant_color, tile_size)
            dominant_color_image[y1:y2, x1:x2] = colored_tile
        
        # Create side-by-side comparison
        comparison = np.hstack([original_with_grid, dominant_color_image])
        return comparison
    
    def analyze_color_accuracy(self, original_image, grid_info, color_analysis):
        """
        Analyze the accuracy of color classification by comparing original vs classified colors.
        
        Args:
            original_image (numpy.ndarray): Original image
            grid_info (list): Grid cell coordinates
            color_analysis (list): Color analysis results
            
        Returns:
            dict: Analysis results including accuracy metrics
        """
        color_differences = []
        large_diff_cells = []
        
        for i, ((y1, y2, x1, x2), (dominant_color, category, category_color)) in enumerate(zip(grid_info, color_analysis)):
            # Get original cell
            original_cell = original_image[y1:y2, x1:x2]
            original_avg = np.mean(original_cell, axis=(0, 1))
            
            # Calculate differences
            dominant_diff = np.linalg.norm(original_avg - dominant_color)
            category_diff = np.linalg.norm(original_avg - np.array(category_color))
            
            color_differences.append({
                'cell_index': i,
                'dominant_color_diff': dominant_diff,
                'category_color_diff': category_diff,
                'original_avg': original_avg,
                'dominant_color': dominant_color,
                'category_color': category_color,
                'category': category
            })
            
            # Flag cells with large differences
            if category_diff > 80:
                large_diff_cells.append({
                    'index': i,
                    'position': (x1, y1, x2, y2),
                    'category': category,
                    'difference': category_diff,
                    'original_rgb': tuple(original_avg.astype(int)),
                    'category_rgb': category_color
                })
        
        # Calculate statistics
        dominant_diffs = [d['dominant_color_diff'] for d in color_differences]
        category_diffs = [d['category_color_diff'] for d in color_differences]
        
        analysis_results = {
            'total_cells': len(color_differences),
            'avg_dominant_diff': np.mean(dominant_diffs),
            'avg_category_diff': np.mean(category_diffs),
            'max_dominant_diff': np.max(dominant_diffs),
            'max_category_diff': np.max(category_diffs),
            'large_diff_cells_count': len(large_diff_cells),
            'large_diff_percentage': len(large_diff_cells) / len(color_differences) * 100,
            'large_diff_cells': large_diff_cells[:10],  # First 10 problematic cells
            'color_differences': color_differences
        }
        
        return analysis_results
    
    def print_color_accuracy_report(self, accuracy_analysis):
        """
        Print a detailed report of color classification accuracy.
        
        Args:
            accuracy_analysis (dict): Results from analyze_color_accuracy
        """
        print(f"\n" + "="*50)
        print(f"COLOR CLASSIFICATION ACCURACY REPORT")
        print(f"="*50)
        
        print(f"Total cells analyzed: {accuracy_analysis['total_cells']}")
        print(f"Average dominant color difference: {accuracy_analysis['avg_dominant_diff']:.1f}")
        print(f"Average category color difference: {accuracy_analysis['avg_category_diff']:.1f}")
        print(f"Maximum category color difference: {accuracy_analysis['max_category_diff']:.1f}")
        
        large_diff_count = accuracy_analysis['large_diff_cells_count']
        large_diff_pct = accuracy_analysis['large_diff_percentage']
        print(f"Cells with large color differences (>80): {large_diff_count} ({large_diff_pct:.1f}%)")
        
        if accuracy_analysis['large_diff_cells']:
            print(f"\nMost problematic cells:")
            for cell in accuracy_analysis['large_diff_cells'][:5]:
                print(f"  Cell {cell['index']:2d}: {cell['category']}")
                print(f"           Original RGB: {cell['original_rgb']}")
                print(f"           Category RGB: {cell['category_rgb']}")
                print(f"           Difference: {cell['difference']:.1f}")
        
        # Provide recommendations
        print(f"\nRECOMMENDATIONS:")
        if large_diff_pct > 20:
            print(f"  ⚠️  High number of misclassified cells detected!")
            print(f"     Consider increasing dynamic palette size or improving")
            print(f"     color classification algorithm.")
        elif large_diff_pct > 10:
            print(f"  ⚠️  Moderate color classification issues detected.")
            print(f"     Fine-tuning may be needed.")
        else:
            print(f"  ✅ Color classification appears to be working well.")
    
    def get_tile_set_preview(self, dynamic_palette, tile_size=(50, 50)):
        """
        Create a preview of the tile set (color palette) with labels.
        
        Args:
            dynamic_palette (dict): Color palette to show
            tile_size (tuple): Size of each preview tile
            
        Returns:
            numpy.ndarray: Tile set preview image
        """
        if not dynamic_palette:
            return np.zeros((tile_size[1], tile_size[0], 3), dtype=np.uint8)
        
        colors = list(dynamic_palette.values())
        color_names = list(dynamic_palette.keys())
        n_colors = len(colors)
        
        # Arrange tiles in a grid (4 columns max)
        cols = min(4, n_colors)
        rows = (n_colors + cols - 1) // cols
        
        preview_width = cols * tile_size[0]
        preview_height = rows * tile_size[1]
        preview = np.zeros((preview_height, preview_width, 3), dtype=np.uint8)
        
        # Create each colored tile
        for i, (color, name) in enumerate(zip(colors, color_names)):
            row = i // cols
            col = i % cols
            
            y1 = row * tile_size[1]
            y2 = y1 + tile_size[1]
            x1 = col * tile_size[0]
            x2 = x1 + tile_size[0]
            
            # Create and place colored tile
            tile = self.create_colored_tile(color, tile_size)
            preview[y1:y2, x1:x2] = tile
        
        return preview
    
    def get_mapping_statistics(self, grid_info, color_analysis):
        """
        Get enhanced statistics about the tile mapping process.
        
        Args:
            grid_info (list): Grid cell coordinates
            color_analysis (list): Color analysis results
            
        Returns:
            dict: Enhanced mapping statistics
        """
        # Count tiles by color category
        color_counts = {}
        tile_areas = []
        dominant_colors = []
        category_colors = []
        
        for (y1, y2, x1, x2), (dominant_color, category, category_color) in zip(grid_info, color_analysis):
            # Count color usage
            color_counts[category] = color_counts.get(category, 0) + 1
            
            # Calculate tile area
            area = (x2 - x1) * (y2 - y1)
            tile_areas.append(area)
            
            # Store colors for analysis
            dominant_colors.append(dominant_color)
            category_colors.append(category_color)
        
        # Calculate color diversity
        unique_dominant_colors = len(set([tuple(c) for c in dominant_colors]))
        unique_category_colors = len(set([tuple(c) for c in category_colors]))
        
        stats = {
            'total_tiles': len(grid_info),
            'unique_colors': len(color_counts),
            'unique_dominant_colors': unique_dominant_colors,
            'unique_category_colors': unique_category_colors,
            'color_distribution': color_counts,
            'avg_tile_area': np.mean(tile_areas) if tile_areas else 0,
            'min_tile_area': min(tile_areas) if tile_areas else 0,
            'max_tile_area': max(tile_areas) if tile_areas else 0,
            'total_area': sum(tile_areas),
            'color_compression_ratio': unique_dominant_colors / unique_category_colors if unique_category_colors > 0 else 1
        }
        
        return stats