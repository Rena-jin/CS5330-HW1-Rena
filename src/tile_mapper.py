import numpy as np

class TileMapper:
    """
    Simple tile mapping module for the mosaic generator.
    Creates simple colored tiles based on classified colors from grid analysis.
    """
    
    def __init__(self):
        """Initialize the tile mapper."""
        pass
    
    def create_colored_tile(self, color, size):
        """
        Create a simple colored square tile.
        
        Args:
            color (tuple): RGB color values
            size (tuple): Tile size (width, height)
            
        Returns:
            numpy.ndarray: Solid color tile
        """
        tile = np.full((size[1], size[0], 3), color, dtype=np.uint8)
        return tile
    
    def generate_mosaic(self, original_image, grid_info, color_analysis, dynamic_palette=None):
        """
        Generate the complete mosaic by replacing each grid cell with a colored tile.
        
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
        
        # Replace each grid cell with corresponding colored tile
        for (y1, y2, x1, x2), (dominant_color, category, category_color) in zip(grid_info, color_analysis):
            # Calculate tile size for this cell
            tile_size = (x2 - x1, y2 - y1)
            
            # Create colored tile using the classified category color
            colored_tile = self.create_colored_tile(category_color, tile_size)
            
            # Place tile in mosaic at the grid cell position
            mosaic[y1:y2, x1:x2] = colored_tile
        
        return mosaic
    
    def get_tile_set_preview(self, dynamic_palette, tile_size=(50, 50)):
        """
        Create a preview of the tile set (color palette).
        
        Args:
            dynamic_palette (dict): Color palette to show
            tile_size (tuple): Size of each preview tile
            
        Returns:
            numpy.ndarray: Tile set preview image
        """
        if not dynamic_palette:
            return np.zeros((tile_size[1], tile_size[0], 3), dtype=np.uint8)
        
        colors = list(dynamic_palette.values())
        n_colors = len(colors)
        
        # Arrange tiles in a single row
        preview_width = n_colors * tile_size[0]
        preview_height = tile_size[1]
        preview = np.zeros((preview_height, preview_width, 3), dtype=np.uint8)
        
        # Create each colored tile
        for i, color in enumerate(colors):
            x1 = i * tile_size[0]
            x2 = x1 + tile_size[0]
            
            # Create and place colored tile
            tile = self.create_colored_tile(color, tile_size)
            preview[:, x1:x2] = tile
        
        return preview
    
    def get_mapping_statistics(self, grid_info, color_analysis):
        """
        Get statistics about the tile mapping process.
        
        Args:
            grid_info (list): Grid cell coordinates
            color_analysis (list): Color analysis results
            
        Returns:
            dict: Mapping statistics
        """
        # Count tiles by color category
        color_counts = {}
        tile_areas = []
        
        for (y1, y2, x1, x2), (dominant_color, category, category_color) in zip(grid_info, color_analysis):
            # Count color usage
            color_counts[category] = color_counts.get(category, 0) + 1
            
            # Calculate tile area
            area = (x2 - x1) * (y2 - y1)
            tile_areas.append(area)
        
        stats = {
            'total_tiles': len(grid_info),
            'unique_colors': len(color_counts),
            'color_distribution': color_counts,
            'avg_tile_area': np.mean(tile_areas) if tile_areas else 0,
            'min_tile_area': min(tile_areas) if tile_areas else 0,
            'max_tile_area': max(tile_areas) if tile_areas else 0
        }
        
        return stats