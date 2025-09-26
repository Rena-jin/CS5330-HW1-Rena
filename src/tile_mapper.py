import numpy as np
import cv2
from PIL import Image, ImageDraw, ImageFilter
import random
from typing import Tuple, List

class TileMapper:
    """
    Enhanced tile mapping module for the mosaic generator.
    Creates three different types of tiles based on classified colors from grid analysis.
    """
    
    def __init__(self):
        """Initialize the tile mapper."""
        self.debug_mode = False
        
    def set_debug_mode(self, debug=True):
        """Enable or disable debug mode for troubleshooting."""
        self.debug_mode = debug
    
    def create_colored_tile(self, color, size):
        """
        Create a simple colored square tile (original implementation).
        
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
    
    def create_pattern_tile(self, color: Tuple[int, int, int], size: Tuple[int, int],
                          original_cell: np.ndarray = None) -> np.ndarray:
        """
        Create subtle geometric pattern tile with enhanced white background detection.
        """
        width, height = size
        
        # Enhanced check for white background area
        if original_cell is not None and original_cell.size > 0:
            avg_color = np.mean(original_cell, axis=(0, 1))
            min_color = np.min(original_cell, axis=(0, 1))
            
            # More strict white detection - check both average and minimum values
            is_white_area = (np.all(avg_color > 235) and np.all(min_color > 200)) or np.all(avg_color > 250)
            
            if is_white_area:
                return self.create_colored_tile((255, 255, 255), size)
            
            # Also check if the color being assigned is very close to white
            if np.all(np.array(color) > 240):
                return self.create_colored_tile(color, size)
        
        tile = np.full((height, width, 3), color, dtype=np.uint8)
        
        # Create subtle contrasting color for pattern
        contrast_color = self._get_contrasting_color(color)
        
        if original_cell is not None and original_cell.size > 0:
            # Analyze gradient direction using Sobel operators
            if len(original_cell.shape) == 3:
                gray_cell = cv2.cvtColor(original_cell, cv2.COLOR_RGB2GRAY)
            else:
                gray_cell = original_cell
            
            if gray_cell.shape[0] >= 3 and gray_cell.shape[1] >= 3:
                # Calculate gradients
                grad_x = cv2.Sobel(gray_cell, cv2.CV_64F, 1, 0, ksize=3)
                grad_y = cv2.Sobel(gray_cell, cv2.CV_64F, 0, 1, ksize=3)
                
                # Calculate dominant gradient direction
                avg_grad_x = np.mean(grad_x)
                avg_grad_y = np.mean(grad_y)
                gradient_magnitude = np.sqrt(avg_grad_x**2 + avg_grad_y**2)
                
                # Choose pattern based on gradient analysis
                if gradient_magnitude > 5:
                    gradient_angle = np.arctan2(avg_grad_y, avg_grad_x) * 180 / np.pi
                    
                    if gradient_angle < 0:
                        gradient_angle += 180
                    
                    if 0 <= gradient_angle < 45 or 135 <= gradient_angle < 180:
                        pattern_type = 'horizontal_stripes'
                    elif 45 <= gradient_angle < 135:
                        pattern_type = 'vertical_stripes'
                    else:
                        pattern_type = 'diagonal_stripes'
                else:
                    pattern_type = 'dots'
            else:
                pattern_type = 'dots'
        else:
            # Fallback pattern selection
            brightness = np.mean(color)
            if brightness < 85:
                pattern_type = 'dots'
            elif brightness < 170:
                pattern_type = 'vertical_stripes'
            else:
                pattern_type = 'horizontal_stripes'
        
        # Draw subtle patterns using PIL
        img = Image.fromarray(tile)
        draw = ImageDraw.Draw(img)
        
        if pattern_type == 'horizontal_stripes':
            # Make stripes thinner and more sparse for subtle effect
            stripe_width = max(1, height // 10)  # Changed from //6 to //10 for thinner stripes
            for y in range(0, height, stripe_width * 4):  # Changed from *2 to *4 for more spacing
                end_y = min(y + stripe_width, height)
                draw.rectangle([0, y, width, end_y], fill=tuple(contrast_color))
                
        elif pattern_type == 'vertical_stripes':
            # Make stripes thinner and more sparse for subtle effect
            stripe_width = max(1, width // 10)  # Changed from //6 to //10 for thinner stripes
            for x in range(0, width, stripe_width * 4):  # Changed from *2 to *4 for more spacing
                end_x = min(x + stripe_width, width)
                draw.rectangle([x, 0, end_x, height], fill=tuple(contrast_color))
                
        elif pattern_type == 'diagonal_stripes':
            # Make diagonal lines more sparse and thinner
            stripe_spacing = max(6, min(width, height) // 3)  # Changed from //5 to //3 for more spacing
            for i in range(-height, width + height, stripe_spacing):
                draw.line([(i, 0), (i + height, height)], 
                         fill=tuple(contrast_color), width=1)  # Changed from width=2 to width=1
                
        elif pattern_type == 'dots':
            # Make dots smaller and more sparse
            dot_spacing = max(width // 3, 6)  # Changed from //4 to //3 for more spacing
            dot_radius = max(1, dot_spacing // 4)  # Changed from //3 to //4 for smaller dots
            
            for y in range(dot_radius, height, dot_spacing):
                for x in range(dot_radius, width, dot_spacing):
                    draw.ellipse([x - dot_radius, y - dot_radius, 
                                x + dot_radius, y + dot_radius], 
                               fill=tuple(contrast_color))
        
        return np.array(img)
    
    def create_oil_painting_tile(self, color: Tuple[int, int, int], size: Tuple[int, int],
                               texture_intensity: float = 0.5) -> np.ndarray:
        """
        Create an oil painting style tile with brush strokes and texture.
        """
        width, height = size
        
        # Create base color with slight variations
        base_tile = np.full((height, width, 3), color, dtype=np.float32)
        
        # Generate color variations for brush strokes
        color_variations = self._generate_color_variations(color, num_colors=4)
        
        # Create PIL image for brush effects
        img = Image.fromarray(base_tile.astype(np.uint8))
        
        # Add brush stroke texture
        brush_strokes = max(2, int(texture_intensity * 6))
        
        for _ in range(brush_strokes):
            # Create brush stroke
            stroke_img = Image.new('RGBA', (width, height), (0, 0, 0, 0))
            stroke_draw = ImageDraw.Draw(stroke_img)
            
            # Random brush stroke parameters
            start_x = random.randint(0, width)
            start_y = random.randint(0, height)
            end_x = start_x + random.randint(-width//3, width//3)
            end_y = start_y + random.randint(-height//3, height//3)
            
            # Ensure stroke stays within bounds
            end_x = max(0, min(width, end_x))
            end_y = max(0, min(height, end_y))
            
            # Select color variation
            stroke_color = random.choice(color_variations)
            stroke_width = random.randint(2, max(3, min(width, height) // 8))
            
            # Draw brush stroke
            stroke_draw.line([start_x, start_y, end_x, end_y], 
                           fill=tuple(list(stroke_color) + [128]), 
                           width=stroke_width)
            
            # Blend with base image
            img = Image.alpha_composite(img.convert('RGBA'), stroke_img).convert('RGB')
        
        # Add subtle texture noise
        tile = np.array(img, dtype=np.float32)
        noise = np.random.normal(0, texture_intensity * 15, tile.shape)
        tile = np.clip(tile + noise, 0, 255)
        
        return tile.astype(np.uint8)
    
    def _get_contrasting_color(self, color: Tuple[int, int, int]) -> np.ndarray:
        """Generate a subtle contrasting color for patterns."""
        brightness = np.mean(color)
        if brightness > 127:
            # Reduce contrast intensity from -100 to -25 for more subtle patterns
            return np.array([max(0, c - 25) for c in color], dtype=np.uint8)
        else:
            # Reduce contrast intensity from +100 to +25 for more subtle patterns
            return np.array([min(255, c + 25) for c in color], dtype=np.uint8)
    
    def _generate_color_variations(self, base_color: Tuple[int, int, int], 
                                 num_colors: int = 4) -> List[Tuple[int, int, int]]:
        """Generate color variations for oil painting effect."""
        variations = []
        base = np.array(base_color)
        
        for _ in range(num_colors):
            variation = base + np.random.normal(0, 20, 3)
            variation = np.clip(variation, 0, 255).astype(int)
            variations.append(tuple(variation))
        
        return variations
    
    def generate_mosaic_with_bounds(self, original_image, grid_info, color_analysis, 
                                  dynamic_palette=None, tile_style="solid", image_bounds=None):
        """Generate mosaic with proper boundary detection."""
        mosaic = np.zeros_like(original_image)
        
        if self.debug_mode:
            print(f"\nGenerating {tile_style} mosaic with {len(grid_info)} cells...")
            if image_bounds:
                print(f"Image bounds: {image_bounds}")
        
        color_usage = {}
        background_cells = 0
        image_cells = 0
        
        for i, ((y1, y2, x1, x2), (dominant_color, category, category_color)) in enumerate(zip(grid_info, color_analysis)):
            tile_size = (x2 - x1, y2 - y1)
            original_cell = original_image[y1:y2, x1:x2]
            
            if image_bounds is not None:
                left, top, right, bottom = image_bounds
                cell_overlaps_image = not (x2 <= left or x1 >= right or y2 <= top or y1 >= bottom)
                
                if not cell_overlaps_image:
                    colored_tile = self.create_colored_tile((255, 255, 255), tile_size)
                    background_cells += 1
                else:
                    brightness = np.mean(original_cell) if original_cell.size > 0 else 128
                    
                    if category in color_usage:
                        color_usage[category] += 1
                    else:
                        color_usage[category] = 1
                    
                    if tile_style == "pattern":
                        colored_tile = self.create_pattern_tile(category_color, tile_size, original_cell)
                    elif tile_style == "oil_painting":
                        texture_intensity = min(1.0, (255 - brightness) / 255 + 0.3)
                        colored_tile = self.create_oil_painting_tile(category_color, tile_size, texture_intensity)
                    else:  # "solid" or default
                        colored_tile = self.create_colored_tile(category_color, tile_size)
                    
                    image_cells += 1
            else:
                brightness = np.mean(original_cell) if original_cell.size > 0 else 128
                
                if tile_style == "pattern":
                    colored_tile = self.create_pattern_tile(category_color, tile_size, original_cell)
                elif tile_style == "oil_painting":
                    texture_intensity = min(1.0, (255 - brightness) / 255 + 0.3)
                    colored_tile = self.create_oil_painting_tile(category_color, tile_size, texture_intensity)
                else:
                    colored_tile = self.create_colored_tile(category_color, tile_size)
            
            mosaic[y1:y2, x1:x2] = colored_tile
        
        if self.debug_mode:
            print(f"\n{tile_style.title()} mosaic completed!")
            print(f"Background: {background_cells}, Image: {image_cells}")
        
        return mosaic
    
    def generate_mosaic(self, original_image, grid_info, color_analysis, dynamic_palette=None, tile_style="solid"):
        """Original method for backward compatibility."""
        return self.generate_mosaic_with_bounds(original_image, grid_info, color_analysis, 
                                              dynamic_palette, tile_style, image_bounds=None)
    
    def generate_debug_comparison(self, original_image, grid_info, color_analysis):
        """Generate debug comparison image."""
        h, w = original_image.shape[:2]
        original_with_grid = original_image.copy()
        dominant_color_image = np.zeros_like(original_image)
        
        for (y1, y2, x1, x2), (dominant_color, category, category_color) in zip(grid_info, color_analysis):
            original_with_grid[y1:y2, x1, :] = [255, 255, 255]
            original_with_grid[y1:y2, x2-1, :] = [255, 255, 255] 
            original_with_grid[y1, x1:x2, :] = [255, 255, 255]
            original_with_grid[y2-1, x1:x2, :] = [255, 255, 255]
            
            tile_size = (x2 - x1, y2 - y1)
            colored_tile = self.create_colored_tile(dominant_color, tile_size)
            dominant_color_image[y1:y2, x1:x2] = colored_tile
        
        comparison = np.hstack([original_with_grid, dominant_color_image])
        return comparison
    
    def analyze_color_accuracy(self, original_image, grid_info, color_analysis):
        """Analyze color classification accuracy."""
        color_differences = []
        large_diff_cells = []
        
        for i, ((y1, y2, x1, x2), (dominant_color, category, category_color)) in enumerate(zip(grid_info, color_analysis)):
            original_cell = original_image[y1:y2, x1:x2]
            original_avg = np.mean(original_cell, axis=(0, 1))
            
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
            
            if category_diff > 80:
                large_diff_cells.append({
                    'index': i,
                    'position': (x1, y1, x2, y2),
                    'category': category,
                    'difference': category_diff,
                    'original_rgb': tuple(original_avg.astype(int)),
                    'category_rgb': category_color
                })
        
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
            'large_diff_cells': large_diff_cells[:10],
            'color_differences': color_differences
        }
        
        return analysis_results
    
    def print_color_accuracy_report(self, accuracy_analysis):
        """Print detailed color classification accuracy report."""
        print(f"\n" + "="*50)
        print(f"COLOR CLASSIFICATION ACCURACY REPORT")
        print(f"="*50)
        
        print(f"Total cells: {accuracy_analysis['total_cells']}")
        print(f"Avg dominant diff: {accuracy_analysis['avg_dominant_diff']:.1f}")
        print(f"Avg category diff: {accuracy_analysis['avg_category_diff']:.1f}")
        print(f"Max category diff: {accuracy_analysis['max_category_diff']:.1f}")
        
        large_diff_count = accuracy_analysis['large_diff_cells_count']
        large_diff_pct = accuracy_analysis['large_diff_percentage']
        print(f"Large diff cells: {large_diff_count} ({large_diff_pct:.1f}%)")
        
        if accuracy_analysis['large_diff_cells']:
            print(f"\nProblematic cells:")
            for cell in accuracy_analysis['large_diff_cells'][:5]:
                print(f"  Cell {cell['index']:2d}: {cell['category']}")
                print(f"    Original: {cell['original_rgb']}")
                print(f"    Category: {cell['category_rgb']}")
                print(f"    Diff: {cell['difference']:.1f}")
    
    def get_tile_set_preview(self, dynamic_palette, tile_size=(50, 50)):
        """Create preview of color palette."""
        if not dynamic_palette:
            return np.zeros((tile_size[1], tile_size[0], 3), dtype=np.uint8)
        
        colors = list(dynamic_palette.values())
        color_names = list(dynamic_palette.keys())
        n_colors = len(colors)
        
        cols = min(4, n_colors)
        rows = (n_colors + cols - 1) // cols
        
        preview_width = cols * tile_size[0]
        preview_height = rows * tile_size[1]
        preview = np.zeros((preview_height, preview_width, 3), dtype=np.uint8)
        
        for i, (color, name) in enumerate(zip(colors, color_names)):
            row = i // cols
            col = i % cols
            
            y1 = row * tile_size[1]
            y2 = y1 + tile_size[1]
            x1 = col * tile_size[0]
            x2 = x1 + tile_size[0]
            
            tile = self.create_colored_tile(color, tile_size)
            preview[y1:y2, x1:x2] = tile
        
        return preview
    
    def get_mapping_statistics(self, grid_info, color_analysis):
        """Get statistics about the tile mapping process."""
        color_counts = {}
        tile_areas = []
        dominant_colors = []
        category_colors = []
        
        for (y1, y2, x1, x2), (dominant_color, category, category_color) in zip(grid_info, color_analysis):
            color_counts[category] = color_counts.get(category, 0) + 1
            area = (x2 - x1) * (y2 - y1)
            tile_areas.append(area)
            dominant_colors.append(dominant_color)
            category_colors.append(category_color)
        
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