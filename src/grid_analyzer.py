import numpy as np
from sklearn.cluster import KMeans
import cv2

class GridAnalyzer:
    """
    Improved grid analysis module for the mosaic generator.
    Better handles geometric patterns and uniform regions with improved color classification.
    """
    
    def __init__(self, base_grid_size=32):
        """
        Initialize the grid analyzer.
        
        Args:
            base_grid_size (int): Base size for grid division
        """
        self.base_grid_size = base_grid_size
        
    def create_adaptive_grid(self, image, complexity_threshold=50):
        """
        Create adaptive grid with improved geometric pattern handling.
        
        Args:
            image (numpy.ndarray): Input image as RGB array
            complexity_threshold (float): Threshold for subdivision decision
            
        Returns:
            list: List of grid cell coordinates (y1, y2, x1, x2)
        """
        h, w = image.shape[:2]
        grid_info = []
        
        # Detect image type to adjust processing
        image_type = self._detect_image_type(image)
        
        # Adjust threshold based on image type
        adjusted_threshold = self._get_adaptive_threshold(complexity_threshold, image_type)
        
        # Calculate number of base grid cells
        grid_h = h // self.base_grid_size
        grid_w = w // self.base_grid_size
        
        for i in range(grid_h):
            for j in range(grid_w):
                # Calculate cell boundaries
                y1 = i * self.base_grid_size
                y2 = min((i + 1) * self.base_grid_size, h)
                x1 = j * self.base_grid_size
                x2 = min((j + 1) * self.base_grid_size, w)
                
                # Extract cell from image
                cell = image[y1:y2, x1:x2]
                
                # Calculate multiple complexity metrics
                complexity_score = self._calculate_improved_complexity(cell, image_type)
                
                # Enhanced subdivision decision
                should_subdivide = self._should_subdivide_improved(
                    cell, complexity_score, adjusted_threshold, image_type, x2-x1, y2-y1
                )
                
                if should_subdivide:
                    # Subdivide into 4 smaller cells
                    subcells = self._subdivide_cell(y1, y2, x1, x2)
                    grid_info.extend(subcells)
                else:
                    # Keep as single cell
                    grid_info.append((y1, y2, x1, x2))
        
        return grid_info
    
    def _detect_image_type(self, image):
        """
        Detect the type of image to adjust processing parameters.
        
        Args:
            image (numpy.ndarray): Input image
            
        Returns:
            str: Image type ('geometric', 'portrait', 'natural', 'mixed')
        """
        # Convert to grayscale for analysis
        gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
        
        # Calculate various image characteristics
        
        # 1. Edge density
        edges = cv2.Canny(gray, 50, 150)
        edge_density = np.sum(edges > 0) / edges.size
        
        # 2. Color variance
        color_variance = np.var(image)
        
        # 3. Dominant colors count (rough estimate)
        unique_colors = len(np.unique(image.reshape(-1, 3), axis=0))
        
        # 4. Gradient uniformity (detect geometric patterns)
        grad_x = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=3)
        grad_y = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=3)
        gradient_magnitude = np.sqrt(grad_x**2 + grad_y**2)
        gradient_std = np.std(gradient_magnitude)
        
        # 5. Check for face-like characteristics (for portraits)
        face_score = self._calculate_face_likelihood(image)
        
        # Classification logic
        if face_score > 0.3:
            return 'portrait'
        elif edge_density > 0.15 and gradient_std > 50 and unique_colors < 50:
            return 'geometric'
        elif color_variance > 1000 and unique_colors > 100:
            return 'natural'
        else:
            return 'mixed'
    
    def _calculate_face_likelihood(self, image):
        """
        Simple heuristic to detect if image contains faces.
        
        Args:
            image (numpy.ndarray): Input image
            
        Returns:
            float: Face likelihood score (0-1)
        """
        # Look for skin-tone colors in central region
        h, w = image.shape[:2]
        center_region = image[h//4:3*h//4, w//4:3*w//4]
        
        # Define skin tone ranges (rough approximation)
        skin_tones = [
            ([100, 80, 60], [255, 220, 180]),   # Light skin
            ([60, 40, 20], [200, 160, 120]),    # Medium skin
            ([30, 20, 10], [120, 80, 60])       # Dark skin
        ]
        
        skin_pixels = 0
        total_pixels = center_region.size // 3
        
        for lower, upper in skin_tones:
            mask = cv2.inRange(center_region, np.array(lower), np.array(upper))
            skin_pixels += np.sum(mask > 0)
        
        return min(skin_pixels / total_pixels, 1.0)
    
    def _get_adaptive_threshold(self, base_threshold, image_type):
        """
        Adjust complexity threshold based on image type.
        
        Args:
            base_threshold (float): Base complexity threshold
            image_type (str): Detected image type
            
        Returns:
            float: Adjusted threshold
        """
        adjustments = {
            'geometric': 1.5,    # Higher threshold - less subdivision for geometric
            'portrait': 0.9,     # Slightly lower - preserve skin detail
            'natural': 1.0,      # Keep base threshold
            'mixed': 1.1         # Slightly higher
        }
        
        return base_threshold * adjustments.get(image_type, 1.0)
    
    def _calculate_improved_complexity(self, cell, image_type):
        """
        Improved complexity calculation that handles different image types better.
        
        Args:
            cell (numpy.ndarray): Image cell
            image_type (str): Type of image being processed
            
        Returns:
            float: Complexity score
        """
        if cell.size == 0:
            return 0
            
        # Base metrics
        color_std = np.std(cell)
        
        # Edge detection with adjusted parameters
        if len(cell.shape) == 3:
            gray = cv2.cvtColor(cell, cv2.COLOR_RGB2GRAY)
        else:
            gray = cell
            
        # Adjust edge detection for different image types
        if image_type == 'geometric':
            # Use stronger edge detection for geometric patterns
            edges = cv2.Canny(gray, 100, 200)  # Higher thresholds
        else:
            # Standard edge detection for natural images
            edges = cv2.Canny(gray, 50, 150)
            
        edge_density = np.sum(edges > 0) / edges.size
        
        # Local contrast using Laplacian
        laplacian = cv2.Laplacian(gray, cv2.CV_64F)
        local_contrast = np.var(laplacian)
        
        # Texture analysis using Local Binary Pattern-like approach
        texture_score = self._calculate_texture_complexity(gray)
        
        # Weighted combination based on image type
        if image_type == 'geometric':
            # For geometric images, prioritize actual detail over boundaries
            complexity = (
                color_std * 0.6 +           # Color variation (more weight)
                edge_density * 30 +         # Edge density (reduced weight)
                local_contrast * 0.001 +    # Local contrast (reduced)
                texture_score * 0.4         # Texture (moderate weight)
            )
        elif image_type == 'portrait':
            # For portraits, balance all factors
            complexity = (
                color_std * 0.4 +
                edge_density * 60 +
                local_contrast * 0.002 +
                texture_score * 0.5
            )
        else:
            # For natural images, use original weighting
            complexity = (
                color_std * 0.4 +
                edge_density * 80 +
                local_contrast * 0.002 +
                texture_score * 0.6
            )
        
        return complexity
    
    def _calculate_texture_complexity(self, gray_cell):
        """
        Calculate texture complexity using local patterns.
        
        Args:
            gray_cell (numpy.ndarray): Grayscale cell
            
        Returns:
            float: Texture complexity score
        """
        if gray_cell.size < 9:  # Too small for texture analysis
            return 0
            
        # Simple texture measure: variance of local differences
        h, w = gray_cell.shape
        if h < 3 or w < 3:
            return 0
            
        # Calculate local differences in 3x3 neighborhoods
        differences = []
        for i in range(1, h-1):
            for j in range(1, w-1):
                center = gray_cell[i, j]
                neighborhood = gray_cell[i-1:i+2, j-1:j+2]
                local_diff = np.std(neighborhood - center)
                differences.append(local_diff)
        
        return np.mean(differences) if differences else 0
    
    def _should_subdivide_improved(self, cell, complexity_score, threshold, image_type, cell_w, cell_h):
        """
        Improved subdivision decision with better uniform region detection.
        
        Args:
            cell: Image cell
            complexity_score: Calculated complexity
            threshold: Complexity threshold
            image_type: Type of image
            cell_w, cell_h: Cell dimensions
            
        Returns:
            bool: Whether to subdivide
        """
        # Basic size and complexity checks
        if cell.size < 100 or cell_w < 16 or cell_h < 16:
            return False
            
        if complexity_score <= threshold:
            return False
        
        # Additional checks for uniform regions
        if self._is_uniform_region(cell, image_type):
            return False
            
        # Check if subdivision would create meaningful differences
        if not self._would_subdivision_help(cell):
            return False
            
        return True
    
    def _is_uniform_region(self, cell, image_type):
        """
        Detect if a region is essentially uniform despite having edges.
        
        Args:
            cell: Image cell
            image_type: Type of image
            
        Returns:
            bool: True if region is uniform
        """
        # Calculate color clusters
        pixels = cell.reshape(-1, 3)
        
        # Try K-means with small number of clusters
        try:
            kmeans = KMeans(n_clusters=min(3, len(pixels)), random_state=42, n_init=10)
            kmeans.fit(pixels)
            
            # Check if most pixels belong to dominant cluster
            labels = kmeans.labels_
            cluster_sizes = np.bincount(labels)
            dominant_ratio = np.max(cluster_sizes) / len(labels)
            
            # For geometric images, be more strict about uniformity
            uniformity_threshold = 0.8 if image_type == 'geometric' else 0.7
            
            if dominant_ratio > uniformity_threshold:
                return True
                
        except:
            pass
        
        # Alternative check: low color variance
        if np.std(cell) < 15:  # Very uniform color
            return True
            
        return False
    
    def _would_subdivision_help(self, cell):
        """
        Check if subdividing this cell would create meaningful differences.
        
        Args:
            cell: Image cell
            
        Returns:
            bool: True if subdivision would help
        """
        h, w = cell.shape[:2]
        mid_h, mid_w = h // 2, w // 2
        
        # Get the 4 quadrants
        quadrants = [
            cell[:mid_h, :mid_w],      # Top-left
            cell[:mid_h, mid_w:],      # Top-right
            cell[mid_h:, :mid_w],      # Bottom-left
            cell[mid_h:, mid_w:]       # Bottom-right
        ]
        
        # Calculate mean color for each quadrant
        quadrant_means = []
        for quad in quadrants:
            if quad.size > 0:
                quadrant_means.append(np.mean(quad, axis=(0, 1)))
        
        if len(quadrant_means) < 2:
            return False
            
        # Check if quadrants are significantly different
        max_diff = 0
        for i in range(len(quadrant_means)):
            for j in range(i+1, len(quadrant_means)):
                diff = np.linalg.norm(quadrant_means[i] - quadrant_means[j])
                max_diff = max(max_diff, diff)
        
        # If maximum difference between quadrants is significant, subdivision helps
        return max_diff > 25  # Adjustable threshold
    
    def _subdivide_cell(self, y1, y2, x1, x2):
        """
        Subdivide a cell into 4 smaller subcells.
        """
        mid_x = (x1 + x2) // 2
        mid_y = (y1 + y2) // 2
        
        return [
            (y1, mid_y, x1, mid_x),    # Top-left
            (y1, mid_y, mid_x, x2),    # Top-right
            (mid_y, y2, x1, mid_x),    # Bottom-left
            (mid_y, y2, mid_x, x2)     # Bottom-right
        ]
    
    def analyze_cell_color(self, cell):
        """
        Improved cell color analysis with better dominant color detection.
        """
        if cell.size == 0:
            return np.array([128, 128, 128])
            
        pixels = cell.reshape(-1, 3)
        valid_pixels = pixels[~np.isnan(pixels).any(axis=1)]
        
        if len(valid_pixels) == 0:
            return np.array([128, 128, 128])
        
        try:
            # Use more clusters for better color analysis
            n_clusters = min(5, max(1, len(valid_pixels) // 100))
            kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
            kmeans.fit(valid_pixels)
            
            # Select the cluster with most pixels as dominant color
            labels = kmeans.labels_
            cluster_sizes = np.bincount(labels)
            dominant_cluster = np.argmax(cluster_sizes)
            dominant_color = kmeans.cluster_centers_[dominant_cluster]
            
            dominant_color = np.clip(dominant_color, 0, 255)
            return dominant_color.astype(int)
        except Exception:
            return np.mean(valid_pixels, axis=0).astype(int)
    
    def generate_dynamic_palette(self, image, n_colors=24):
        """
        Generate a more comprehensive dynamic color palette with increased color count.
        
        Args:
            image (numpy.ndarray): Input image
            n_colors (int): Number of colors in the palette (increased from 16 to 24)
            
        Returns:
            dict: Dictionary mapping color names to RGB values
        """
        # Reshape image to pixel array
        pixels = image.reshape(-1, 3)
        
        # Remove any invalid pixels
        valid_pixels = pixels[~np.isnan(pixels).any(axis=1)]
        
        if len(valid_pixels) == 0:
            return self._get_fallback_palette()
        
        # Use K-means to find dominant colors
        try:
            kmeans = KMeans(n_clusters=n_colors, random_state=42, n_init=10)
            kmeans.fit(valid_pixels)
            palette_colors = kmeans.cluster_centers_
            
            # Sort colors by frequency (most common first)
            labels = kmeans.labels_
            color_counts = np.bincount(labels)
            sorted_indices = np.argsort(color_counts)[::-1]
            
            # Create dynamic palette dictionary
            dynamic_palette = {}
            color_names = self._generate_color_names(palette_colors[sorted_indices])
            
            for i, (name, color) in enumerate(zip(color_names, palette_colors[sorted_indices])):
                # Ensure colors are in valid range
                color_rgb = np.clip(color, 0, 255).astype(int)
                dynamic_palette[name] = tuple(color_rgb)
            
            return dynamic_palette
            
        except Exception as e:
            print(f"Warning: Dynamic palette generation failed: {e}")
            return self._get_fallback_palette()
    
    def _generate_color_names(self, colors):
        """
        Generate meaningful names for colors based on their RGB values.
        
        Args:
            colors (numpy.ndarray): Array of RGB color values
            
        Returns:
            list: List of color names
        """
        color_names = []
        
        for i, color in enumerate(colors):
            r, g, b = color
            
            # Create descriptive names based on color characteristics
            if r > 200 and g > 200 and b > 200:
                name = f"light_tone_{i+1}"
            elif r < 50 and g < 50 and b < 50:
                name = f"dark_tone_{i+1}"
            elif r > g and r > b:
                if r > 150:
                    name = f"bright_red_{i+1}" if r - max(g, b) > 50 else f"warm_tone_{i+1}"
                else:
                    name = f"muted_red_{i+1}"
            elif g > r and g > b:
                if g > 150:
                    name = f"bright_green_{i+1}" if g - max(r, b) > 50 else f"natural_tone_{i+1}"
                else:
                    name = f"forest_tone_{i+1}"
            elif b > r and b > g:
                if b > 150:
                    name = f"bright_blue_{i+1}" if b - max(r, g) > 50 else f"cool_tone_{i+1}"
                else:
                    name = f"deep_blue_{i+1}"
            elif abs(r - g) < 30 and abs(g - b) < 30 and abs(r - b) < 30:
                # Grayscale
                if r > 180:
                    name = f"light_gray_{i+1}"
                elif r < 80:
                    name = f"charcoal_{i+1}"
                else:
                    name = f"medium_gray_{i+1}"
            else:
                # Mixed colors
                if r > 100 and g > 100:
                    name = f"earth_tone_{i+1}"
                elif r > 100 and b > 100:
                    name = f"purple_tone_{i+1}"
                elif g > 100 and b > 100:
                    name = f"teal_tone_{i+1}"
                else:
                    name = f"mixed_tone_{i+1}"
            
            color_names.append(name)
        
        return color_names
    
    def _get_fallback_palette(self):
        """
        Enhanced fallback palette with more colors for better classification.
        """
        return {
            'white': (248, 248, 255),
            'light_gray': (192, 192, 192),
            'medium_gray': (128, 128, 128),
            'dark_gray': (64, 64, 64),
            'black': (25, 25, 25),
            'bright_red': (220, 20, 20),
            'dark_red': (139, 0, 0),
            'bright_green': (0, 255, 0),
            'forest_green': (34, 139, 34),
            'dark_green': (0, 100, 0),
            'bright_blue': (30, 144, 255),
            'navy_blue': (0, 0, 128),
            'sky_blue': (135, 206, 235),
            'yellow': (255, 255, 0),
            'gold': (255, 215, 0),
            'orange': (255, 165, 0),
            'purple': (138, 43, 226),
            'violet': (238, 130, 238),
            'brown': (139, 69, 19),
            'tan': (210, 180, 140),
            'pink': (255, 182, 193),
            'cyan': (0, 255, 255),
            'teal': (0, 128, 128),
            'beige': (245, 245, 220)
        }
    
    def classify_color_category(self, dominant_color, dynamic_palette=None):
        """
        Enhanced color classification using LAB color space for better accuracy.
        
        Args:
            dominant_color (numpy.ndarray): RGB color values
            dynamic_palette (dict): Optional dynamic color palette
            
        Returns:
            tuple: (category_name, category_color_rgb)
        """
        # Use dynamic palette if provided, otherwise fallback to static
        if dynamic_palette is None:
            color_categories = self._get_fallback_palette()
        else:
            color_categories = dynamic_palette
        
        try:
            # Convert dominant color to LAB color space for perceptually accurate matching
            dominant_lab = cv2.cvtColor(
                np.uint8([[dominant_color]]), 
                cv2.COLOR_RGB2LAB
            )[0][0].astype(float)
            
            min_distance = float('inf')
            closest_category = list(color_categories.keys())[0]
            closest_color = list(color_categories.values())[0]
            
            for category, color in color_categories.items():
                # Convert palette color to LAB
                color_lab = cv2.cvtColor(
                    np.uint8([[color]]), 
                    cv2.COLOR_RGB2LAB
                )[0][0].astype(float)
                
                # Calculate Euclidean distance in LAB space
                distance = np.linalg.norm(dominant_lab - color_lab)
                
                if distance < min_distance:
                    min_distance = distance
                    closest_category = category
                    closest_color = color
            
            return closest_category, closest_color
            
        except Exception as e:
            print(f"Warning: LAB color space conversion failed: {e}")
            # Fallback to RGB-based matching with perceptual weights
            weights = np.array([0.3, 0.59, 0.11])  # RGB weights for human perception
            
            min_distance = float('inf')
            closest_category = list(color_categories.keys())[0]
            closest_color = list(color_categories.values())[0]
            
            for category, color in color_categories.items():
                diff = dominant_color - np.array(color)
                distance = np.sqrt(np.sum((diff * weights) ** 2))
                
                if distance < min_distance:
                    min_distance = distance
                    closest_category = category
                    closest_color = color
                    
            return closest_category, closest_color
    
    def analyze_image_grid(self, image, complexity_threshold=50):
        """
        Complete grid analysis pipeline with improved dynamic color palette generation.
        
        Args:
            image (numpy.ndarray): Input image as RGB array
            complexity_threshold (float): Complexity threshold for subdivision
            
        Returns:
            tuple: (grid_info, color_analysis, dynamic_palette)
                - grid_info: List of cell coordinates
                - color_analysis: List of (dominant_color, category, category_color) for each cell
                - dynamic_palette: Generated color palette for this image
        """
        # Step 1: Generate enhanced dynamic color palette with more colors
        print("Generating enhanced dynamic color palette...")
        dynamic_palette = self.generate_dynamic_palette(image, n_colors=24)
        print(f"Generated palette with {len(dynamic_palette)} colors")
        
        # Step 2: Create adaptive grid
        grid_info = self.create_adaptive_grid(image, complexity_threshold)
        
        # Step 3: Analyze color for each cell using enhanced methods
        color_analysis = []
        
        for i, (y1, y2, x1, x2) in enumerate(grid_info):
            cell = image[y1:y2, x1:x2]
            
            # Analyze dominant color with improved method
            dominant_color = self.analyze_cell_color(cell)
            
            # Classify into category using LAB color space
            category, category_color = self.classify_color_category(dominant_color, dynamic_palette)
            
            color_analysis.append((dominant_color, category, category_color))
            
            # Debug output for first few cells
            if i < 3:
                print(f"Cell {i}: RGB{tuple(dominant_color)} -> {category} RGB{category_color}")
        
        return grid_info, color_analysis, dynamic_palette
    
    def get_grid_statistics(self, grid_info, color_analysis):
        """
        Get detailed statistics about the grid analysis results.
        """
        cell_sizes = [(x2-x1) * (y2-y1) for y1, y2, x1, x2 in grid_info]
        categories = [analysis[1] for analysis in color_analysis]
        unique_categories = list(set(categories))
        
        base_cell_size = self.base_grid_size ** 2
        subdivided_cells = sum(1 for size in cell_sizes if size < base_cell_size)
        
        return {
            'total_cells': len(grid_info),
            'unique_colors': len(unique_categories),
            'color_categories': unique_categories,
            'avg_cell_size': np.mean(cell_sizes) if cell_sizes else 0,
            'min_cell_size': min(cell_sizes) if cell_sizes else 0,
            'max_cell_size': max(cell_sizes) if cell_sizes else 0,
            'subdivision_ratio': subdivided_cells / len(cell_sizes) if cell_sizes else 0,
            'base_cell_size': base_cell_size,
            'subdivided_cells': subdivided_cells,
            'large_cells': len(cell_sizes) - subdivided_cells
        }