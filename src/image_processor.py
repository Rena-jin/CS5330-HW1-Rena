import numpy as np
from PIL import Image
import cv2

class ImagePreprocessor:
    """
    Image preprocessing module for the mosaic generator.
    Handles image loading, resizing, and optional color quantization.
    """
    
    def __init__(self, target_size=(512, 512)):
        """
        Initialize the image preprocessor.
        
        Args:
            target_size (tuple): Target image size (width, height)
        """
        self.target_size = target_size
    
    def load_and_resize(self, image_input):
        """
        Load and resize image to target size while maintaining aspect ratio.
        
        Args:
            image_input: Can be file path (str) or numpy array
            
        Returns:
            numpy.ndarray: Processed image as RGB array
        """
        # Handle different input types
        if isinstance(image_input, str):
            # Load from file path
            image = Image.open(image_input)
        elif isinstance(image_input, np.ndarray):
            # Convert numpy array to PIL Image
            if image_input.dtype != np.uint8:
                image_input = (image_input * 255).astype(np.uint8)
            image = Image.fromarray(image_input)
        else:
            raise ValueError("Input must be file path (str) or numpy array")
        
        # Convert to RGB if necessary
        if image.mode != 'RGB':
            image = image.convert('RGB')
        
        # Resize while maintaining aspect ratio
        image.thumbnail(self.target_size, Image.Resampling.LANCZOS)
        
        # Create new image with target size and white background
        new_image = Image.new('RGB', self.target_size, (255, 255, 255))
        
        # Center the image
        paste_x = (self.target_size[0] - image.width) // 2
        paste_y = (self.target_size[1] - image.height) // 2
        new_image.paste(image, (paste_x, paste_y))
        
        return np.array(new_image)
    
    def apply_color_quantization(self, image, n_colors=16):
        """
        Apply color quantization to reduce the number of colors in the image.
        
        Args:
            image (numpy.ndarray): Input image as RGB array
            n_colors (int): Number of colors to reduce to
            
        Returns:
            numpy.ndarray: Color quantized image
        """
        # Reshape image to be a list of pixels
        data = image.reshape((-1, 3))
        data = np.float32(data)
        
        # Apply K-means clustering to find color centers
        criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 20, 1.0)
        _, labels, centers = cv2.kmeans(
            data, n_colors, None, criteria, 10, cv2.KMEANS_RANDOM_CENTERS
        )
        
        # Convert back to uint8 and assign colors
        centers = np.uint8(centers)
        quantized_data = centers[labels.flatten()]
        quantized_image = quantized_data.reshape(image.shape)
        
        return quantized_image
    
    def preprocess_image(self, image_input, apply_quantization=False, n_colors=16):
        """
        Complete preprocessing pipeline.
        
        Args:
            image_input: Input image (file path or numpy array)
            apply_quantization (bool): Whether to apply color quantization
            n_colors (int): Number of colors for quantization
            
        Returns:
            numpy.ndarray: Fully processed image
        """
        # Step 1: Load and resize
        processed_image = self.load_and_resize(image_input)
        
        # Step 2: Optional color quantization
        if apply_quantization:
            processed_image = self.apply_color_quantization(processed_image, n_colors)
        
        return processed_image
    
    def get_image_info(self, image):
        """
        Get basic information about the processed image.
        
        Args:
            image (numpy.ndarray): Image array
            
        Returns:
            dict: Image information
        """
        return {
            'shape': image.shape,
            'dtype': image.dtype,
            'min_value': image.min(),
            'max_value': image.max(),
            'unique_colors': len(np.unique(image.reshape(-1, image.shape[-1]), axis=0))
        }