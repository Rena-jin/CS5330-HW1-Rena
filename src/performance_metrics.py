import numpy as np
import cv2
from skimage.metrics import structural_similarity as ssim

class PerformanceMetrics:
    """
    Performance metrics module for evaluating mosaic reconstruction quality.
    Implements similarity metrics as required by Step 5.
    """
    
    def __init__(self):
        """Initialize the performance metrics calculator."""
        pass
    
    def calculate_mse(self, original, mosaic):
        """
        Calculate Mean Squared Error between original and mosaic images.
        
        Args:
            original (numpy.ndarray): Original image
            mosaic (numpy.ndarray): Reconstructed mosaic image
            
        Returns:
            float: MSE value (lower is better, 0 = perfect match)
        """
        if original.shape != mosaic.shape:
            raise ValueError(f"Image shapes don't match: {original.shape} vs {mosaic.shape}")
        
        # Calculate MSE
        mse = np.mean((original.astype(float) - mosaic.astype(float)) ** 2)
        return mse
    
    def calculate_ssim(self, original, mosaic):
        """
        Calculate Structural Similarity Index between original and mosaic images.
        
        Args:
            original (numpy.ndarray): Original image
            mosaic (numpy.ndarray): Reconstructed mosaic image
            
        Returns:
            float: SSIM value (higher is better, range -1 to 1, 1 = perfect match)
        """
        if original.shape != mosaic.shape:
            raise ValueError(f"Image shapes don't match: {original.shape} vs {mosaic.shape}")
        
        # Convert to grayscale for SSIM calculation
        if len(original.shape) == 3:
            original_gray = cv2.cvtColor(original, cv2.COLOR_RGB2GRAY)
            mosaic_gray = cv2.cvtColor(mosaic, cv2.COLOR_RGB2GRAY)
        else:
            original_gray = original
            mosaic_gray = mosaic
        
        # Calculate SSIM with appropriate parameters
        ssim_score = ssim(
            original_gray, 
            mosaic_gray, 
            data_range=255,
            full=False
        )
        
        return ssim_score
    
    def calculate_color_fidelity(self, original, mosaic):
        """
        Calculate color fidelity using histogram comparison.
        
        Args:
            original (numpy.ndarray): Original image
            mosaic (numpy.ndarray): Reconstructed mosaic image
            
        Returns:
            float: Color fidelity score (higher is better, 0-1)
        """
        color_similarities = []
        
        # Compare histograms for each RGB channel
        for channel in range(3):
            hist_orig = cv2.calcHist([original], [channel], None, [256], [0, 256])
            hist_mosaic = cv2.calcHist([mosaic], [channel], None, [256], [0, 256])
            
            # Normalize histograms
            hist_orig = hist_orig.flatten() / (hist_orig.sum() + 1e-10)
            hist_mosaic = hist_mosaic.flatten() / (hist_mosaic.sum() + 1e-10)
            
            # Calculate correlation coefficient
            correlation = cv2.compareHist(hist_orig, hist_mosaic, cv2.HISTCMP_CORREL)
            color_similarities.append(max(0, correlation))
        
        return np.mean(color_similarities)
    
    def calculate_perceptual_similarity(self, original, mosaic):
        """
        Calculate perceptual similarity using LAB color space.
        
        Args:
            original (numpy.ndarray): Original image
            mosaic (numpy.ndarray): Reconstructed mosaic image
            
        Returns:
            float: Perceptual similarity score (lower difference is better)
        """
        # Convert to LAB color space (more perceptually uniform)
        original_lab = cv2.cvtColor(original, cv2.COLOR_RGB2LAB)
        mosaic_lab = cv2.cvtColor(mosaic, cv2.COLOR_RGB2LAB)
        
        # Calculate mean absolute difference in LAB space
        lab_difference = np.mean(np.abs(original_lab.astype(float) - mosaic_lab.astype(float)))
        
        return lab_difference
    
    def calculate_edge_preservation(self, original, mosaic):
        """
        Calculate how well edges are preserved in the mosaic.
        
        Args:
            original (numpy.ndarray): Original image
            mosaic (numpy.ndarray): Reconstructed mosaic image
            
        Returns:
            float: Edge preservation score (higher is better, 0-1)
        """
        # Convert to grayscale
        original_gray = cv2.cvtColor(original, cv2.COLOR_RGB2GRAY)
        mosaic_gray = cv2.cvtColor(mosaic, cv2.COLOR_RGB2GRAY)
        
        # Extract edges using Canny edge detection
        original_edges = cv2.Canny(original_gray, 100, 200)
        mosaic_edges = cv2.Canny(mosaic_gray, 100, 200)
        
        # Calculate edge overlap (Intersection over Union)
        intersection = np.logical_and(original_edges, mosaic_edges)
        union = np.logical_or(original_edges, mosaic_edges)
        
        if np.sum(union) == 0:
            return 1.0  # No edges in either image
        
        edge_iou = np.sum(intersection) / np.sum(union)
        return edge_iou
    
    def evaluate_comprehensive_quality(self, original, mosaic):
        """
        Comprehensive quality evaluation combining multiple metrics.
        
        Args:
            original (numpy.ndarray): Original image
            mosaic (numpy.ndarray): Reconstructed mosaic image
            
        Returns:
            dict: Complete evaluation results
        """
        try:
            # Calculate all individual metrics
            mse = self.calculate_mse(original, mosaic)
            ssim_score = self.calculate_ssim(original, mosaic)
            color_fidelity = self.calculate_color_fidelity(original, mosaic)
            perceptual_diff = self.calculate_perceptual_similarity(original, mosaic)
            edge_preservation = self.calculate_edge_preservation(original, mosaic)
            
            # Normalize metrics to 0-100 scale for easier interpretation
            # MSE: lower is better, normalize relative to max possible error
            mse_normalized = max(0, 100 - (mse / 1000))  # Adjust divisor based on typical values
            
            # SSIM: -1 to 1, convert to 0-100
            ssim_normalized = (ssim_score + 1) * 50
            
            # Color fidelity: 0-1, convert to 0-100
            color_normalized = color_fidelity * 100
            
            # Perceptual difference: lower is better, normalize
            perceptual_normalized = max(0, 100 - (perceptual_diff / 5))
            
            # Edge preservation: 0-1, convert to 0-100
            edge_normalized = edge_preservation * 100
            
            # Calculate weighted overall score
            overall_score = (
                ssim_normalized * 0.3 +      # Structural similarity - 30%
                mse_normalized * 0.25 +      # Pixel accuracy - 25%
                color_normalized * 0.25 +    # Color preservation - 25%
                edge_normalized * 0.1 +      # Edge preservation - 10%
                perceptual_normalized * 0.1  # Perceptual quality - 10%
            )
            
            # Create comprehensive results
            results = {
                'mse': mse,
                'mse_normalized': mse_normalized,
                'ssim': ssim_score,
                'ssim_normalized': ssim_normalized,
                'color_fidelity': color_fidelity,
                'color_normalized': color_normalized,
                'perceptual_difference': perceptual_diff,
                'perceptual_normalized': perceptual_normalized,
                'edge_preservation': edge_preservation,
                'edge_normalized': edge_normalized,
                'overall_score': overall_score
            }
            
            return results
            
        except Exception as e:
            print(f"Error calculating performance metrics: {e}")
            return self._get_default_metrics()
    
    def _get_default_metrics(self):
        """Return default metrics in case of calculation error."""
        return {
            'mse': 0,
            'mse_normalized': 0,
            'ssim': 0,
            'ssim_normalized': 0,
            'color_fidelity': 0,
            'color_normalized': 0,
            'perceptual_difference': 0,
            'perceptual_normalized': 0,
            'edge_preservation': 0,
            'edge_normalized': 0,
            'overall_score': 0
        }
    
    def get_quality_assessment(self, metrics):
        """
        Provide qualitative assessment based on calculated metrics.
        
        Args:
            metrics (dict): Calculated performance metrics
            
        Returns:
            str: Quality assessment description
        """
        overall = metrics['overall_score']
        ssim_score = metrics['ssim']
        mse = metrics['mse']
        
        if overall >= 85:
            quality = "Excellent"
            description = "Outstanding mosaic quality with high similarity to original"
        elif overall >= 75:
            quality = "Very Good"
            description = "High quality mosaic with good preservation of image features"
        elif overall >= 65:
            quality = "Good"
            description = "Good mosaic quality with acceptable feature preservation"
        elif overall >= 50:
            quality = "Fair"
            description = "Reasonable mosaic quality with some detail loss"
        elif overall >= 35:
            quality = "Poor"
            description = "Low quality mosaic with significant differences from original"
        else:
            quality = "Very Poor"
            description = "Major structural and color differences from original"
        
        # Add specific insights
        insights = []
        
        if ssim_score > 0.8:
            insights.append("excellent structural preservation")
        elif ssim_score > 0.6:
            insights.append("good structural preservation")
        elif ssim_score > 0.4:
            insights.append("moderate structural preservation")
        else:
            insights.append("poor structural preservation")
        
        if mse < 300:
            insights.append("low pixel error")
        elif mse < 800:
            insights.append("moderate pixel error")
        else:
            insights.append("high pixel error")
        
        if metrics['color_fidelity'] > 0.8:
            insights.append("excellent color matching")
        elif metrics['color_fidelity'] > 0.6:
            insights.append("good color matching")
        else:
            insights.append("color matching needs improvement")
        
        assessment = f"{quality} ({overall:.1f}/100)\n{description}\nKey findings: {', '.join(insights)}"
        
        return assessment