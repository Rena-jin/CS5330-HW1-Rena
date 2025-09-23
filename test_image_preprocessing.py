import sys
sys.path.append('src')

import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
from image_processor import ImagePreprocessor
import os

def test_all_example_images():
    """
    Test preprocessing with all images from examples folder.
    """
    print("Testing Image Preprocessing with Example Images...")
    print("="*60)
    
    preprocessor = ImagePreprocessor(target_size=(512, 512))
    
    # Check if examples folder exists
    if not os.path.exists('examples'):
        print("❌ No examples folder found!")
        return False
    
    # Get all image files from examples folder
    image_files = []
    for file in os.listdir('examples'):
        if file.lower().endswith(('.png', '.jpg', '.jpeg')):
            image_files.append(os.path.join('examples', file))
    
    if not image_files:
        print("❌ No image files found in examples folder!")
        return False
    
    print(f"📁 Found {len(image_files)} test images:")
    for img_file in image_files:
        print(f"   • {img_file}")
    
    # Test each image
    results = []
    for i, image_path in enumerate(image_files):
        print(f"\n🧪 Testing image {i+1}/{len(image_files)}: {os.path.basename(image_path)}")
        
        try:
            # Load original image
            original = Image.open(image_path)
            original_array = np.array(original)
            
            # Test basic preprocessing
            processed_basic = preprocessor.preprocess_image(image_path, apply_quantization=False)
            
            # Test with color quantization
            processed_quantized = preprocessor.preprocess_image(image_path, apply_quantization=True, n_colors=8)
            
            # Get image information
            original_info = preprocessor.get_image_info(original_array)
            basic_info = preprocessor.get_image_info(processed_basic)
            quantized_info = preprocessor.get_image_info(processed_quantized)
            
            # Store results
            result = {
                'name': os.path.basename(image_path),
                'original': original_array,
                'processed': processed_basic,
                'quantized': processed_quantized,
                'original_info': original_info,
                'basic_info': basic_info,
                'quantized_info': quantized_info
            }
            results.append(result)
            
            print(f"   ✅ Original: {original_info['shape'][:2]} | Colors: {original_info['unique_colors']}")
            print(f"   ✅ Processed: {basic_info['shape'][:2]} | Colors: {basic_info['unique_colors']}")
            print(f"   ✅ Quantized: {quantized_info['shape'][:2]} | Colors: {quantized_info['unique_colors']}")
            
        except Exception as e:
            print(f"   ❌ Error processing {image_path}: {e}")
            continue
    
    if not results:
        print("❌ No images processed successfully!")
        return False
    
    # Create visualization
    print(f"\n📊 Creating visualization for {len(results)} images...")
    create_comparison_visualization(results)
    
    return True

def create_comparison_visualization(results):
    """
    Create a comprehensive visualization of all test results.
    """
    num_images = len(results)
    fig, axes = plt.subplots(num_images, 3, figsize=(15, 5*num_images))
    
    # Handle single image case
    if num_images == 1:
        axes = axes.reshape(1, -1)
    
    for i, result in enumerate(results):
        # Original image
        axes[i, 0].imshow(result['original'])
        axes[i, 0].set_title(f"{result['name']}\nOriginal\n"
                            f"Size: {result['original_info']['shape'][:2]}\n"
                            f"Colors: {result['original_info']['unique_colors']}")
        axes[i, 0].axis('off')
        
        # Processed image
        axes[i, 1].imshow(result['processed'])
        axes[i, 1].set_title(f"Processed (Resized)\n"
                            f"Size: {result['basic_info']['shape'][:2]}\n"
                            f"Colors: {result['basic_info']['unique_colors']}")
        axes[i, 1].axis('off')
        
        # Quantized image
        axes[i, 2].imshow(result['quantized'])
        axes[i, 2].set_title(f"Color Quantized (8 colors)\n"
                            f"Size: {result['quantized_info']['shape'][:2]}\n"
                            f"Colors: {result['quantized_info']['unique_colors']}")
        axes[i, 2].axis('off')
    
    plt.tight_layout()
    plt.savefig('test_all_preprocessing_results.png', dpi=150, bbox_inches='tight')
    plt.show()
    
    print(f"✅ Results saved as 'test_all_preprocessing_results.png'")

def print_summary(results):
    """
    Print a summary of all test results.
    """
    print(f"\n📋 PREPROCESSING SUMMARY")
    print("="*60)
    
    for result in results:
        print(f"\n🖼️  {result['name']}:")
        print(f"   Original:  {result['original_info']['shape'][:2]} → {result['original_info']['unique_colors']} colors")
        print(f"   Processed: {result['basic_info']['shape'][:2]} → {result['basic_info']['unique_colors']} colors")
        print(f"   Quantized: {result['quantized_info']['shape'][:2]} → {result['quantized_info']['unique_colors']} colors")
        
        # Calculate compression ratio
        original_colors = result['original_info']['unique_colors']
        quantized_colors = result['quantized_info']['unique_colors']
        compression_ratio = original_colors / quantized_colors if quantized_colors > 0 else 0
        print(f"   Color Compression: {compression_ratio:.1f}x reduction")

if __name__ == "__main__":
    success = test_all_example_images()
    
    if success:
        print(f"\n🎉 All image preprocessing tests completed successfully!")
        print(f"📁 Check 'test_all_preprocessing_results.png' for visual results")
    else:
        print(f"\n❌ Image preprocessing tests failed!")