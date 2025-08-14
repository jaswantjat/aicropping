#!/usr/bin/env python3
"""
Test script to demonstrate ID card cropping functionality
"""
import os
import io
from PIL import Image
from src.cropper import Settings, process_image_bytes

def test_with_image(image_path):
    """Test cropping with a specific image"""
    print(f"\n🧪 Testing with: {image_path}")
    
    if not os.path.exists(image_path):
        print(f"❌ Image not found: {image_path}")
        return False
    
    # Load image
    with open(image_path, 'rb') as f:
        image_data = f.read()
    
    print(f"📁 Loaded {len(image_data)} bytes")
    
    # Process with default settings
    cfg = Settings()
    result = process_image_bytes(image_data, cfg)
    
    print(f"📋 Result: {result['reason']}")
    
    if result.get('ok'):
        print("✅ SUCCESS! Card detected and cropped")
        
        # Save the cropped result
        output_path = f"cropped_{os.path.basename(image_path)}"
        with open(output_path, 'wb') as f:
            f.write(result['image_bytes'])
        print(f"💾 Saved cropped image to: {output_path}")
        
        # Print metadata
        meta = result.get('meta', {})
        print(f"📊 Metadata:")
        for key, value in meta.items():
            if isinstance(value, float):
                print(f"   {key}: {value:.3f}")
            else:
                print(f"   {key}: {value}")
        
        return True
    else:
        print(f"❌ FAILED: {result['reason']}")
        
        # Print debug info
        meta = result.get('meta', {})
        if meta:
            print(f"🔍 Debug info:")
            for key, value in meta.items():
                if isinstance(value, float):
                    print(f"   {key}: {value:.3f}")
                else:
                    print(f"   {key}: {value}")
        
        # Suggest parameter adjustments
        print("\n💡 Troubleshooting suggestions:")
        if result['reason'] == 'no_quad':
            print("   - Try lowering ar_tol (aspect ratio tolerance)")
            print("   - Try lowering min_area_frac if card appears small")
            print("   - Check if image has clear card edges")
        elif result['reason'] == 'area_gate':
            area_frac = meta.get('area_frac', 0)
            print(f"   - Area fraction {area_frac:.3f} outside limits")
            print(f"   - Current limits: {cfg.min_area_frac} - {cfg.max_area_frac}")
        elif result['reason'] == 'ar_gate':
            ar_after = meta.get('ar_after', 0)
            print(f"   - Aspect ratio {ar_after:.3f} too far from 1.586")
            print(f"   - Current tolerance: {cfg.ar_tol_after}")
        
        return False

def main():
    print("🎯 ID Card Cropping Test Suite")
    print("=" * 50)
    
    # Test with our synthetic image first
    success_count = 0
    total_tests = 0
    
    if os.path.exists("test_id_card.png"):
        total_tests += 1
        if test_with_image("test_id_card.png"):
            success_count += 1
    
    # Look for any other images in the current directory
    image_extensions = ['.jpg', '.jpeg', '.png', '.bmp', '.tif', '.tiff', '.webp']
    for file in os.listdir('.'):
        if any(file.lower().endswith(ext) for ext in image_extensions) and file != "test_id_card.png":
            total_tests += 1
            if test_with_image(file):
                success_count += 1
    
    print("\n" + "=" * 50)
    print(f"📊 Test Results: {success_count}/{total_tests} successful")
    
    if total_tests == 0:
        print("\n💡 No test images found. To test with real ID cards:")
        print("   1. Add some ID card photos to this directory")
        print("   2. Run this script again")
        print("   3. Or use the Streamlit UI at http://localhost:8501")
    elif success_count == 0:
        print("\n🔧 All tests failed. This might mean:")
        print("   - Images don't contain clear ID card boundaries")
        print("   - Parameters need adjustment for your specific images")
        print("   - Try the Streamlit UI to adjust parameters interactively")
    else:
        print(f"\n🎉 {success_count} images processed successfully!")
        print("   Check the cropped_*.jpg files in this directory")

if __name__ == "__main__":
    main()
