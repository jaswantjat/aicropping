#!/usr/bin/env python3
"""
Smoke test script for the improved AI cropping pipeline.
Tests the fixes mentioned in the analysis document.
"""
import os
import sys
import time
from pathlib import Path
from src.cropper import Settings, process_image_bytes

def test_with_settings(image_path, **kwargs):
    """Test cropping with custom settings"""
    print(f"\n🧪 Testing: {image_path}")
    print(f"⚙️  Settings: {kwargs}")
    
    if not os.path.exists(image_path):
        print(f"❌ Image not found: {image_path}")
        return False
    
    # Load image
    with open(image_path, 'rb') as f:
        image_data = f.read()
    
    print(f"📁 Loaded {len(image_data)} bytes")
    
    # Create settings with custom parameters
    cfg = Settings(**kwargs)
    
    # Process image
    start_time = time.time()
    result = process_image_bytes(image_data, cfg)
    elapsed = time.time() - start_time
    
    print(f"⏱️  Processing time: {elapsed:.2f}s")
    print(f"📋 Result: {result['reason']}")
    
    if result.get('ok'):
        print("✅ SUCCESS! Card detected and cropped")
        
        # Save the cropped result
        base_name = Path(image_path).stem
        output_path = f"smoke_test_{base_name}_crop.jpg"
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
        
        return False

def run_smoke_tests():
    """Run comprehensive smoke tests"""
    print("🚀 Starting AI Cropping Pipeline Smoke Tests")
    print("=" * 60)
    
    # Test cases with different settings
    test_cases = [
        {
            "name": "Default Settings",
            "settings": {}
        },
        {
            "name": "High Resolution Output",
            "settings": {"target_height": 800}
        },
        {
            "name": "Lenient Aspect Ratio",
            "settings": {"ar_tol": 0.5, "ar_tol_after": 0.4}
        },
        {
            "name": "Small Area Detection",
            "settings": {"min_area_frac": 0.01}
        },
        {
            "name": "Aggressive Edge Detection",
            "settings": {"dilate_iter": 3, "canny_lo_mult": 0.4, "canny_hi_mult": 1.8}
        }
    ]
    
    # Look for test images
    test_images = []
    for pattern in ["*.jpg", "*.jpeg", "*.png"]:
        test_images.extend(Path(".").glob(pattern))
        test_images.extend(Path("raw").glob(pattern) if Path("raw").exists() else [])
    
    if not test_images:
        print("⚠️  No test images found. Creating a synthetic test image...")
        # Import and run the debug script to create a test image
        try:
            from debug_cropper import create_test_id_card
            test_card = create_test_id_card()
            test_card.save("synthetic_id_card.png")
            test_images = [Path("synthetic_id_card.png")]
            print("✅ Created synthetic test image: synthetic_id_card.png")
        except Exception as e:
            print(f"❌ Could not create synthetic image: {e}")
            return
    
    print(f"📁 Found {len(test_images)} test images: {[img.name for img in test_images]}")
    
    # Run tests
    total_tests = 0
    passed_tests = 0
    
    for image_path in test_images[:2]:  # Test with first 2 images to avoid spam
        print(f"\n{'='*60}")
        print(f"🖼️  Testing with image: {image_path}")
        print(f"{'='*60}")
        
        for test_case in test_cases:
            print(f"\n📋 Test Case: {test_case['name']}")
            print("-" * 40)
            
            success = test_with_settings(str(image_path), **test_case['settings'])
            total_tests += 1
            if success:
                passed_tests += 1
    
    # Summary
    print(f"\n{'='*60}")
    print(f"📊 SMOKE TEST SUMMARY")
    print(f"{'='*60}")
    print(f"✅ Passed: {passed_tests}/{total_tests}")
    print(f"❌ Failed: {total_tests - passed_tests}/{total_tests}")
    print(f"📈 Success Rate: {100 * passed_tests / total_tests:.1f}%")
    
    if passed_tests == total_tests:
        print("🎉 All tests passed! The pipeline improvements are working correctly.")
    elif passed_tests > 0:
        print("⚠️  Some tests passed. The pipeline has been improved but may need fine-tuning.")
    else:
        print("❌ All tests failed. Check the implementation and test images.")

if __name__ == "__main__":
    if len(sys.argv) > 1:
        # Test specific image
        image_path = sys.argv[1]
        settings = {}
        
        # Parse additional arguments as settings
        for arg in sys.argv[2:]:
            if "=" in arg:
                key, value = arg.split("=", 1)
                try:
                    # Try to convert to appropriate type
                    if "." in value:
                        settings[key] = float(value)
                    else:
                        settings[key] = int(value)
                except ValueError:
                    settings[key] = value
        
        test_with_settings(image_path, **settings)
    else:
        # Run full smoke test suite
        run_smoke_tests()
