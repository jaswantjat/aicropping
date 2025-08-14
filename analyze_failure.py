#!/usr/bin/env python3
"""
Analyze why specific images fail card detection.
This script helps diagnose common failure patterns.
"""
import cv2
import numpy as np
from PIL import Image
import os
import sys
from src.cropper import Settings, find_card_quad, pil_to_bgr, _exif_correct

def analyze_failure_patterns():
    """
    Analyze common failure patterns and suggest solutions.
    """
    print("🔍 COMMON FAILURE PATTERNS ANALYSIS")
    print("="*60)
    
    patterns = [
        {
            "name": "Low Contrast",
            "description": "Card and background have similar colors",
            "solutions": [
                "Increase dilate_iter to 3-4",
                "Lower min_area_frac to 0.01", 
                "Increase ar_tol to 0.5"
            ]
        },
        {
            "name": "Textured Background", 
            "description": "Complex patterns (fabric, wood, etc.)",
            "solutions": [
                "Advanced method should handle this",
                "Check if CLAHE is creating too much noise",
                "Try higher Canny thresholds"
            ]
        },
        {
            "name": "Poor Lighting",
            "description": "Shadows, uneven lighting",
            "solutions": [
                "CLAHE preprocessing should help",
                "Try different clipLimit values (1.0-4.0)",
                "Adjust tileGridSize (4x4 to 16x16)"
            ]
        },
        {
            "name": "Card Too Small",
            "description": "Card occupies <2% of image",
            "solutions": [
                "Lower min_area_frac to 0.005",
                "Check image resolution vs card size",
                "Consider cropping to focus on card area"
            ]
        },
        {
            "name": "Card Too Large", 
            "description": "Card fills entire frame",
            "solutions": [
                "Increase max_area_frac to 0.95+",
                "Check if card edges are visible",
                "May need manual cropping first"
            ]
        },
        {
            "name": "Rotation/Perspective",
            "description": "Extreme angles, heavy perspective",
            "solutions": [
                "minAreaRect should handle rotation",
                "Check if 4 corners are clearly visible",
                "May need multiple detection attempts"
            ]
        }
    ]
    
    for i, pattern in enumerate(patterns, 1):
        print(f"\n{i}. {pattern['name']}")
        print(f"   Problem: {pattern['description']}")
        print("   Solutions:")
        for solution in pattern['solutions']:
            print(f"   • {solution}")

def create_test_settings():
    """
    Create different settings configurations for testing difficult cases.
    """
    configs = {
        "default": Settings(),
        
        "low_contrast": Settings(
            dilate_iter=3,
            min_area_frac=0.01,
            ar_tol=0.5,
            canny_lo_mult=0.4,
            canny_hi_mult=1.8
        ),
        
        "textured_background": Settings(
            dilate_iter=1,  # Less dilation to avoid noise
            min_area_frac=0.05,
            ar_tol=0.4,
            debug_mode=True
        ),
        
        "small_card": Settings(
            min_area_frac=0.005,
            ar_tol=0.6,
            dilate_iter=2
        ),
        
        "large_card": Settings(
            max_area_frac=0.98,
            min_area_frac=0.3,
            ar_tol=0.3
        ),
        
        "extreme_perspective": Settings(
            ar_tol=0.8,
            min_area_frac=0.01,
            dilate_iter=4
        )
    }
    
    return configs

def test_image_with_configs(image_path):
    """
    Test an image with different configuration settings.
    """
    if not os.path.exists(image_path):
        print(f"❌ Image not found: {image_path}")
        return
    
    print(f"\n🧪 TESTING: {image_path}")
    print("="*60)
    
    # Load image
    try:
        pil_img = _exif_correct(Image.open(image_path))
        bgr = pil_to_bgr(pil_img)
        H, W = bgr.shape[:2]
        print(f"📏 Image size: {W}×{H}")
    except Exception as e:
        print(f"❌ Failed to load image: {e}")
        return
    
    # Test with different configurations
    configs = create_test_settings()
    
    for name, cfg in configs.items():
        print(f"\n📋 Testing with '{name}' settings:")
        
        try:
            result = find_card_quad(bgr, cfg, debug=True)
            
            if result:
                quad, score = result
                area = cv2.contourArea(quad.astype(np.float32))
                area_frac = area / (H * W)
                
                print(f"   ✅ SUCCESS!")
                print(f"   Score: {score:.3f}")
                print(f"   Area fraction: {area_frac:.3f}")
                
                # Calculate aspect ratio
                pts = quad
                wA = np.linalg.norm(pts[2]-pts[3])
                wB = np.linalg.norm(pts[1]-pts[0])
                hA = np.linalg.norm(pts[1]-pts[2])
                hB = np.linalg.norm(pts[0]-pts[3])
                w = (wA+wB)/2.0
                h = (hA+hB)/2.0
                ar = w/h if h > 0 else 0
                print(f"   Aspect ratio: {ar:.3f} (target: 1.586)")
                
            else:
                print(f"   ❌ FAILED")
                
        except Exception as e:
            print(f"   💥 ERROR: {e}")

def suggest_fixes_for_failure(image_path):
    """
    Analyze a failed image and suggest specific fixes.
    """
    print(f"\n🔧 FAILURE ANALYSIS FOR: {image_path}")
    print("="*60)
    
    if not os.path.exists(image_path):
        print(f"❌ Image not found: {image_path}")
        return
    
    # Load and analyze image
    pil_img = _exif_correct(Image.open(image_path))
    bgr = pil_to_bgr(pil_img)
    H, W = bgr.shape[:2]
    
    # Basic image analysis
    gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
    
    # Brightness analysis
    mean_brightness = np.mean(gray)
    std_brightness = np.std(gray)
    
    # Edge analysis
    edges = cv2.Canny(gray, 50, 150)
    edge_density = np.sum(edges > 0) / (H * W)
    
    print(f"📊 IMAGE ANALYSIS:")
    print(f"   Size: {W}×{H}")
    print(f"   Mean brightness: {mean_brightness:.1f}")
    print(f"   Brightness std: {std_brightness:.1f}")
    print(f"   Edge density: {edge_density:.4f}")
    
    # Suggest fixes based on analysis
    print(f"\n💡 SUGGESTED FIXES:")
    
    if mean_brightness < 80:
        print("   • Low brightness detected - try CLAHE preprocessing")
    
    if std_brightness < 30:
        print("   • Low contrast detected - increase dilate_iter, lower min_area_frac")
    
    if edge_density > 0.1:
        print("   • High edge density (textured background) - use advanced method")
    elif edge_density < 0.01:
        print("   • Low edge density - increase Canny sensitivity")
    
    if W * H > 4000000:  # > 4MP
        print("   • Large image - ensure max_side_px scaling is working")
    
    # Test with recommended settings
    print(f"\n🧪 TESTING WITH RECOMMENDED SETTINGS:")
    
    recommended_cfg = Settings(
        dilate_iter=3 if std_brightness < 30 else 2,
        min_area_frac=0.005 if mean_brightness < 80 else 0.02,
        ar_tol=0.6 if edge_density > 0.1 else 0.35,
        debug_mode=True
    )
    
    result = find_card_quad(bgr, recommended_cfg, debug=True)
    
    if result:
        print("   ✅ Recommended settings worked!")
    else:
        print("   ❌ Still failing - may need manual intervention")

if __name__ == "__main__":
    if len(sys.argv) > 1:
        image_path = sys.argv[1]
        
        if sys.argv[1] == "--patterns":
            analyze_failure_patterns()
        elif sys.argv[1] == "--test-configs":
            if len(sys.argv) > 2:
                test_image_with_configs(sys.argv[2])
            else:
                print("Usage: python analyze_failure.py --test-configs <image_path>")
        else:
            suggest_fixes_for_failure(image_path)
    else:
        print("🔍 AI Cropping Failure Analysis Tool")
        print("="*40)
        print("Usage:")
        print("  python analyze_failure.py <image_path>           # Analyze specific image")
        print("  python analyze_failure.py --patterns             # Show common patterns")
        print("  python analyze_failure.py --test-configs <image> # Test with different configs")
        print()
        analyze_failure_patterns()
