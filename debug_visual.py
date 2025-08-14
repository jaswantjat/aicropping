#!/usr/bin/env python3
"""
Visual debugging script to understand why card detection is failing.
This will save intermediate images to see exactly where the process breaks down.
"""
import cv2
import numpy as np
from PIL import Image
import os
from src.cropper import Settings, find_card_quad_simple, find_card_quad_advanced, pil_to_bgr, _exif_correct

def debug_card_detection(image_path, output_dir="debug_output"):
    """
    Debug card detection step by step with visual outputs.
    """
    print(f"🔍 Debugging card detection for: {image_path}")
    
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    
    # Load image
    pil_img = _exif_correct(Image.open(image_path))
    bgr = pil_to_bgr(pil_img)
    
    print(f"📏 Original image size: {bgr.shape}")
    
    # Save original
    cv2.imwrite(f"{output_dir}/01_original.jpg", bgr)
    
    # Test both methods
    cfg = Settings()
    
    print("\n" + "="*50)
    print("TESTING SIMPLE METHOD")
    print("="*50)
    
    result_simple = debug_simple_method(bgr, cfg, output_dir)
    
    print("\n" + "="*50)
    print("TESTING ADVANCED METHOD")
    print("="*50)
    
    result_advanced = debug_advanced_method(bgr, cfg, output_dir)
    
    # Summary
    print("\n" + "="*50)
    print("SUMMARY")
    print("="*50)
    
    if result_simple:
        quad, score = result_simple
        print(f"✅ Simple method: SUCCESS (score: {score:.3f})")
    else:
        print("❌ Simple method: FAILED")
    
    if result_advanced:
        quad, score = result_advanced
        print(f"✅ Advanced method: SUCCESS (score: {score:.3f})")
    else:
        print("❌ Advanced method: FAILED")
    
    print(f"\n📁 Debug images saved to: {output_dir}/")

def debug_simple_method(bgr, cfg, output_dir):
    """Debug the simple detection method step by step."""
    H, W = bgr.shape[:2]
    scale = cfg.max_side_px / max(H, W)
    img = cv2.resize(bgr, None, fx=scale, fy=scale) if scale < 1 else bgr.copy()
    
    # Step 1: Grayscale conversion
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    cv2.imwrite(f"{output_dir}/02_simple_gray.jpg", gray)
    print(f"   Gray image saved: {gray.shape}")
    
    # Step 2: Gaussian blur
    gray = cv2.GaussianBlur(gray, (5, 5), 0)
    cv2.imwrite(f"{output_dir}/03_simple_blur.jpg", gray)
    
    # Step 3: Adaptive Canny
    v = np.median(gray)
    lower = int(max(0, 0.66 * v))
    upper = int(min(255, 1.33 * v))
    print(f"   Median: {v:.1f}, Canny thresholds: {lower}-{upper}")
    
    edges = cv2.Canny(gray, lower, upper)
    cv2.imwrite(f"{output_dir}/04_simple_edges.jpg", edges)
    print(f"   Edge pixels: {np.sum(edges > 0)}")
    
    # Step 4: Dilation
    if cfg.dilate_iter > 0:
        edges = cv2.dilate(edges, np.ones((3,3),np.uint8), iterations=cfg.dilate_iter)
        cv2.imwrite(f"{output_dir}/05_simple_dilated.jpg", edges)
    
    # Step 5: Find contours
    cnts, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    print(f"   Found {len(cnts)} contours")
    
    # Visualize all contours
    contour_img = img.copy()
    cv2.drawContours(contour_img, cnts, -1, (0, 255, 0), 2)
    cv2.imwrite(f"{output_dir}/06_simple_all_contours.jpg", contour_img)
    
    # Try the actual detection
    result = find_card_quad_simple(bgr, cfg)
    
    if result:
        quad, score = result
        # Draw the detected quad
        quad_img = bgr.copy()
        quad_int = quad.astype(np.int32)
        cv2.polylines(quad_img, [quad_int], True, (0, 0, 255), 3)
        cv2.imwrite(f"{output_dir}/07_simple_detected_quad.jpg", quad_img)
        print(f"   ✅ Detection successful! Score: {score:.3f}")
    else:
        print(f"   ❌ Detection failed")
    
    return result

def debug_advanced_method(bgr, cfg, output_dir):
    """Debug the advanced detection method step by step."""
    H, W = bgr.shape[:2]
    scale = cfg.max_side_px / max(H, W)
    img = cv2.resize(bgr, None, fx=scale, fy=scale) if scale < 1 else bgr.copy()
    
    # Step 1: LAB conversion
    lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
    l_channel = lab[:, :, 0]
    cv2.imwrite(f"{output_dir}/08_advanced_l_channel.jpg", l_channel)
    print(f"   L channel saved: {l_channel.shape}")
    
    # Step 2: CLAHE
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    l_enhanced = clahe.apply(l_channel)
    cv2.imwrite(f"{output_dir}/09_advanced_clahe.jpg", l_enhanced)
    print(f"   CLAHE applied")
    
    # Step 3: Blur
    gray = cv2.GaussianBlur(l_enhanced, (5, 5), 0)
    cv2.imwrite(f"{output_dir}/10_advanced_blur.jpg", gray)
    
    # Step 4: Canny
    v = np.median(gray)
    lower = int(max(0, 0.66 * v))
    upper = int(min(255, 1.33 * v))
    print(f"   Median: {v:.1f}, Canny thresholds: {lower}-{upper}")
    
    edges = cv2.Canny(gray, lower, upper)
    cv2.imwrite(f"{output_dir}/11_advanced_edges.jpg", edges)
    print(f"   Edge pixels: {np.sum(edges > 0)}")
    
    # Step 5: Dilation
    if cfg.dilate_iter > 0:
        edges = cv2.dilate(edges, np.ones((3,3),np.uint8), iterations=cfg.dilate_iter)
        cv2.imwrite(f"{output_dir}/12_advanced_dilated.jpg", edges)
    
    # Step 6: Find contours
    cnts, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    print(f"   Found {len(cnts)} contours")
    
    # Visualize all contours
    contour_img = img.copy()
    cv2.drawContours(contour_img, cnts, -1, (0, 255, 0), 2)
    cv2.imwrite(f"{output_dir}/13_advanced_all_contours.jpg", contour_img)
    
    # Try the actual detection
    result = find_card_quad_advanced(bgr, cfg)
    
    if result:
        quad, score = result
        # Draw the detected quad
        quad_img = bgr.copy()
        quad_int = quad.astype(np.int32)
        cv2.polylines(quad_img, [quad_int], True, (0, 0, 255), 3)
        cv2.imwrite(f"{output_dir}/14_advanced_detected_quad.jpg", quad_img)
        print(f"   ✅ Detection successful! Score: {score:.3f}")
    else:
        print(f"   ❌ Detection failed")
    
    return result

if __name__ == "__main__":
    import sys
    
    if len(sys.argv) > 1:
        image_path = sys.argv[1]
    else:
        # Use test image
        image_path = "test_id_card.png"
    
    if not os.path.exists(image_path):
        print(f"❌ Image not found: {image_path}")
        print("Usage: python debug_visual.py <image_path>")
        sys.exit(1)
    
    debug_card_detection(image_path)
