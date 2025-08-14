#!/usr/bin/env python3
"""
Debug script to test and visualize the ID card cropping pipeline
"""
import cv2
import numpy as np
from PIL import Image, ImageDraw
import io
from src.cropper import Settings, find_card_quad, process_image_bytes, pil_to_bgr, bgr_to_pil

def create_test_id_card():
    """Create a synthetic ID card image for testing"""
    # Create a card-like rectangle with ID-1 aspect ratio
    card_w, card_h = 400, int(400 / 1.586)  # ~252
    
    # Create background
    bg_w, bg_h = 800, 600
    img = Image.new('RGB', (bg_w, bg_h), color='lightgray')
    draw = ImageDraw.Draw(img)
    
    # Draw a card-like rectangle (slightly rotated)
    center_x, center_y = bg_w // 2, bg_h // 2
    
    # Card corners (slightly rotated)
    angle = 15  # degrees
    cos_a, sin_a = np.cos(np.radians(angle)), np.sin(np.radians(angle))
    
    # Calculate rotated corners
    corners = []
    for dx, dy in [(-card_w//2, -card_h//2), (card_w//2, -card_h//2), 
                   (card_w//2, card_h//2), (-card_w//2, card_h//2)]:
        x = center_x + dx * cos_a - dy * sin_a
        y = center_y + dx * sin_a + dy * cos_a
        corners.append((x, y))
    
    # Draw the card
    draw.polygon(corners, fill='white', outline='black', width=3)
    
    # Add some text to make it look like an ID
    draw.text((center_x-50, center_y-20), "ID CARD", fill='black')
    draw.text((center_x-60, center_y), "1234567890", fill='black')
    draw.text((center_x-40, center_y+20), "SPAIN", fill='black')
    
    return img

def debug_quad_detection(image_path_or_pil):
    """Debug the quad detection process step by step"""
    print("🔍 Debugging quad detection...")
    
    # Load image
    if isinstance(image_path_or_pil, str):
        pil_img = Image.open(image_path_or_pil)
        print(f"📁 Loaded image from: {image_path_or_pil}")
    else:
        pil_img = image_path_or_pil
        print("🖼️ Using provided PIL image")
    
    print(f"📏 Image size: {pil_img.size}")
    
    # Convert to BGR
    bgr = pil_to_bgr(pil_img)
    print(f"🔄 Converted to BGR: {bgr.shape}")
    
    # Test quad detection
    cfg = Settings()
    print(f"⚙️ Using settings: ar_tol={cfg.ar_tol}, min_area_frac={cfg.min_area_frac}")
    
    quad_result = find_card_quad(bgr, cfg)

    if quad_result is not None:
        quad, det_score = quad_result
        print(f"✅ Found quad with corners:")
        for i, corner in enumerate(quad):
            print(f"   Corner {i}: ({corner[0]:.1f}, {corner[1]:.1f})")

        # Calculate area and aspect ratio
        area = cv2.contourArea(quad.astype(np.float32))
        H, W = bgr.shape[:2]
        area_frac = area / (H * W)

        # Calculate aspect ratio of detected quad
        pts = quad
        wA = np.linalg.norm(pts[2]-pts[3])
        wB = np.linalg.norm(pts[1]-pts[0])
        hA = np.linalg.norm(pts[1]-pts[2])
        hB = np.linalg.norm(pts[0]-pts[3])
        w = (wA+wB)/2.0
        h = (hA+hB)/2.0
        ar = w/h if h > 0 else 0

        print(f"📊 Quad stats:")
        print(f"   Detection score: {det_score:.3f}")
        print(f"   Area fraction: {area_frac:.3f}")
        print(f"   Aspect ratio: {ar:.3f} (target: 1.586)")
        print(f"   AR error: {abs(ar - 1.586)/1.586:.3f}")
        
        # Test full pipeline
        print("\n🔄 Testing full pipeline...")
        buf = io.BytesIO()
        pil_img.save(buf, format='JPEG')
        result = process_image_bytes(buf.getvalue(), cfg)
        
        print(f"📋 Pipeline result: {result['reason']}")
        if 'meta' in result:
            meta = result['meta']
            for key, value in meta.items():
                print(f"   {key}: {value}")
                
        return result
    else:
        print("❌ No quad detected")
        
        # Debug why no quad was found
        print("\n🔍 Debugging edge detection...")
        H, W = bgr.shape[:2]
        scale = cfg.max_side_px / max(H, W)
        img = cv2.resize(bgr, None, fx=scale, fy=scale) if scale < 1 else bgr.copy()
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        gray = cv2.GaussianBlur(gray,(5,5),0)

        v = float(np.median(gray))
        lo = int(max(0, cfg.canny_lo_mult * v))
        hi = int(min(255, cfg.canny_hi_mult * v))
        print(f"   Median gray value: {v:.1f}")
        print(f"   Canny thresholds: {lo} - {hi}")
        
        edges = cv2.Canny(gray, lo, hi)
        if cfg.dilate_iter > 0:
            edges = cv2.dilate(edges, np.ones((3,3),np.uint8), iterations=cfg.dilate_iter)
        
        cnts, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        print(f"   Found {len(cnts)} contours")
        
        if cnts:
            # Analyze contours
            img_area = img.shape[0] * img.shape[1]
            for i, c in enumerate(cnts[:5]):  # Check first 5 contours
                area = cv2.contourArea(c)
                peri = cv2.arcLength(c, True)
                approx = cv2.approxPolyDP(c, 0.02*peri, True)
                area_frac = area / img_area
                print(f"   Contour {i}: {len(approx)} points, area_frac={area_frac:.3f}")
        
        return {"ok": False, "reason": "no_quad"}

if __name__ == "__main__":
    print("🧪 Creating test ID card...")
    test_card = create_test_id_card()
    test_card.save("test_id_card.png")
    print("💾 Saved test card as test_id_card.png")
    
    print("\n" + "="*50)
    result = debug_quad_detection(test_card)
    print("="*50)
    
    if result.get("ok"):
        print("🎉 Success! The pipeline is working correctly.")
    else:
        print(f"⚠️ Pipeline returned: {result.get('reason')}")
        print("This might be normal for synthetic images. Try with real ID card photos.")
