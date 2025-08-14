import cv2, numpy as np, io, os, re
from dataclasses import dataclass
from typing import Optional, Dict, Any, Tuple
from PIL import Image, ImageOps
import pytesseract
from pytesseract import Output
try:
    import imutils
except ImportError:
    imutils = None

ID1_AR = 85.60/53.98  # ~1.586

@dataclass
class Settings:
    ar_tol: float = 0.6           # allowed deviation around ID1_AR before warp (more lenient for perspective)
    ar_tol_after: float = 0.4     # allowed deviation after warp (more lenient after warping)
    min_area_frac: float = 0.005  # quad must occupy >= 0.5% of image (detect smaller cards)
    max_area_frac: float = 0.98   # allow larger cards
    max_side_px: int = 1600       # downscale for speed before edge finding
    target_height: int = 600      # target height for output (width computed by AR)
    jpeg_quality: int = 92
    canny_lo_mult: float = 0.4    # lower threshold for better edge detection
    canny_hi_mult: float = 1.8    # higher threshold for better edge detection
    dilate_iter: int = 2          # more dilation to connect edges
    border_trim: int = 2          # pixels to trim after warp
    debug_mode: bool = False      # enable debug visualization

def _exif_correct(pil_img: Image.Image) -> Image.Image:
    # respect EXIF orientation
    return ImageOps.exif_transpose(pil_img)

def pil_to_bgr(pil_img: Image.Image) -> np.ndarray:
    rgb = np.array(pil_img.convert("RGB"))
    return cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)

def bgr_to_pil(bgr: np.ndarray) -> Image.Image:
    rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
    return Image.fromarray(rgb)

def order_pts(pts: np.ndarray) -> np.ndarray:
    """
    Deterministic point ordering that guarantees TL-TR-BR-BL order
    regardless of contour detection order. Critical for perspective transform.
    """
    pts = pts.reshape(4, 2)
    s = pts.sum(axis=1)  # sum of x + y coordinates
    d = np.diff(pts, axis=1)[:, 0]  # difference x - y coordinates

    # Top-left: smallest sum (x + y)
    tl = pts[np.argmin(s)]
    # Bottom-right: largest sum (x + y)
    br = pts[np.argmax(s)]
    # Top-right: smallest difference (x - y)
    tr = pts[np.argmin(d)]
    # Bottom-left: largest difference (x - y)
    bl = pts[np.argmax(d)]

    return np.array([tl, tr, br, bl], dtype="float32")

def find_card_by_color(bgr: np.ndarray) -> np.ndarray:
    """
    Spanish ID cards have distinctive white/light background.
    This creates a mask for potential card regions based on color.
    """
    hsv = cv2.cvtColor(bgr, cv2.COLOR_BGR2HSV)

    # Detect white/light regions (typical for ID cards)
    lower_white = np.array([0, 0, 180])    # Low saturation, high value
    upper_white = np.array([180, 50, 255]) # Any hue, low saturation, high value
    white_mask = cv2.inRange(hsv, lower_white, upper_white)

    # Clean up the mask
    kernel = np.ones((5,5), np.uint8)
    white_mask = cv2.morphologyEx(white_mask, cv2.MORPH_CLOSE, kernel)
    white_mask = cv2.morphologyEx(white_mask, cv2.MORPH_OPEN, kernel)

    return white_mask

def warp_to_id1_canvas(bgr: np.ndarray, box_pts: np.ndarray, width: int = 850) -> np.ndarray:
    """
    Warp detected card region to exact ID-1 canvas using 4-point perspective transform.
    This removes skew, fills frame with card, and creates proper top-down scan.
    """
    src = order_pts(box_pts.astype(np.float32))
    height = int(round(width / ID1_AR))  # Exact ID-1 aspect ratio

    # Destination points for perfect rectangle
    dst = np.array([
        [0, 0],                    # top-left
        [width-1, 0],              # top-right
        [width-1, height-1],       # bottom-right
        [0, height-1]              # bottom-left
    ], dtype=np.float32)

    # Compute perspective transform matrix
    M = cv2.getPerspectiveTransform(src, dst)

    # Apply transform with high-quality interpolation
    warped = cv2.warpPerspective(bgr, M, (width, height), flags=cv2.INTER_CUBIC)

    return warped

def find_card_quad_simple(bgr: np.ndarray, cfg: Settings) -> Optional[Tuple[np.ndarray, float]]:
    """
    Simple, robust card detection - the original working method.
    """
    H, W = bgr.shape[:2]
    scale = cfg.max_side_px / max(H, W)
    img = cv2.resize(bgr, None, fx=scale, fy=scale) if scale < 1 else bgr.copy()
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    gray = cv2.GaussianBlur(gray, (5, 5), 0)

    # Simple adaptive Canny thresholds based on median
    v = np.median(gray)
    lower = int(max(0, 0.66 * v))
    upper = int(min(255, 1.33 * v))
    edges = cv2.Canny(gray, lower, upper)

    if cfg.dilate_iter > 0:
        edges = cv2.dilate(edges, np.ones((3,3),np.uint8), iterations=cfg.dilate_iter)

    cnts, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not cnts:
        return None

    img_area = img.shape[0] * img.shape[1]
    candidates = []

    # Try approxPolyDP method first (original working approach)
    for cnt in cnts:
        peri = cv2.arcLength(cnt, True)
        approx = cv2.approxPolyDP(cnt, 0.02*peri, True)
        if len(approx) == 4 and cv2.isContourConvex(approx):
            area = cv2.contourArea(approx)
            area_frac = area / img_area

            if area_frac > cfg.min_area_frac and area_frac <= cfg.max_area_frac:
                # Calculate aspect ratio
                pts = approx.reshape(4,2).astype("float32")
                pts_o = order_pts(pts)
                wA = np.linalg.norm(pts_o[2]-pts_o[3]); wB = np.linalg.norm(pts_o[1]-pts_o[0])
                hA = np.linalg.norm(pts_o[1]-pts_o[2]); hB = np.linalg.norm(pts_o[0]-pts_o[3])
                w = (wA+wB)/2.0; h = (hA+hB)/2.0
                if h <= 0 or w <= 0: continue
                ar = w/h
                ar_err = abs(ar - ID1_AR) / ID1_AR

                if ar_err <= cfg.ar_tol:
                    score = 2.0 * area_frac - 0.6 * ar_err
                    candidates.append((score, pts_o, area_frac))

    if candidates:
        best_score, best_pts, best_area_frac = max(candidates, key=lambda x: x[0])
        scaled_pts = best_pts / (scale if scale < 1 else 1)
        return scaled_pts.astype("float32"), best_score

    return None

def find_card_quad_advanced(bgr: np.ndarray, cfg: Settings) -> Optional[Tuple[np.ndarray, float]]:
    """
    Advanced card detection with CLAHE and minAreaRect - for difficult cases.
    """
    H, W = bgr.shape[:2]
    scale = cfg.max_side_px / max(H, W)
    img = cv2.resize(bgr, None, fx=scale, fy=scale) if scale < 1 else bgr.copy()

    # CLAHE preprocessing on L channel for textured backgrounds
    lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
    l_channel = lab[:, :, 0]

    # Apply CLAHE (Contrast Limited Adaptive Histogram Equalization)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    l_channel = clahe.apply(l_channel)

    # Convert back to grayscale for edge detection
    gray = l_channel
    gray = cv2.GaussianBlur(gray, (5, 5), 0)

    # Adaptive Canny thresholds based on median
    v = np.median(gray)
    lower = int(max(0, 0.66 * v))
    upper = int(min(255, 1.33 * v))
    edges = cv2.Canny(gray, lower, upper)
    if cfg.dilate_iter > 0:
        edges = cv2.dilate(edges, np.ones((3,3),np.uint8), iterations=cfg.dilate_iter)

    cnts, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not cnts:
        return None

    img_area = img.shape[0] * img.shape[1]
    candidates = []

    # Score candidates using minAreaRect for better rotated rectangle handling
    for cnt in cnts:
        area = cv2.contourArea(cnt)
        if area < cfg.min_area_frac * img_area:
            continue

        # Get minimum area rectangle (handles rotation better)
        rect = cv2.minAreaRect(cnt)
        box = cv2.boxPoints(rect)
        box_area = cv2.contourArea(box)

        if box_area <= 0:
            continue

        # Calculate aspect ratio from rotated rectangle
        (_, _), (w, h), angle = rect
        ar = max(w, h) / min(w, h) if min(w, h) > 0 else 0

        # Handle both landscape and portrait orientations
        # Spanish ID cards are typically landscape (1.586), but check both orientations
        ar_error_landscape = abs(ar - ID1_AR) / ID1_AR
        ar_error_portrait = abs(ar - (1/ID1_AR)) / (1/ID1_AR)
        ar_error = min(ar_error_landscape, ar_error_portrait)

        # Score based on multiple factors
        area_frac = area / img_area
        rectangularity = area / box_area  # How rectangular is the contour

        # Composite score: prefer large, rectangular, ID-1 aspect ratio
        score = (
            2.0 * area_frac +           # Size importance
            1.0 * rectangularity +      # Shape regularity
            -1.0 * ar_error            # Aspect ratio closeness
        )

        # Filter candidates
        if (1.4 < ar < 1.8 and                    # Reasonable aspect ratio range
            area_frac > 0.05 and                  # Minimum size
            area_frac <= cfg.max_area_frac and    # Maximum size
            rectangularity > 0.7):               # Must be reasonably rectangular

            candidates.append((score, box, area_frac))

    if candidates:
        # Get best candidate by score
        best_score, best_box, best_area_frac = max(candidates, key=lambda x: x[0])
        # Scale back to original image coordinates
        scaled_box = best_box / (scale if scale < 1 else 1)
        return order_pts(scaled_box.astype("float32")), best_score

    return None

def find_card_quad_color_assisted(bgr: np.ndarray, cfg: Settings) -> Optional[Tuple[np.ndarray, float]]:
    """
    Color-assisted detection as a last resort fallback.
    Uses color information to guide edge detection.
    """
    # Get color mask for potential card regions
    color_mask = find_card_by_color(bgr)

    # Use color mask to guide contour detection
    H, W = bgr.shape[:2]
    scale = cfg.max_side_px / max(H, W)
    img = cv2.resize(bgr, None, fx=scale, fy=scale) if scale < 1 else bgr.copy()
    mask_scaled = cv2.resize(color_mask, (img.shape[1], img.shape[0])) if scale < 1 else color_mask

    # Find contours in the color mask
    cnts, _ = cv2.findContours(mask_scaled, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not cnts:
        return None

    img_area = img.shape[0] * img.shape[1]
    candidates = []

    for cnt in cnts:
        area = cv2.contourArea(cnt)
        if area < cfg.min_area_frac * img_area:
            continue

        # Get minimum area rectangle
        rect = cv2.minAreaRect(cnt)
        box = cv2.boxPoints(rect)
        box_area = cv2.contourArea(box)

        if box_area <= 0:
            continue

        # Calculate metrics
        (_, _), (w, h), angle = rect
        ar = max(w, h) / min(w, h) if min(w, h) > 0 else 0
        area_frac = area / img_area
        rectangularity = area / box_area

        # More lenient scoring for color-based detection
        ar_error_landscape = abs(ar - ID1_AR) / ID1_AR
        ar_error_portrait = abs(ar - (1/ID1_AR)) / (1/ID1_AR)
        ar_error = min(ar_error_landscape, ar_error_portrait)

        score = (
            1.5 * area_frac +           # Size importance
            0.8 * rectangularity +      # Shape regularity
            -0.5 * ar_error            # Aspect ratio closeness (less strict)
        )

        # Very lenient filtering for color-based method
        if (0.5 < ar < 3.0 and                      # Very wide aspect ratio range
            area_frac > cfg.min_area_frac and       # Minimum size
            area_frac <= cfg.max_area_frac and      # Maximum size
            rectangularity > 0.5):                 # Less strict rectangularity

            candidates.append((score, box, area_frac))

    if candidates:
        best_score, best_box, best_area_frac = max(candidates, key=lambda x: x[0])
        scaled_box = best_box / (scale if scale < 1 else 1)
        return order_pts(scaled_box.astype("float32")), best_score

    return None

def find_card_quad(bgr: np.ndarray, cfg: Settings, debug: bool = False) -> Optional[Tuple[np.ndarray, float]]:
    """
    Triple-fallback card detection strategy: simple -> advanced -> color-assisted.
    This provides maximum robustness across different image conditions.
    """
    # Strategy 1: Simple threshold (good for high-contrast, clean backgrounds)
    result = find_card_quad_simple(bgr, cfg)
    if result is not None:
        quad, score = result
        if debug:
            print(f"✅ Simple method succeeded with score: {score:.3f}")
        return result

    # Strategy 2: Advanced method (good for textured backgrounds, shadows)
    result = find_card_quad_advanced(bgr, cfg)
    if result is not None:
        quad, score = result
        if debug:
            print(f"✅ Advanced method succeeded with score: {score:.3f}")
        return result

    # Strategy 3: Color-assisted method (last resort for difficult cases)
    result = find_card_quad_color_assisted(bgr, cfg)
    if result is not None:
        quad, score = result
        if debug:
            print(f"✅ Color-assisted method succeeded with score: {score:.3f}")
        return result

    if debug:
        print("❌ All three methods failed to find card")

    return None

def warp_card(bgr: np.ndarray, quad: np.ndarray, cfg: Settings) -> np.ndarray:
    """
    Warp card to ID-1 canvas. Now uses proper perspective transform for document scanning.
    """
    # Use configurable width, compute height for exact ID-1 ratio
    target_w = int(cfg.target_height * ID1_AR)  # Width from height to maintain AR

    # Use the new perspective warp function
    warped = warp_to_id1_canvas(bgr, quad, target_w)

    # Optional border trimming
    if cfg.border_trim > 0 and warped.shape[0] > 2*cfg.border_trim and warped.shape[1] > 2*cfg.border_trim:
        bt = cfg.border_trim
        warped = warped[bt:-bt, bt:-bt]

    return warped

def orient_with_osd(img_bgr: np.ndarray, min_width: int = 1000, conf_thresh: float = 5.0) -> Tuple[np.ndarray, int, float]:
    """
    Enhanced OSD with proper upscaling, binarization, and confidence parsing.
    Returns (corrected_image, rotation_degrees, confidence_score)
    """
    try:
        # Convert to grayscale
        g = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)

        # Upscale if too small (OSD works better on larger images)
        if g.shape[1] < min_width:
            scale = float(min_width) / g.shape[1]
            new_width = min_width
            new_height = int(g.shape[0] * scale)
            g = cv2.resize(g, (new_width, new_height), interpolation=cv2.INTER_CUBIC)

        # Binarize with Otsu thresholding for better OSD accuracy
        _, g = cv2.threshold(g, 0, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)

        # Run OSD with PSM 0 (orientation and script detection only)
        osd_output = pytesseract.image_to_osd(g, config="--psm 0")

        # Parse rotation and confidence from output
        rot_match = re.search(r"Rotate:\s+(\d+)", osd_output)
        conf_match = re.search(r"Orientation confidence:\s+([\d.]+)", osd_output)

        rot = int(rot_match.group(1)) if rot_match else 0
        conf = float(conf_match.group(1)) if conf_match else 0.0

        # Apply rotation correction only if confident
        if conf >= conf_thresh and rot in (90, 180, 270):
            if rot == 90:
                img_bgr = cv2.rotate(img_bgr, cv2.ROTATE_90_COUNTERCLOCKWISE)
            elif rot == 180:
                img_bgr = cv2.rotate(img_bgr, cv2.ROTATE_180)
            elif rot == 270:
                img_bgr = cv2.rotate(img_bgr, cv2.ROTATE_90_CLOCKWISE)

        return img_bgr, rot, conf

    except Exception as e:
        # If OSD fails completely, return original image with zero confidence
        return img_bgr, 0, 0.0

def blur_metric(bgr: np.ndarray) -> float:
    g = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
    return float(cv2.Laplacian(g, cv2.CV_64F).var())

def process_image_bytes(data: bytes, cfg: Settings) -> Dict[str, Any]:
    # Open via PIL to respect EXIF, then to BGR
    pil = _exif_correct(Image.open(io.BytesIO(data)))
    bgr = pil_to_bgr(pil)

    H, W = bgr.shape[:2]
    quad_result = find_card_quad(bgr, cfg)
    if quad_result is None:
        return {"ok": False, "reason": "no_quad", "meta": {"H": H, "W": W}}

    quad, det_score = quad_result
    area = cv2.contourArea(quad.astype(np.float32))
    area_frac = float(area / (H * W))

    # Warp to proper ID-1 canvas (removes skew and background)
    warped = warp_card(bgr, quad, cfg)
    ar_after = warped.shape[1] / warped.shape[0]
    ar_err_after = abs(ar_after - ID1_AR) / ID1_AR

    # Enhanced OSD with real confidence
    warped, rot, osd_conf = orient_with_osd(warped)
    blur = blur_metric(warped)

    # Quality gates
    if not (cfg.min_area_frac <= area_frac <= cfg.max_area_frac):
        return {
            "ok": False,
            "reason": "area_gate",
            "meta": {
                "area_frac": area_frac,
                "ar_after": ar_after,
                "rotate": rot,
                "det_score": det_score,
                "osd_conf": osd_conf,
                "blur_var": blur
            }
        }

    if ar_err_after > cfg.ar_tol_after:
        return {
            "ok": False,
            "reason": "ar_gate",
            "meta": {
                "area_frac": area_frac,
                "ar_after": ar_after,
                "rotate": rot,
                "det_score": det_score,
                "osd_conf": osd_conf,
                "blur_var": blur
            }
        }

    # Encode final result
    ok, buf = cv2.imencode(".jpg", warped, [int(cv2.IMWRITE_JPEG_QUALITY), int(cfg.jpeg_quality)])
    if not ok:
        return {"ok": False, "reason": "encode_fail", "meta": {}}

    return {
        "ok": True,
        "reason": "ok",
        "image_bytes": buf.tobytes(),
        "meta": {
            "area_frac": area_frac,
            "ar_after": ar_after,
            "rotate": rot,
            "det_score": det_score,      # Detection quality score
            "osd_conf": osd_conf,        # OSD rotation confidence
            "blur_var": blur,
            "out_w": int(warped.shape[1]),
            "out_h": int(warped.shape[0])
        }
    }
