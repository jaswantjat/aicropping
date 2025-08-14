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
    ar_tol: float = 0.35          # allowed deviation around ID1_AR before warp (more lenient)
    ar_tol_after: float = 0.25    # allowed deviation after warp (more lenient)
    min_area_frac: float = 0.02   # quad must occupy >= 2% of image (lower threshold)
    max_area_frac: float = 0.99
    max_side_px: int = 1600       # downscale for speed before edge finding
    warp_w: int = 1024            # output width (height computed by AR) - DEPRECATED
    target_height: int = 600      # target height for output (width computed by AR)
    jpeg_quality: int = 92
    canny_lo_mult: float = 0.5    # lower threshold for better edge detection
    canny_hi_mult: float = 1.5    # higher threshold for better edge detection
    dilate_iter: int = 2          # more dilation to connect edges
    border_trim: int = 2          # pixels to trim after warp

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
    Deterministic point ordering that guarantees UL-UR-LR-LL order
    regardless of contour detection order.
    """
    pts = pts.reshape(4, 2)
    s = pts.sum(1)  # sum of x + y coordinates
    diff = np.diff(pts, axis=1)  # difference x - y coordinates

    # Top-left: smallest sum (x + y)
    tl = pts[np.argmin(s)]
    # Top-right: smallest difference (x - y)
    tr = pts[np.argmin(diff)]
    # Bottom-right: largest sum (x + y)
    br = pts[np.argmax(s)]
    # Bottom-left: largest difference (x - y)
    bl = pts[np.argmax(diff)]

    return np.array([tl, tr, br, bl], dtype="float32")

def find_card_quad(bgr: np.ndarray, cfg: Settings) -> Optional[np.ndarray]:
    H, W = bgr.shape[:2]
    scale = cfg.max_side_px / max(H, W)
    img = cv2.resize(bgr, None, fx=scale, fy=scale) if scale < 1 else bgr.copy()
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    gray = cv2.GaussianBlur(gray,(5,5),0)

    # Adaptive Canny thresholds based on median
    v = np.median(gray)
    lower = int(max(0, 0.66 * v))
    upper = int(min(255, 1.33 * v))
    edges = cv2.Canny(gray, lower, upper)
    if cfg.dilate_iter > 0:
        edges = cv2.dilate(edges, np.ones((3,3),np.uint8), iterations=cfg.dilate_iter)

    cnts,_ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not cnts: return None

    img_area = img.shape[0]*img.shape[1]

    # Keep only contours with ~ID-1 aspect ratio
    candidates = []
    for cnt in cnts:
        peri = cv2.arcLength(cnt, True)
        approx = cv2.approxPolyDP(cnt, 0.02*peri, True)
        if len(approx) == 4 and cv2.isContourConvex(approx):
            x, y, w, h = cv2.boundingRect(approx)
            if h <= 0: continue
            ar = w / float(h)
            area = cv2.contourArea(approx)

            # Strict aspect ratio filter for ID cards (1.45 < AR < 1.75)
            # and minimum area requirement (15% of frame)
            if (1.45 < ar < 1.75 and
                area > 0.15 * img_area and
                area <= cfg.max_area_frac * img_area):
                candidates.append((area, approx))

    if not candidates:
        # Fallback: try with more lenient aspect ratio
        for cnt in cnts:
            peri = cv2.arcLength(cnt, True)
            approx = cv2.approxPolyDP(cnt, 0.02*peri, True)
            if len(approx) == 4 and cv2.isContourConvex(approx):
                area = cv2.contourArea(approx)
                if (cfg.min_area_frac*img_area <= area <= cfg.max_area_frac*img_area):
                    pts = approx.reshape(4,2).astype("float32")
                    pts_o = order_pts(pts)
                    wA = np.linalg.norm(pts_o[2]-pts_o[3]); wB = np.linalg.norm(pts_o[1]-pts_o[0])
                    hA = np.linalg.norm(pts_o[1]-pts_o[2]); hB = np.linalg.norm(pts_o[0]-pts_o[3])
                    w = (wA+wB)/2.0; h = (hA+hB)/2.0
                    if h <= 0 or w <= 0: continue
                    ar = w/h
                    ar_err = abs(ar - ID1_AR) / ID1_AR
                    if ar_err <= cfg.ar_tol:
                        candidates.append((area, approx))

    if candidates:
        # Get largest valid quad
        _, best_approx = max(candidates, key=lambda x: x[0])
        pts = best_approx.reshape(4,2).astype("float32")
        return order_pts(pts) / (scale if scale < 1 else 1)

    # Final fallback: largest rotated box
    if cnts:
        c = max(cnts, key=cv2.contourArea)
        rect = cv2.minAreaRect(c)
        box = cv2.boxPoints(rect)
        return order_pts((box/(scale if scale<1 else 1)).astype("float32"))

    return None

def warp_card(bgr: np.ndarray, quad: np.ndarray, cfg: Settings) -> np.ndarray:
    # Use target_height and maintain exact ID-1 aspect ratio
    target_h = int(cfg.target_height)
    target_w = int(target_h * ID1_AR)  # maintain exact ratio

    dst = np.array([[0,0],[target_w-1,0],[target_w-1,target_h-1],[0,target_h-1]], dtype="float32")
    M = cv2.getPerspectiveTransform(quad, dst)
    warped = cv2.warpPerspective(bgr, M, (target_w, target_h), flags=cv2.INTER_CUBIC)

    if cfg.border_trim > 0 and warped.shape[0] > 2*cfg.border_trim and warped.shape[1] > 2*cfg.border_trim:
        bt = cfg.border_trim
        warped = warped[bt:-bt, bt:-bt]
    return warped

def orient_with_tesseract(bgr: np.ndarray) -> Tuple[np.ndarray, int, float]:
    try:
        rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)

        # Try OSD detection first
        try:
            osd = pytesseract.image_to_osd(rgb, output_type=Output.DICT)
            rot = int(osd.get("rotate", 0))
            conf = float(osd.get("orientation_confidence", 0.0))
        except:
            # Fallback: try string-based OSD parsing
            osd_str = pytesseract.image_to_osd(rgb)
            rot_match = re.search(r'Rotate: (\d+)', osd_str)
            rot = int(rot_match.group(1)) if rot_match else 0
            conf_match = re.search(r'Orientation confidence: ([\d.]+)', osd_str)
            conf = float(conf_match.group(1)) if conf_match else 0.0

        # Apply rotation correction
        if rot in (90, 180, 270):
            if imutils is not None:
                # Use imutils for better rotation (handles bounds properly)
                bgr = imutils.rotate_bound(bgr, -rot)  # negative to correct rotation
            else:
                # Fallback to cv2 rotation
                rot_map = {90: cv2.ROTATE_90_COUNTERCLOCKWISE,
                          180: cv2.ROTATE_180,
                          270: cv2.ROTATE_90_CLOCKWISE}
                bgr = cv2.rotate(bgr, rot_map[rot])

        return bgr, rot, conf
    except Exception as e:
        # If Tesseract fails completely, return original image
        return bgr, 0, 0.0

def blur_metric(bgr: np.ndarray) -> float:
    g = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
    return float(cv2.Laplacian(g, cv2.CV_64F).var())

def process_image_bytes(data: bytes, cfg: Settings) -> Dict[str, Any]:
    # Open via PIL to respect EXIF, then to BGR
    pil = _exif_correct(Image.open(io.BytesIO(data)))
    bgr = pil_to_bgr(pil)

    H,W = bgr.shape[:2]
    quad = find_card_quad(bgr, cfg)
    if quad is None:
        return {"ok": False, "reason": "no_quad", "meta": {"H":H,"W":W}}

    area = cv2.contourArea(quad.astype(np.float32))
    area_frac = float(area/(H*W))

    warped = warp_card(bgr, quad, cfg)
    ar_after = warped.shape[1]/warped.shape[0]
    ar_err_after = abs(ar_after - ID1_AR)/ID1_AR

    warped, rot, conf = orient_with_tesseract(warped)
    blur = blur_metric(warped)

    # gates
    if not (cfg.min_area_frac <= area_frac <= cfg.max_area_frac):
        return {"ok": False, "reason": "area_gate", "meta": {"area_frac": area_frac, "ar_after": ar_after, "rotate": rot, "rotate_conf": conf, "blur_var": blur}}
    if ar_err_after > cfg.ar_tol_after:
        return {"ok": False, "reason": "ar_gate", "meta": {"area_frac": area_frac, "ar_after": ar_after, "rotate": rot, "rotate_conf": conf, "blur_var": blur}}

    # encode
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
            "rotate_conf": conf,
            "blur_var": blur,
            "out_w": int(warped.shape[1]), "out_h": int(warped.shape[0])
        }
    }
