import cv2, numpy as np, io, os
from dataclasses import dataclass
from typing import Optional, Dict, Any, Tuple
from PIL import Image, ImageOps
import pytesseract
from pytesseract import Output

ID1_AR = 85.60/53.98  # ~1.586

@dataclass
class Settings:
    ar_tol: float = 0.35          # allowed deviation around ID1_AR before warp (more lenient)
    ar_tol_after: float = 0.25    # allowed deviation after warp (more lenient)
    min_area_frac: float = 0.02   # quad must occupy >= 2% of image (lower threshold)
    max_area_frac: float = 0.99
    max_side_px: int = 1600       # downscale for speed before edge finding
    warp_w: int = 1024            # output width (height computed by AR)
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
    s = pts.sum(axis=1)
    diff = np.diff(pts, axis=1).reshape(-1)
    tl = pts[np.argmin(s)]
    tr = pts[np.argmin(diff)]
    br = pts[np.argmax(s)]
    bl = pts[np.argmax(diff)]
    return np.array([tl,tr,br,bl], dtype="float32")

def find_card_quad(bgr: np.ndarray, cfg: Settings) -> Optional[np.ndarray]:
    H, W = bgr.shape[:2]
    scale = cfg.max_side_px / max(H, W)
    img = cv2.resize(bgr, None, fx=scale, fy=scale) if scale < 1 else bgr.copy()
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    gray = cv2.GaussianBlur(gray,(5,5),0)

    v = float(np.median(gray))
    lo = int(max(0, cfg.canny_lo_mult * v))
    hi = int(min(255, cfg.canny_hi_mult * v))
    edges = cv2.Canny(gray, lo, hi)
    if cfg.dilate_iter > 0:
        edges = cv2.dilate(edges, np.ones((3,3),np.uint8), iterations=cfg.dilate_iter)

    cnts,_ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not cnts: return None

    img_area = img.shape[0]*img.shape[1]
    best, best_score = None, -1e9
    for c in cnts:
        peri = cv2.arcLength(c, True)
        approx = cv2.approxPolyDP(c, 0.02*peri, True)
        if len(approx) == 4 and cv2.isContourConvex(approx):
            area = cv2.contourArea(approx)
            if not (cfg.min_area_frac*img_area <= area <= cfg.max_area_frac*img_area):
                continue
            pts = approx.reshape(4,2).astype("float32")
            pts_o = order_pts(pts)
            wA = np.linalg.norm(pts_o[2]-pts_o[3]); wB = np.linalg.norm(pts_o[1]-pts_o[0])
            hA = np.linalg.norm(pts_o[1]-pts_o[2]); hB = np.linalg.norm(pts_o[0]-pts_o[3])
            w = (wA+wB)/2.0; h = (hA+hB)/2.0
            if h <= 0 or w <= 0: continue
            ar = w/h
            ar_err = abs(ar - ID1_AR) / ID1_AR
            area_frac = area/img_area
            score = 2.0*area_frac - 0.6*ar_err  # prefer big + near AR
            if ar_err <= cfg.ar_tol and score > best_score:
                best, best_score = pts_o/(scale if scale<1 else 1), score

    if best is not None:
        return best

    # Fallback: largest rotated box
    c = max(cnts, key=cv2.contourArea)
    rect = cv2.minAreaRect(c)
    box = cv2.boxPoints(rect)
    return order_pts((box/(scale if scale<1 else 1)).astype("float32"))

def warp_card(bgr: np.ndarray, quad: np.ndarray, cfg: Settings) -> np.ndarray:
    out_w = int(cfg.warp_w)
    out_h = int(round(out_w / ID1_AR))
    dst = np.array([[0,0],[out_w-1,0],[out_w-1,out_h-1],[0,out_h-1]], dtype="float32")
    M = cv2.getPerspectiveTransform(quad, dst)
    warped = cv2.warpPerspective(bgr, M, (out_w, out_h), flags=cv2.INTER_CUBIC)
    if cfg.border_trim > 0 and warped.shape[0] > 2*cfg.border_trim and warped.shape[1] > 2*cfg.border_trim:
        bt = cfg.border_trim
        warped = warped[bt:-bt, bt:-bt]
    return warped

def orient_with_tesseract(bgr: np.ndarray) -> Tuple[np.ndarray, int, float]:
    try:
        rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
        osd = pytesseract.image_to_osd(rgb, output_type=Output.DICT)
        rot = int(osd.get("rotate", 0))
        conf = float(osd.get("orientation_confidence", 0.0))
        if rot in (90,180,270):
            rot_map = {90: cv2.ROTATE_90_CLOCKWISE, 180: cv2.ROTATE_180, 270: cv2.ROTATE_90_COUNTERCLOCKWISE}
            bgr = cv2.rotate(bgr, rot_map[rot])
        return bgr, rot, conf
    except Exception:
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
