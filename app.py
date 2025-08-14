import streamlit as st
import zipfile, io, time
from src.cropper import Settings, process_image_bytes

st.set_page_config(page_title="ID Card Cropper MVP", layout="wide")

st.title("Spanish ID Cropper MVP (Free stack)")
st.caption("OpenCV + Tesseract OSD. Drag-and-drop multiple photos at any angle and get normalized crops.")

with st.sidebar:
    st.header("Processing settings")
    warp_w = st.slider("Output width (px)", 640, 1600, 1024, 32)
    ar_tol = st.slider("AR tolerance before warp", 0.05, 0.30, 0.18, 0.01)
    ar_tol_after = st.slider("AR tolerance after warp", 0.05, 0.30, 0.12, 0.01)
    min_area = st.slider("Min area fraction", 0.01, 0.20, 0.06, 0.01)
    max_area = st.slider("Max area fraction", 0.70, 1.00, 0.99, 0.01)
    canny_lo = st.slider("Canny low mult", 0.3, 1.0, 0.66, 0.01)
    canny_hi = st.slider("Canny high mult", 1.0, 2.5, 1.33, 0.01)
    dilate_iter = st.selectbox("Dilate iterations", [0,1,2], index=1)
    border_trim = st.number_input("Border trim (px)", 0, 10, 2)
    run_btn = st.button("Process")

cfg = Settings(
    warp_w=warp_w, ar_tol=ar_tol, ar_tol_after=ar_tol_after,
    min_area_frac=min_area, max_area_frac=max_area,
    canny_lo_mult=canny_lo, canny_hi_mult=canny_hi,
    dilate_iter=int(dilate_iter), border_trim=int(border_trim)
)

files = st.file_uploader("Upload one or more images", type=["jpg","jpeg","png","bmp","tif","tiff","webp"], accept_multiple_files=True)

if run_btn and files:
    results = []
    zip_buffer = io.BytesIO()
    with zipfile.ZipFile(zip_buffer, "w", zipfile.ZIP_DEFLATED) as zf:
        for f in files:
            data = f.read()
            t0 = time.time()
            res = process_image_bytes(data, cfg)
            elapsed = int(1000*(time.time()-t0))
            if res.get("ok"):
                img_bytes = res["image_bytes"]
                zf.writestr(f"{f.name.rsplit('.',1)[0]}_crop.jpg", img_bytes)
            res["meta"] = res.get("meta", {})
            res["meta"]["latency_ms"] = elapsed
            results.append((f, res))

    # gallery
    cols = st.columns(2)
    ok_count = sum(1 for _,r in results if r.get("ok"))
    st.success(f"Processed {len(results)} images. OK: {ok_count}, Fail: {len(results)-ok_count}")

    with cols[0]:
        st.subheader("Inputs")
        for f, r in results:
            st.image(f, caption=f.name, use_column_width=True)

    with cols[1]:
        st.subheader("Outputs")
        for f, r in results:
            if r.get("ok"):
                st.image(r["image_bytes"], caption=f"{f.name} ✓  AR {r['meta'].get('ar_after'):.3f}  rot {r['meta'].get('rotate')}° conf {r['meta'].get('rotate_conf',0):.1f}", use_column_width=True)
            else:
                st.error(f"{f.name}: {r.get('reason')}")

    # download all
    zip_buffer.seek(0)
    st.download_button("Download all crops as ZIP", data=zip_buffer, file_name="id_crops.zip", mime="application/zip")
