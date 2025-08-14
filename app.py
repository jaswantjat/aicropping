import streamlit as st
import zipfile, io, time
import cv2
import numpy as np
from PIL import Image
from src.cropper import Settings, process_image_bytes

st.set_page_config(page_title="ID Card Cropper MVP", layout="wide")

st.title("Spanish ID Cropper MVP (Free stack)")
st.caption("OpenCV + Tesseract OSD. Drag-and-drop multiple photos at any angle and get normalized crops.")

with st.sidebar:
    st.header("Processing settings")

    # Preset configurations
    preset = st.selectbox("Preset Configuration", [
        "Default",
        "Low Contrast",
        "Textured Background",
        "Small Card",
        "Large Card",
        "Extreme Perspective",
        "Custom"
    ])

    # Set defaults based on preset (updated with improved parameters)
    if preset == "Low Contrast":
        default_ar_tol, default_min_area, default_dilate = 0.5, 0.06, 3
    elif preset == "Textured Background":
        default_ar_tol, default_min_area, default_dilate = 0.4, 0.08, 1
    elif preset == "Small Card":
        default_ar_tol, default_min_area, default_dilate = 0.6, 0.05, 2
    elif preset == "Large Card":
        default_ar_tol, default_min_area, default_dilate = 0.3, 0.15, 2
    elif preset == "Extreme Perspective":
        default_ar_tol, default_min_area, default_dilate = 0.8, 0.06, 4
    else:  # Default or Custom (updated with improved defaults)
        default_ar_tol, default_min_area, default_dilate = 0.6, 0.08, 2

    target_height = st.slider("Target height (px)", 400, 1200, 600, 50)
    ar_tol = st.slider("AR tolerance before warp", 0.05, 1.0, default_ar_tol, 0.01)
    ar_tol_after = st.slider("AR tolerance after warp", 0.05, 0.50, 0.4, 0.01)  # Updated default
    min_area = st.slider("Min area fraction", 0.001, 0.50, default_min_area, 0.001)
    max_area = st.slider("Max area fraction", 0.70, 1.00, 0.98, 0.01)  # Updated default
    dilate_iter = st.selectbox("Dilate iterations", [0,1,2,3,4], index=default_dilate)
    border_trim = st.number_input("Border trim (px)", 0, 10, 2)

    st.subheader("Advanced")
    with st.expander("Edge Detection"):
        canny_lo = st.slider("Canny low mult", 0.3, 1.0, 0.4, 0.01)  # Updated default
        canny_hi = st.slider("Canny high mult", 1.0, 2.5, 1.8, 0.01)  # Updated default

    with st.expander("Debug Options"):
        debug_mode = st.checkbox("Enable debug mode", help="Save failed images and show detailed debug info")
        save_intermediates = st.checkbox("Save intermediate images", help="Save edge detection steps for analysis")

    run_btn = st.button("🚀 Process Images", type="primary")

cfg = Settings(
    target_height=target_height, ar_tol=ar_tol, ar_tol_after=ar_tol_after,
    min_area_frac=min_area, max_area_frac=max_area,
    canny_lo_mult=canny_lo, canny_hi_mult=canny_hi,
    dilate_iter=int(dilate_iter), border_trim=int(border_trim),
    debug_mode=debug_mode
)

def create_thumbnail(image_data, max_size=300):
    """Create a thumbnail for preview to avoid UI freezing"""
    try:
        img = Image.open(io.BytesIO(image_data))
        img.thumbnail((max_size, max_size), Image.Resampling.LANCZOS)
        return img
    except:
        return None

files = st.file_uploader("📁 Upload one or more ID card images",
                        type=["jpg","jpeg","png","bmp","tif","tiff","webp"],
                        accept_multiple_files=True,
                        help="Drag and drop multiple photos at any angle")

if run_btn and files:
    with st.spinner(f"🔄 Processing {len(files)} images..."):
        results = []
        zip_buffer = io.BytesIO()

        # Create progress bar
        progress_bar = st.progress(0)
        status_text = st.empty()

        with zipfile.ZipFile(zip_buffer, "w", zipfile.ZIP_DEFLATED) as zf:
            for i, f in enumerate(files):
                status_text.text(f"Processing {f.name}...")
                progress_bar.progress((i + 1) / len(files))

                try:
                    data = f.read()
                    t0 = time.time()
                    res = process_image_bytes(data, cfg)
                    elapsed = int(1000*(time.time()-t0))

                    if res.get("ok"):
                        img_bytes = res["image_bytes"]
                        zf.writestr(f"{f.name.rsplit('.',1)[0]}_crop.jpg", img_bytes)
                    else:
                        # Save failed images for debugging
                        debug_filename = f"debug_failed_{f.name}"
                        with open(debug_filename, "wb") as debug_file:
                            debug_file.write(data)
                        st.info(f"💾 Saved failed image for debugging: {debug_filename}")

                    res["meta"] = res.get("meta", {})
                    res["meta"]["latency_ms"] = elapsed
                    results.append((f, res))

                except Exception as e:
                    st.error(f"❌ Crop failed for {f.name}: {str(e)}")
                    # Create a failed result entry
                    results.append((f, {
                        "ok": False,
                        "reason": f"exception: {str(e)}",
                        "meta": {"latency_ms": 0}
                    }))

        status_text.text("✅ Processing complete!")
        time.sleep(0.5)  # Brief pause to show completion
        progress_bar.empty()
        status_text.empty()

    # Results summary
    ok_count = sum(1 for _,r in results if r.get("ok"))
    fail_count = len(results) - ok_count

    if ok_count > 0:
        st.success(f"✅ Successfully processed {ok_count}/{len(results)} images")
    if fail_count > 0:
        st.warning(f"⚠️ Failed to process {fail_count} images")

    # Gallery with thumbnails
    cols = st.columns(2)

    with cols[0]:
        st.subheader("📥 Input Images")
        for f, r in results:
            # Create thumbnail for input preview
            f.seek(0)  # Reset file pointer
            thumbnail = create_thumbnail(f.read())
            if thumbnail:
                st.image(thumbnail, caption=f"📄 {f.name}", use_column_width=True)
            else:
                st.error(f"Could not load {f.name}")

    with cols[1]:
        st.subheader("📤 Cropped Results")
        for f, r in results:
            if r.get("ok"):
                meta = r.get('meta', {})
                caption = (f"✅ {f.name}\n"
                          f"AR: {meta.get('ar_after', 0):.3f} | "
                          f"Rotation: {meta.get('rotate', 0)}° | "
                          f"Det Score: {meta.get('det_score', 0):.2f} | "
                          f"OSD Conf: {meta.get('osd_conf', 0):.1f} | "
                          f"Time: {meta.get('latency_ms', 0)}ms")
                st.image(r["image_bytes"], caption=caption, use_column_width=True)
            else:
                reason = r.get('reason', 'unknown')
                st.error(f"❌ {f.name}: {reason}")

                # Show debug info and suggestions for failed images
                meta = r.get('meta', {})
                if meta:
                    with st.expander(f"🔍 Debug info for {f.name}"):
                        for key, value in meta.items():
                            if isinstance(value, float):
                                st.text(f"{key}: {value:.3f}")
                            else:
                                st.text(f"{key}: {value}")

                        # Suggest fixes based on failure reason
                        st.subheader("💡 Suggested Fixes:")
                        if reason == "no_quad":
                            st.write("**Card detection failed. Try:**")
                            st.write("• Use 'Low Contrast' preset if card/background are similar colors")
                            st.write("• Use 'Small Card' preset if card appears small in image")
                            st.write("• Use 'Textured Background' preset for complex backgrounds")
                            st.write("• Enable debug mode to save failed images for analysis")
                        elif reason == "area_gate":
                            area_frac = meta.get('area_frac', 0)
                            if area_frac > 0.8:
                                st.write("**Card too large. Try:**")
                                st.write("• Use 'Large Card' preset")
                                st.write("• Increase max area fraction")
                            else:
                                st.write("**Card too small. Try:**")
                                st.write("• Use 'Small Card' preset")
                                st.write("• Decrease min area fraction")
                        elif reason == "ar_gate":
                            ar_after = meta.get('ar_after', 0)
                            st.write(f"**Aspect ratio {ar_after:.3f} too far from 1.586. Try:**")
                            st.write("• Use 'Extreme Perspective' preset for angled shots")
                            st.write("• Increase AR tolerance settings")

    # Download section
    st.divider()
    col1, col2 = st.columns(2)

    with col1:
        zip_buffer.seek(0)
        st.download_button(
            "📦 Download all crops as ZIP",
            data=zip_buffer,
            file_name="id_crops.zip",
            mime="application/zip",
            type="primary"
        )

    with col2:
        if ok_count > 0:
            st.metric("Success Rate", f"{ok_count}/{len(results)}", f"{100*ok_count/len(results):.1f}%")
