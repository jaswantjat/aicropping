# Spanish ID Cropper MVP (Free)

- Pipeline: OpenCV edges/contours → quad → perspective warp → Tesseract OSD orientation
- Aspect ratio prior: ID-1 (85.60 x 53.98 mm) ≈ 1.586
- Outputs: normalized crops, JSONL log (CLI), Streamlit UI for testing

## Local
- macOS: `brew install tesseract`
- Ubuntu: `sudo apt-get install -y tesseract-ocr`
- Python: `make install`

Run CLI:
```
python -m src.cli --inp raw --out out --workers 8
```

Run UI:
```
streamlit run app.py
```

## Docker
```
make docker-cli
# or UI
make docker-ui
```

## Notes
- All deps are permissive/free.
- If cluttered backgrounds hurt recall, consider a tiny rotated detector (MMRotate) as a ROI provider later.
