FROM python:3.11-slim

# System deps
RUN apt-get update && apt-get install -y --no-install-recommends \
    tesseract-ocr libglib2.0-0 libgl1 && \
    rm -rf /var/lib/apt/lists/*

# Python deps
WORKDIR /app
COPY requirements.txt /app/requirements.txt
RUN pip install --no-cache-dir -r requirements.txt

# App
COPY src /app/src
COPY app.py /app/app.py

# Default: run CLI; override to run Streamlit
ENTRYPOINT ["python", "-m", "src.cli"]
