FROM python:3.11-slim

ENV PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    HF_HOME=/opt/hf \
    YOLO_CONFIG_DIR=/tmp/ultralytics

RUN apt-get update && apt-get install -y --no-install-recommends \
        ffmpeg libgl1 libglib2.0-0 fonts-dejavu-core \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# CPU-only torch keeps the image small (no CUDA wheels)
RUN pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu \
 && python -c "from torchvision.ops import nms; print('torchvision OK')"
COPY requirements.txt .
RUN pip install -r requirements.txt

# Bake model weights into the image so runs work offline and start fast
RUN python -c "from ultralytics import YOLO; YOLO('yolov8n.pt')" \
 && python -c "from huggingface_hub import snapshot_download; snapshot_download('HuggingFaceTB/SmolVLM-256M-Instruct')" \
 && chmod -R a+rX /opt/hf /app

COPY main.py .
COPY pipeline ./pipeline

ENTRYPOINT ["python", "main.py"]
CMD ["--input", "/data/video.mp4", "--out", "/data/results"]

