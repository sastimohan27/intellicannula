FROM python:3.12-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1

WORKDIR /vein_module

# OpenCV needs libGL / libglib even in headless containers.
RUN apt-get update && apt-get install -y --no-install-recommends \
        libgl1 libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY src ./src

# Not a long-running service in Phase 1 - this image is used for training and
# batch inference. Phase 6 wraps it behind the backend API instead.
CMD ["python", "-c", "import torch; print('vein_module ready, torch', torch.__version__)"]
