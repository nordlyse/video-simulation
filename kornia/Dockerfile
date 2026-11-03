FROM python:3.12-slim

# Source: https://github.com/kornia/kornia tag v0.9.0 (Apache-2.0).
# kornia_rs is Apache-2.0.
# torch and numpy are BSD-3-Clause and are required by Kornia.
# The CPU wheel avoids the larger CUDA build.
# MediaPipe is Apache-2.0. The selfie segmenter model is Apache-2.0.
# opencv-python-headless, absl-py and flatbuffers are Apache-2.0.
RUN pip install --no-cache-dir --index-url https://download.pytorch.org/whl/cpu torch \
    && pip install --no-cache-dir "numpy>=1.21" "kornia_rs>=0.2.0" \
    && pip install --no-cache-dir --no-deps \
        "kornia @ https://github.com/kornia/kornia/archive/refs/tags/v0.9.0.tar.gz" \
    && pip install --no-cache-dir --no-deps mediapipe==1.2.2 \
    && pip install --no-cache-dir opencv-python-headless absl-py flatbuffers \
    && mkdir -p /app \
    && python -c "import urllib.request; urllib.request.urlretrieve('https://storage.googleapis.com/mediapipe-models/image_segmenter/selfie_segmenter/float16/latest/selfie_segmenter.tflite', '/app/selfie_segmenter.tflite')"

# certifi is MPL-2.0 and matplotlib is BSD-style. MediaPipe imports both.
# sounddevice is MIT.
RUN pip install --no-cache-dir certifi matplotlib "sounddevice~=0.5"

# MediaPipe loads libEGL. Mesa's libegl1 is MIT.
RUN apt-get update \
    && apt-get install -y --no-install-recommends libegl1 libgles2 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app
COPY capture.py /app/capture.py

CMD ["python", "/app/capture.py"]
