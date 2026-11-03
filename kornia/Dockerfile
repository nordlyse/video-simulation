FROM python:3.12-slim

# Source: https://github.com/kornia/kornia tag v0.9.0 (Apache-2.0).
# kornia_rs is Apache-2.0.
# torch and numpy are BSD-3-Clause and are required by Kornia.
# The CPU wheel avoids the larger CUDA build.
RUN pip install --no-cache-dir --index-url https://download.pytorch.org/whl/cpu torch \
    && pip install --no-cache-dir "numpy>=1.21" "kornia_rs>=0.2.0" \
    && pip install --no-cache-dir --no-deps \
        "kornia @ https://github.com/kornia/kornia/archive/refs/tags/v0.9.0.tar.gz"

WORKDIR /app
COPY capture.py /app/capture.py

CMD ["python", "/app/capture.py"]
