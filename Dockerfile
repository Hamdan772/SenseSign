FROM python:3.11-slim

# System deps for OpenCV + MediaPipe
RUN apt-get update && apt-get install -y --no-install-recommends \
        libgl1 libglib2.0-0 libsm6 libxrender1 libxext6 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

COPY requirements-cloud.txt .
RUN pip install --no-cache-dir -r requirements-cloud.txt

COPY . .

# Render (and most platforms) inject PORT env var
ENV PORT=5000
EXPOSE ${PORT}

# 1 worker + 4 threads: less memory, faster cold start on small instances
CMD ["sh", "-c", "python -m gunicorn --bind 0.0.0.0:${PORT} --workers 1 --threads 4 --timeout 120 web_app:app"]
