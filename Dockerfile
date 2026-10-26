FROM python:3.13-slim

# System libraries required by opencv-python (GUI build) on slim images
RUN apt-get update && apt-get install -y --no-install-recommends libgl1 libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app
COPY pyproject.toml README.md ./
COPY src ./src
COPY configs ./configs

RUN pip install --no-cache-dir --extra-index-url https://download.pytorch.org/whl/cpu .

EXPOSE 8000 8501
