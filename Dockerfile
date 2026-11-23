FROM python:3.13-slim

# System libraries required by opencv-python (GUI build) on slim images
RUN apt-get update && apt-get install -y --no-install-recommends libgl1 libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app
COPY pyproject.toml README.md requirements.lock ./
COPY src ./src
COPY configs ./configs

RUN pip install --no-cache-dir -c requirements.lock --extra-index-url https://download.pytorch.org/whl/cpu -e .

# The image has no .git: the commit runs record is passed at build time, see README ("Docker")
ARG GIT_COMMIT=""
ENV RESILENS_GIT_COMMIT=$GIT_COMMIT

EXPOSE 8000 8501
