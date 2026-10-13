FROM python:3.13-slim

WORKDIR /app
COPY pyproject.toml README.md ./
COPY src ./src
COPY configs ./configs

RUN pip install --no-cache-dir --extra-index-url https://download.pytorch.org/whl/cpu .

EXPOSE 8000 8501
