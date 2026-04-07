# Use Python 3.11 slim
FROM python:3.11-slim

ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1

WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    libgl1 \
    libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

# Install standard packages
RUN pip install --no-cache-dir \
    streamlit \
    streamlit-drawable-canvas \
    fastapi \
    uvicorn \
    pillow \
    requests

# Install Torch (CPU version)
RUN pip install --no-cache-dir \
    torch \
    torchvision \
    --index-url https://download.pytorch.org/whl/cpu

# Copy all files (including model.py and train.py)
COPY . .

# Check if model exists; if not, train it.
RUN if [ ! -f "mnist_model.pth" ]; then python train.py; fi

# Expose ports
EXPOSE 8000
EXPOSE 8501