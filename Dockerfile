# Use Python 3.11 slim
FROM python:3.11-slim

# Set environment variables
ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1
ENV PYTHONPATH=/app

# Set working directory
WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    libgl1 \
    libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

# Install standard packages
# Added --upgrade pip for stability
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir \
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

# --- CRITICAL STEP ---
# Copy everything from your local directory into /app in the image
COPY . /app/

# List files during build to verify they are actually there (Check your build logs!)
RUN ls -la /app

# Check if model exists; if not, train it.
# Ensure train.py exists in your local folder!
RUN if [ -f "train.py" ]; then \
        if [ ! -f "mnist_model.pth" ]; then python train.py; fi; \
    else \
        echo "Warning: train.py not found, skipping training step"; \
    fi

# Note: Only one CMD can run per container. 
# Usually, you'd override this in docker-compose for each service.
EXPOSE 8000
EXPOSE 8501
