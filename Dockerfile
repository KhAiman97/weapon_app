FROM python:3.11-slim

# Set working directory
WORKDIR /app

# Install system dependencies and clean up in one layer
RUN apt-get update && apt-get install -y \
    libgl1 \
    libglib2.0-0 \
    libsm6 \
    libxext6 \
    libxrender-dev \
    libgomp1 \
    ffmpeg \
    curl \
    && rm -rf /var/lib/apt/lists/* \
    && apt-get clean

# Copy requirements first for better caching
COPY requirements.txt .

# Install Python dependencies with pip cache cleanup
# Split installation to reduce peak disk usage
RUN pip install --no-cache-dir \
    fastapi==0.104.1 \
    uvicorn[standard]==0.24.0 \
    python-multipart==0.0.6 \
    pillow==10.1.0 \
    numpy==1.24.3 \
    && pip cache purge

RUN pip install --no-cache-dir \
    opencv-python-headless==4.8.1.78 \
    onnxruntime==1.16.3 \
    && pip cache purge

RUN pip install --no-cache-dir \
    ultralytics==8.3.0 \
    && pip cache purge

# Copy application files
COPY main.py .

# Copy your model file (choose one based on your preference)
# For .pt file:
COPY model.pt .
# OR for .onnx file:
# COPY model.onnx .

# Set environment variable for model path
ENV MODEL_PATH=model.pt
# OR for ONNX:
# ENV MODEL_PATH=model.onnx

# Expose port
EXPOSE 8000

# Health check (using curl instead of requests)
# Increased start-period to 120s to allow model loading time
HEALTHCHECK --interval=30s --timeout=10s --start-period=120s --retries=3 \
    CMD curl -f http://localhost:8000/health || exit 1

# Run the application with uvicorn directly for better production handling
# Increased timeout for slow model loading
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000", "--timeout-keep-alive", "120"]