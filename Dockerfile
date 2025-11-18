FROM python:3.11-slim

# Set working directory
WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    libgl1 \
    libglib2.0-0 \
    libsm6 \
    libxext6 \
    libxrender-dev \
    libgomp1 \
    ffmpeg \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements first for better caching
COPY requirements.txt .

# Install Python dependencies
RUN pip install --no-cache-dir -r requirements.txt

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