
FROM nvidia/cuda:12.6.3-runtime-ubuntu22.04

# Set system-level environment flags
ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    DEBIAN_FRONTEND=noninteractive

# Set working directory inside the container
WORKDIR /app

# Install system dependencies (including ffmpeg which Whisper absolutely requires for audio processing)
RUN apt-get update && apt-get install -y --no-install-recommends \
    python3-pip \
    python3-dev \
    ffmpeg \
    git \
    curl \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

# Upgrade pip to the latest production version
RUN pip3 install --no-cache-dir --upgrade pip

# Copy only requirements first to leverage Docker build cache layers
COPY requirements.txt .

RUN pip3 install --no-cache-dir -r requirements.txt --extra-index-url https://download.pytorch.org/whl/cu126

# Copy the rest of your application code into the workspace
COPY ./app .

# Expose the default FastAPI port
EXPOSE 8000

# Launch Uvicorn bound to all interfaces inside the container network mesh
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]