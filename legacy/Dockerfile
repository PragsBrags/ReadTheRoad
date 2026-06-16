FROM python:3.11-slim

#set working dir inside the container
WORKDIR /app 

# Install system dependencies in one layer to reduce image size
RUN apt-get update && apt-get install -y \
    git \
    curl \
    wget \
    gcc \
    g++ \
    libgl1 \
    libglib2.0-0 \
    libgthread-2.0-0 \
    libgomp1 \
    libsm6 \
    libxext6 \
    libxrender-dev \
    libusb-1.0-0 \
    v4l-utils \
    && rm -rf /var/lib/apt/lists/*

# Install Python dependencies (copy deps first from host to container)
COPY requirements.txt .

RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt

# Copy project source
COPY . .

ENV PADDLE_PDX_DISABLE_MODEL_SOURCE_CHECK=True
ENV PYTHONPATH=/app

CMD ["python","main.py"]