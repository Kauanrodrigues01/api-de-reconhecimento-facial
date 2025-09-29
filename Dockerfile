# Use Python 3.12 slim image as base
FROM python:3.12-slim

# Set environment variables
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    DEBIAN_FRONTEND=noninteractive \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1

# Set work directory
WORKDIR /app

# Install system dependencies in stages for better caching
RUN apt-get update && apt-get install -y --no-install-recommends \
    # Essential build tools
    build-essential \
    pkg-config \
    curl \
    wget \
    && apt-get install -y --no-install-recommends \
    # OpenCV and multimedia dependencies
    libglib2.0-0 \
    libsm6 \
    libxext6 \
    libxrender-dev \
    libgomp1 \
    libgtk-3-0 \
    libglu1-mesa-dev \
    libgl1-mesa-dri \
    && apt-get install -y --no-install-recommends \
    # Video/Audio codecs
    libavcodec-dev \
    libavformat-dev \
    libswscale-dev \
    libv4l-dev \
    && apt-get install -y --no-install-recommends \
    # Image processing libraries
    libjpeg-dev \
    libpng-dev \
    libtiff-dev \
    && apt-get install -y --no-install-recommends \
    # Math libraries
    libopenblas-dev \
    liblapack-dev \
    gfortran \
    && apt-get install -y --no-install-recommends \
    # Database client
    libpq-dev \
    && rm -rf /var/lib/apt/lists/* \
    && apt-get clean

# Copy requirements first (for better caching)
COPY requirements.txt .

# Upgrade pip and install Python dependencies
RUN pip install --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt

# Copy project
COPY . .

# Create non-root user
RUN adduser --disabled-password --gecos '' --shell /bin/bash appuser && \
    chown -R appuser:appuser /app
USER appuser

# Create directories for media files and logs
RUN mkdir -p /app/media /app/staticfiles /app/logs

# Collect static files
RUN python manage.py collectstatic --noinput || true

# Expose port
EXPOSE 8000

# Health check
HEALTHCHECK --interval=30s --timeout=30s --start-period=5s --retries=3 \
    CMD curl -f http://localhost:8000/swagger/ || exit 1

# Run the application
CMD ["python", "manage.py", "runserver", "0.0.0.0:8000"]