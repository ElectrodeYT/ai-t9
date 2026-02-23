FROM python:3.11-slim

# Install system dependencies
RUN apt-get update && apt-get install -y \
    git \
    && rm -rf /var/lib/apt/lists/*

# Set working directory
WORKDIR /app

# Copy project files
COPY . /app

# Install Python dependencies
RUN pip install --no-cache-dir -e .[train,data,vast]

# Create data directory
RUN mkdir -p /data

# Set environment variables for R2
# These should be set when running the container
# ENV R2_ACCESS_KEY_ID=...
# ENV R2_SECRET_ACCESS_KEY=...
# ENV R2_ENDPOINT=https://<account>.r2.cloudflarestorage.com
# ENV R2_BUCKET=ai-t9-data

# Copy the training script
COPY scripts/vast_train.py /app/vast_train.py

# Default command
CMD ["python", "/app/vast_train.py"]