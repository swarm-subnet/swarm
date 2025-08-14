FROM python:3.10-slim

# Security: Run as non-root user
RUN useradd -m -u 1000 evaluator && \
    mkdir -p /workspace /app && \
    chown -R evaluator:evaluator /workspace /app

# Install system dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    git \
    gcc \
    g++ \
    libgl1-mesa-dri \
    libglib2.0-0 \
    libsm6 \
    libxext6 \
    libxrender-dev \
    libgomp1 \
    && rm -rf /var/lib/apt/lists/*

# Install build tools for git packages
RUN pip install --upgrade pip setuptools wheel

# Install Docker-specific Python dependencies (without gym-pybullet-drones)
COPY swarm/validator/docker-requirements.txt /tmp/docker-requirements.txt
RUN pip install --no-cache-dir -r /tmp/docker-requirements.txt

# Install gym-pybullet-drones (with numpy 1.x) then upgrade to numpy 2.x
RUN pip install --no-cache-dir git+https://github.com/utiasDSL/gym-pybullet-drones.git
RUN pip install --no-cache-dir --upgrade "numpy>=2.0.0"

WORKDIR /app

# Copy entire swarm package
COPY --chown=evaluator:evaluator swarm/ /app/swarm/

# Switch to non-root user
USER evaluator

# Set working directory for evaluation
WORKDIR /workspace

# Environment variables for stable execution
ENV PYTHONUNBUFFERED=1
ENV OPENBLAS_NUM_THREADS=1
ENV MKL_NUM_THREADS=1
ENV NUMEXPR_NUM_THREADS=1
ENV OMP_NUM_THREADS=1

# Entry point
ENTRYPOINT ["python", "/app/swarm/core/evaluator.py"]