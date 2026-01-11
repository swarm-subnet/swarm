# syntax=docker/dockerfile:1
FROM python:3.10-slim

ENV PYTHONUNBUFFERED=1 \
    OPENBLAS_NUM_THREADS=1 \
    MKL_NUM_THREADS=1 \
    NUMEXPR_NUM_THREADS=1 \
    OMP_NUM_THREADS=1

RUN useradd -m -u 1000 evaluator && \
    mkdir -p /workspace /app && \
    chown -R evaluator:evaluator /workspace /app

RUN --mount=type=cache,target=/var/cache/apt,sharing=locked \
    --mount=type=cache,target=/var/lib/apt,sharing=locked \
    apt-get update && apt-get install -y --no-install-recommends \
    git gcc g++ iptables libgl1-mesa-dri libglib2.0-0 libsm6 \
    libxext6 libxrender-dev libgomp1

RUN pip install --upgrade pip setuptools wheel

COPY swarm/validator/docker/docker-requirements.txt /tmp/docker-requirements.txt

RUN --mount=type=cache,target=/root/.cache/pip \
    pip install -r /tmp/docker-requirements.txt

RUN --mount=type=cache,target=/root/.cache/pip \
    pip install git+https://github.com/swarm-subnet/swarm-gym-pybullet-drones.git && \
    pip install --upgrade "numpy>=2.0.0"

WORKDIR /app

COPY --chown=evaluator:evaluator swarm/ /app/swarm/

USER evaluator

WORKDIR /workspace

CMD ["python", "/workspace/submission/main.py"]
