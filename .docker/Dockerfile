# syntax=docker/dockerfile:1
FROM python:3.11-slim-bookworm

COPY --from=ghcr.io/astral-sh/uv:0.12.8 /uv /uvx /bin/

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PYTHONPATH=/app \
    OMP_NUM_THREADS=1 \
    OPENBLAS_NUM_THREADS=1 \
    MKL_NUM_THREADS=1

# gcc/g++/git -> source builds for pycapnp / the pybullet forks
# libgl1 … libgomp1 -> pybullet + opencv need these at import, even headless
RUN apt-get update && apt-get install -y --no-install-recommends \
        git gcc g++ \
        libgl1 libglib2.0-0 libsm6 libxext6 libxrender1 libgomp1 \
    && rm -rf /var/lib/apt/lists/*

RUN uv pip install --system --no-cache --upgrade wheel

# CPU-only torch at the exact pin from requirements.txt, so the next
# install treats it as already satisfied instead of pulling the CUDA wheel.
RUN uv pip install --system --no-cache torch==2.10.0 \
        --index-url https://download.pytorch.org/whl/cpu

COPY requirements.txt /tmp/requirements.txt

# setuptools 82 drops pkg_resources.fixup_namespace_packages, which pytest's monkeypatch imports.
RUN uv pip install --system --no-cache -r /tmp/requirements.txt "setuptools<82"

WORKDIR /app
CMD ["pytest"]
