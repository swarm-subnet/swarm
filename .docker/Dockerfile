# syntax=docker/dockerfile:1
FROM python:3.11-slim

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    OMP_NUM_THREADS=1 \
    OPENBLAS_NUM_THREADS=1 \
    MKL_NUM_THREADS=1

# bash        -> tests/test_scripts_shell.py runs `bash -n` on scripts/**/*.sh
# gcc/g++/git -> source builds for pycapnp / the pybullet forks
# libgl1 … libgomp1 -> pybullet + opencv need these at import, even headless
RUN apt-get update && apt-get install -y --no-install-recommends \
        bash git gcc g++ \
        libgl1 libglib2.0-0 libsm6 libxext6 libxrender1 libgomp1 \
    && rm -rf /var/lib/apt/lists/*

RUN pip install --no-cache-dir --upgrade pip wheel

COPY requirements.txt /tmp/requirements.txt

# CPU-only torch at the exact pin from requirements.txt, so the next
# install treats it as already satisfied instead of pulling the CUDA wheel.
RUN pip install --no-cache-dir torch==2.10.0 \
        --index-url https://download.pytorch.org/whl/cpu

# setuptools 82 drops pkg_resources.fixup_namespace_packages, which pytest's monkeypatch imports.
RUN pip install --no-cache-dir -r /tmp/requirements.txt "setuptools<82"

WORKDIR /app
CMD ["pytest"]
