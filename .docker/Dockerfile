# syntax=docker/dockerfile:1
# The MIT License (MIT)
# Copyright © 2026 Swarm

# Permission is hereby granted, free of charge, to any person obtaining a copy of this software and associated
# documentation files (the “Software”), to deal in the Software without restriction, including without limitation
# the rights to use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies of the Software,
# and to permit persons to whom the Software is furnished to do so, subject to the following conditions:

# The above copyright notice and this permission notice shall be included in all copies or substantial portions of
# the Software.

# THE SOFTWARE IS PROVIDED “AS IS”, WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO
# THE WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL
# THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION
# OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER
# DEALINGS IN THE SOFTWARE.

FROM python:3.11-slim-bookworm

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
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

RUN pip install --no-cache-dir --upgrade pip wheel

# CPU-only torch at the exact pin from requirements.txt, so the next
# install treats it as already satisfied instead of pulling the CUDA wheel.
RUN pip install --no-cache-dir torch==2.10.0 \
        --index-url https://download.pytorch.org/whl/cpu

COPY requirements.txt /tmp/requirements.txt

# setuptools 82 drops pkg_resources.fixup_namespace_packages, which pytest's monkeypatch imports.
RUN pip install --no-cache-dir -r /tmp/requirements.txt "setuptools<82"

WORKDIR /app
CMD ["pytest"]
