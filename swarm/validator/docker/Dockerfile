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

FROM python:3.11-slim

COPY --from=ghcr.io/astral-sh/uv:0.12.8 /uv /uvx /bin/

ENV PYTHONUNBUFFERED=1 \
    PYTHONPATH=/app \
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

# Runtime deps live in their own environment so the per-submission install can
# resolve against them instead of rebuilding a separate closure, and so it never
# needs write access to /usr/local.
RUN uv venv /opt/env
ENV PATH=/opt/env/bin:$PATH

RUN uv pip install --python /opt/env/bin/python --upgrade setuptools wheel

COPY swarm/validator/docker/docker-requirements.txt /tmp/docker-requirements.txt

RUN --mount=type=cache,target=/root/.cache/uv \
    uv pip install --python /opt/env/bin/python \
    --index-url https://download.pytorch.org/whl/cpu torch

RUN --mount=type=cache,target=/root/.cache/uv \
    uv pip install --python /opt/env/bin/python -r /tmp/docker-requirements.txt

RUN --mount=type=cache,target=/root/.cache/uv \
    uv pip install --python /opt/env/bin/python --upgrade "numpy>=2.0.1,<3.0.0"

# The dependency install runs as the validator host's uid, unknown at build time.
# Only the two directories a package lands in are opened up; the interpreter and
# the rest of the environment stay 755, and evaluation runs read-only.
RUN chmod -R a+w /opt/env/lib/python3.11/site-packages /opt/env/bin

COPY swarm /app/swarm

USER evaluator

WORKDIR /workspace

CMD ["python", "/workspace/submission/main.py"]
