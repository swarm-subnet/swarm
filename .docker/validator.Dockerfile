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

# The validator, shipped as an image instead of an install.
#
# Built on the same base the runner image uses, so the validator and the flights it
# scores carry one package set. An operator pulls this and runs it; nothing is compiled
# on their host and nothing depends on what their machine already had installed.
#
# Build context is the repository root:
#   docker build -f .docker/validator.Dockerfile -t swarm-validator:dev .

ARG BASE_IMAGE=ghcr.io/swarm-subnet/swarm:base
FROM ${BASE_IMAGE}

# Brought in here as well as in the base: a host holding a base image from before
# uv arrived would otherwise fail this build with "uv: not found".
COPY --from=ghcr.io/astral-sh/uv:0.12.8 /uv /uvx /bin/

# The validator drives the host's Docker daemon through the mounted socket, so it needs
# the client and buildx to build and run the evaluation containers. iptables and nsenter
# are what the runner uses to cut a container's network off mid-flight.
RUN apt-get update && apt-get install -y --no-install-recommends \
        ca-certificates curl gnupg iptables util-linux libcap2-bin \
    && install -m 0755 -d /etc/apt/keyrings \
    && curl -fsSL https://download.docker.com/linux/debian/gpg \
        | gpg --dearmor -o /etc/apt/keyrings/docker.gpg \
    && chmod a+r /etc/apt/keyrings/docker.gpg \
    && echo "deb [arch=$(dpkg --print-architecture) signed-by=/etc/apt/keyrings/docker.gpg] \
https://download.docker.com/linux/debian $(. /etc/os-release && echo "$VERSION_CODENAME") stable" \
        > /etc/apt/sources.list.d/docker.list \
    && apt-get update && apt-get install -y --no-install-recommends \
        docker-ce-cli docker-buildx-plugin docker-compose-plugin \
    && rm -rf /var/lib/apt/lists/*

# File capabilities rather than running the whole validator as root: it drops to an
# unprivileged uid and only these two binaries keep the privilege they actually need.
RUN setcap cap_sys_admin+ep /usr/bin/nsenter \
    && setcap cap_net_admin,cap_net_raw+ep /usr/sbin/xtables-legacy-multi || true

WORKDIR /opt/swarm-validator
COPY . /opt/swarm-validator

# The validator's own list decides what it runs on; the package itself is then
# installed without resolving the root requirements a second time.
RUN uv pip install --system --no-cache -r /opt/swarm-validator/validator/requirements.txt \
    && uv pip install --system --no-cache --no-deps -e /opt/swarm-validator

COPY .docker/validator-entrypoint.sh /usr/local/bin/validator-entrypoint
RUN chmod +x /usr/local/bin/validator-entrypoint

# The tag this image is published under, and what the auto-updater compares.
ARG SWARM_VERSION=dev
LABEL swarm.__version__="${SWARM_VERSION}"

ENV PYTHONPATH=/opt/swarm-validator \
    SWARM_VALIDATOR_HOME=/opt/swarm-validator

ENTRYPOINT ["/usr/local/bin/validator-entrypoint"]
