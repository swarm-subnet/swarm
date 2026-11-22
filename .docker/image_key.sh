#!/usr/bin/env bash
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

# ---------------------------------------------------------------
# image_key.sh – print the name suffix of the CI images this checkout describes.
#
#   bash .docker/image_key.sh base                  -> the test image's key
#   bash .docker/image_key.sh validator <base ref>  -> the validator image's key on that base
#
# Tests and the operator publish both name images with it, so they always agree.
set -euo pipefail

cd "$(dirname "${BASH_SOURCE[0]}")/.."
case "${1:-}" in
  base)
    sha256sum .docker/Dockerfile requirements.txt | sha256sum | cut -c1-16
    ;;
  validator)
    [ -n "${2:-}" ] || { echo "usage: image_key.sh validator <base image ref>" >&2; exit 2; }
    { echo "$2"; sha256sum .docker/validator.Dockerfile .docker/validator.Dockerfile.dockerignore \
        .docker/validator-entrypoint.sh validator/requirements.txt; } | sha256sum | cut -c1-16
    ;;
  *)
    echo "usage: image_key.sh base | validator <base image ref>" >&2
    exit 2
    ;;
esac
