#!/usr/bin/env bash
set -euo pipefail

echo "Dependencies (apt)"
apt-get update
apt-get install -y --no-install-recommends \
    build-essential \
    cmake \
    python3-dev \
    python3-pip \
    zlib1g-dev \
    curl

echo "Compilation..."
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release -DWITH_CUDA="${WITH_CUDA:-OFF}"
cmake --build build --parallel $(nproc)
cp build/pyai "$PWD/pyai"
