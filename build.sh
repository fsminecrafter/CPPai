#!/usr/bin/env bash
set -euo pipefail

with_cuda="${WITH_CUDA:-OFF}"
for arg in "$@"; do
    case "$arg" in
        -DWITH_CUDA=ON|--cuda) with_cuda=ON ;;
        -DWITH_CUDA=OFF|--no-cuda) with_cuda=OFF ;;
        *)
            echo "Usage: $0 [--cuda|--no-cuda|-DWITH_CUDA=ON|-DWITH_CUDA=OFF]" >&2
            exit 2
            ;;
    esac
done

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
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release -DWITH_CUDA="$with_cuda"
cmake --build build --parallel $(nproc)
cp build/pyai "$PWD/pyai"
