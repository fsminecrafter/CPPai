#!/usr/bin/env bash
set -euo pipefail

with_cuda="${WITH_CUDA:-OFF}"
with_vulkan="${WITH_VULKAN:-ON}"
for arg in "$@"; do
    case "$arg" in
        -DWITH_CUDA=ON|--cuda) with_cuda=ON ;;
        -DWITH_CUDA=OFF|--no-cuda) with_cuda=OFF ;;
        -DWITH_VULKAN=ON|--vulkan) with_vulkan=ON ;;
        -DWITH_VULKAN=OFF|--no-vulkan) with_vulkan=OFF ;;
        *)
            echo "Usage: $0 [--cuda|--no-cuda] [--vulkan|--no-vulkan]" >&2
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
    libvulkan-dev \
    glslc \
    curl

echo "Compilation..."
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release -DWITH_CUDA="$with_cuda" -DWITH_VULKAN="$with_vulkan"
cmake --build build --parallel $(nproc)
cp build/pyai "$PWD/pyai"
