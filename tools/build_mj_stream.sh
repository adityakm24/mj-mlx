#!/bin/zsh
# Build the GPU-driven expert streaming extension (src/mj_mlx/ext/_mj_stream).
# Needs cmake and nanobind in the venv (uv pip install cmake nanobind). nanobind domain "mlx" and the venv's libmlx,
# so mx.array crosses the boundary and there is one MLX runtime in the process.
set -e
cd "$(dirname "$0")/../src/mj_mlx/ext"
ROOT=../../..
$ROOT/.venv/bin/cmake -S . -B build -DMJ_NB_DOMAIN=mlx -DPython_EXECUTABLE=$PWD/$ROOT/.venv/bin/python
$ROOT/.venv/bin/cmake --build build -j 8
