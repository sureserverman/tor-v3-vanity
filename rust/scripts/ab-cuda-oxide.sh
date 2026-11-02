#!/usr/bin/env bash
# A/B: NVPTX baseline dump vs cuda-oxide spike dump for the fixed seed.
# Requires a GPU, nightly+NVPTX toolchain, and cargo-oxide + Rust 1.99.1.
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"

NVPTX_OUT="$(mktemp)"
OXIDE_OUT="$(mktemp)"
trap 'rm -f "$NVPTX_OUT" "$OXIDE_OUT"' EXIT

export PATH="${PATH}:/usr/local/cuda/bin"
# Link against the driver stub when no real libcuda is installed (compile-only
# hosts still need this for the NVPTX host binary; a GPU box with the driver
# can leave CUDA_LIB_DIR unset).
if [[ -z "${CUDA_LIB_DIR:-}" && -e /usr/local/cuda/lib64/stubs/libcuda.so ]]; then
  export CUDA_LIB_DIR=/usr/local/cuda/lib64/stubs
fi

echo "== NVPTX baseline (features=cuda-oxide example) =="
cargo +nightly run --release --example ab_nvptx_baseline --features cuda-oxide \
  | tee "$NVPTX_OUT"

echo "== cuda-oxide spike dump =="
(cd oxide_spike && cargo oxide run -- dump) | tee "$OXIDE_OUT"

echo "== diff (sorted hex scalars; headers ignored) =="
if diff -u <(grep -v '^backend=' "$NVPTX_OUT" | sort) \
          <(grep -v '^backend=' "$OXIDE_OUT" | sort); then
  echo "A/B PASS: identical scalars"
else
  echo "A/B FAIL: scalar dumps differ" >&2
  exit 1
fi
