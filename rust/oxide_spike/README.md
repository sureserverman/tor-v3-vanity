# cuda-oxide spike (`oxide_spike`)

Isolated evaluation of [NVIDIA cuda-oxide v0.3.0](https://github.com/NVIDIA/cuda-rust/releases/tag/cuda-oxide%2Fv0.3.0)
as an alternate compiler for tor-v3-vanity's **incremental** Ed25519 search kernel.

The default NVPTX / `rustacuda` build of `tor-v3-vanity` is unchanged. This crate
is a sibling package (its own `[workspace]`) and is only built via `cargo oxide`.

## Requirements

| Piece | Version |
|-------|---------|
| Rust | **1.99.1** (pinned in `rust-toolchain.toml`) with `rust-src`, `rustc-dev`, `llvm-tools` |
| cargo-oxide | `cargo +1.99.1 install --git https://github.com/NVIDIA/cuda-rust.git --tag cuda-oxide/v0.3.0 cargo-oxide` |
| CUDA Toolkit | **13.0+** (`cuda.h`, `curand.h`, `nvcc`) |
| Driver | **R580+** (needed to *run*; `cargo oxide build` works without a GPU) |
| Clang | **21+** (`clang-21`, `libclang-*-21-dev`) for host `cuda-bindings` bindgen |
| GPU | Ampere+ recommended by cuda-oxide docs (`sm_80+`) |

## Build / run

```bash
cd rust/oxide_spike
export PATH="/usr/local/cuda/bin:$PATH"
cargo oxide doctor
cargo oxide build --arch sm_75   # or sm_80 / your device
cargo oxide run -- check
cargo oxide run -- dump
cargo oxide run -- bench
```

Modes:

- **`check`** — 1 block × 1 thread, accept-all prefix, fixed seed; compares recorded
  scalars to a CPU `add_u256` oracle.
- **`dump`** — prints sorted hex scalars for the same fixed seed / geometry as the
  parent crate's `ab_nvptx_baseline` example (for cross-backend diff).
- **`bench`** — multi-block launch loop; prints `keys_per_s`.

## A/B against the NVPTX backend

On a machine with **both** toolchains (nightly + NVPTX components, and 1.99.1 +
cuda-oxide) and a GPU:

```bash
# From rust/
./scripts/ab-cuda-oxide.sh
```

Or manually:

```bash
# NVPTX baseline (parent crate, feature-gated example)
CUDA_LIB_DIR=/usr/local/cuda/lib64/stubs \
  cargo +nightly run --release --example ab_nvptx_baseline --features cuda-oxide \
  > /tmp/nvptx.dump

# cuda-oxide dump
(cd oxide_spike && cargo oxide run -- dump > /tmp/oxide.dump)

# Compare the hex scalar lines (ignore the backend= header)
diff <(grep -v '^backend=' /tmp/nvptx.dump | sort) \
     <(grep -v '^backend=' /tmp/oxide.dump | sort)
```

Throughput A/B: run `t3v` (default backend) with `T3V_ITERS=...` on a throwaway
prefix, and `cargo oxide run -- bench` here; compare `keys_per_s`.

## What was ported

Only `render_incremental` (field/point walk, windowed batch invert, prefix match,
atomic claim into a 64-slot ring). Host glue is a minimal cuda-core launch, not
the full CLI.
