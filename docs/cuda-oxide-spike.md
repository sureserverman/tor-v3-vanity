# cuda-oxide backend spike

Isolated, feature-flagged evaluation of NVIDIA **cuda-oxide v0.4.0**
([release](https://github.com/NVIDIA/cuda-rust/releases/tag/cuda-oxide%2Fv0.4.0))
as an alternate compiler for this repo's incremental GPU search kernel.

## Layout

| Path | Role |
|------|------|
| `rust/oxide_spike/` | Standalone cuda-oxide crate (`#[kernel] render_incremental` + host launch) |
| `rust` feature `cuda-oxide` | Gates the NVPTX A/B dump example only; **default build unchanged** |
| `rust/examples/ab_nvptx_baseline.rs` | NVPTX dump with the same fixed seed / geometry |
| `rust/scripts/ab-cuda-oxide.sh` | Convenience A/B + PTX `.local` check |
| `rust/oxide_spike/scripts/check-ptx-local.sh` | Grep emitted PTX for `.local` / `ld.local` / `st.local` |

## Default backend (unchanged)

```bash
cd rust
# nightly + nvptx64-nvidia-cuda + llvm-bitcode-linker (see README)
cargo +nightly build --release
cargo +nightly run --release --example gpu_selftest
```

## cuda-oxide backend

```bash
# once
rustup toolchain install 1.99.1
rustup component add rust-src rustc-dev llvm-tools clippy rustfmt --toolchain 1.99.1
cargo +1.99.1 install --git https://github.com/NVIDIA/cuda-rust.git \
  --tag cuda-oxide/v0.4.0 cargo-oxide
# CUDA Toolkit 13+, Clang 21+, driver R580+ for runs

cd rust/oxide_spike
cargo oxide doctor
cargo oxide build --arch sm_75
./scripts/check-ptx-local.sh
cargo oxide run -- check
cargo oxide run -- dump
cargo oxide run -- bench
```

Pinned at tag `cuda-oxide/v0.4.0`: `cuda-device` / `cuda-host` 0.4.0, `cuda-core` 0.5.0.
See `rust/oxide_spike/README.md` for full requirements and v0.4.0 caveat notes
(#1815, #1752, #1704, #1822).

### PTX `.local` result (`cargo oxide build --arch sm_75`)

Measured on the spike kernel after the v0.4.0 bump:

| Pattern | Lines / occurrences |
|---------|---------------------|
| `.local` declarations | 1 / 1 (`__local_depot0[4096]`) |
| `ld.local` | 49 / 49 |
| `st.local` | 628 / 628 |

Total traffic hits: **678**. The 4 KiB local depot matches the windowed `ys`/`zs`/`scratch`
Fe arrays (`WINDOW=32`). Treat unexpected growth after kernel edits as a #1815-style
regression; `./scripts/check-ptx-local.sh` warns (or fails with `PTX_LOCAL_STRICT=1`).

## A/B correctness + throughput

```bash
cd rust
./scripts/ab-cuda-oxide.sh
```

`check` / `dump` use a fixed seed and a 1×1 launch so claim order is
deterministic. `bench` prints `keys_per_s` for comparison against `t3v`
on the same GPU (use `T3V_ITERS` to pin the NVPTX side). The A/B script also
builds the oxide crate and runs the PTX `.local` spill check.
