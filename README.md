# tor-v3-vanity
A TOR v3 vanity url generator designed to run on an NVIDIA GPU.

Disclaimer: This project is brand new and hasn't been thoroughly vetted.
Please report any bugs you find [here](https://github.com/dr-bonez/tor-v3-vanity/issues).

The program is designed to use all available CUDA devices, and will automatically decide the number of threads and blocks to use.

Now supports multiple prefixes!

## Compatibility

- **Rust:** 2021 edition; stable or nightly (nightly required for CUDA build).
- **CLI:** [clap](https://clap.rs) 4 with derive API.
- **Errors:** [anyhow](https://github.com/dtolnay/anyhow) for application error handling.
- **GPU target:** `nvptx64-nvidia-cuda`; kernel built with Rust’s **llvm-bitcode-linker** (no ptx-linker).

## Installation

- [Install Rust](https://rustup.rs) (rustup)
- **CUDA** — required to build and run. See [CUDA install](https://developer.nvidia.com/cuda-downloads).

### Why CUDA might be “missing” on the build machine

- **Only the NVIDIA driver is installed** — the driver gives you `libcuda.so` at runtime, but on some distros the **CUDA toolkit** (libraries and headers for building) is a separate install. You need the toolkit (or at least the libs) to *build*.
- **CUDA is in a non-default path** — e.g. installed under `/opt`, or from a package that puts libs in `/usr/lib/x86_64-linux-gnu`. Use `CUDA_LIB_DIR` or `CUDA_HOME` (see Troubleshooting).
- **Different machine** — building on a machine without CUDA (e.g. CI or a laptop) will fail at link; build on a machine with CUDA, or install the CUDA toolkit there.

### How to check CUDA is present

- **Libraries (needed to build):**  
  `ls /usr/local/cuda/lib64/libcuda.so` or `ls /usr/lib/x86_64-linux-gnu/libcuda.so`  
  At least one should exist (or set `CUDA_LIB_DIR` to the directory that contains `libcuda.so`).
- **Optional (for sanity):**  
  `nvidia-smi` (driver/runtime) and/or `nvcc --version` (toolkit compiler).

### How to install CUDA (if missing)

- **Linux:** [NVIDIA CUDA download](https://developer.nvidia.com/cuda-downloads) — choose your distro and install the toolkit (or the “runfile” and select the libraries/development components).
- **Ubuntu/Debian:** You can install the meta-package and libs, e.g.  
  `sudo apt install nvidia-cuda-toolkit`  
  (libs often end up in `/usr/lib/x86_64-linux-gnu`; the build will look there as a fallback.)

**One-time setup (reproducible on any machine).** Either:

- **Makefile (recommended):** from `rust/`, run `make` — it runs setup then build. Or run `make setup` once, then `cargo +nightly build --release`.
- **Manual:** run once:
  ```bash
  rustup install nightly && rustup target add nvptx64-nvidia-cuda --toolchain nightly && rustup component add llvm-bitcode-linker llvm-tools rust-src --toolchain nightly
  ```
  Then build: `cargo +nightly build --release` (from `rust/`).

Or install the binary: `cargo +nightly install --path .` (from `rust/`).

**ARM64 (aarch64):** `make arm64`. On an ARM64 host this builds natively; on an amd64 host it cross-compiles (run `make fetch-aarch64-cuda-libs` first, optionally `FETCH_DRIVER=1` for libcuda). See [docs/arm64-build.md](docs/arm64-build.md).

**AMD64 (x86_64):** `make amd64`. On an x86_64 host this builds natively; on an arm64 host it cross-compiles (run `make fetch-x86-64-cuda-libs` first, optionally `FETCH_DRIVER=1` for libcuda). See [docs/amd64-build.md](docs/amd64-build.md).

## Troubleshooting

- **`Cargo.lock does not exist` / `unable to build with the standard library` / `rust-src`**
  - The build needs the nightly standard library source. Run the one-time setup from the Installation section (single line with `rustup install nightly && ... rust-src ...`), then `cargo +nightly build --release`.

- **Build appears stuck at `tor-v3-vanity(build)`**
  - The build script compiles the NVPTX kernel as a separate step.
  - First build can take several minutes while toolchain artifacts are compiled.
  - Run with `cargo +nightly build --release -vv` to see detailed progress.

- **`linker rust-ptx-linker not found`**
  - Do **not** install legacy `ptx-linker`.
  - Run the one-time setup (Installation section), then `cargo +nightly build --release`.

- **Nightly-only NVPTX errors**
  - Run the one-time setup. Check active compiler: `rustc +nightly -V`.

- **`unable to find library -lcuda` / `-lcudart` / `-lcublas`**
  - CUDA libraries are missing or not on the default path. See **Why CUDA might be “missing”** and **How to check / install CUDA** above.
  - If CUDA is installed but in a different directory, set `CUDA_LIB_DIR` (path to the dir containing `libcuda.so`) or `CUDA_HOME` (CUDA root; build uses `$CUDA_HOME/lib64`). Example: `make CUDA_LIB_DIR=/usr/lib/x86_64-linux-gnu`

## Usage

- Create output dir
  - `mkdir mykeys`
- Run `t3v`
  - `t3v --dst mykeys/ myprefix1,myprefix2`
- Use the resulting file as your `hs_ed25519_secret_key`
  - `cat mykeys/myprefixwhatever.onion > /var/lib/tor/hidden_service/hs_ed25519_secret_key`

## Bench
On my 1070ti, I get the following time estimates:

| Prefix Length | Time       |
| ------------- | ---------- |
|             5 | 7 minutes  |
|             6 | 3.5 hours  |
|             7 | 5 days     |
|             8 | 22.5 weeks |
|             9 | 14 years   |

### 8× H100, incremental algorithm

Measured on an 8×H100 node with `--algo incremental` (the default): roughly
**19 G keys/s** (≈2.4 G/s per H100) — about **200×** the original seed kernel, which
does ~95 M keys/s on the same box. Expected (mean) time to land a prefix:

| Prefix Length | Expected time |
| ------------- | ------------- |
|             6 | < 1 second    |
|             7 | ~2 seconds    |
|             8 | ~1 minute     |
|             9 | ~30 minutes   |
|            10 | ~16 hours     |
|            11 | ~22 days      |
|            12 | ~2 years      |

Throughput scales with GPU count and varies with contention; each extra character is
32× the work.

## Performance & live dashboard

The program automatically spawns one worker thread per CUDA device and uses every
GPU it finds, so a multi-GPU box (e.g. an 8×H100 node) is saturated out of the box.

On startup each GPU **autotunes itself**: it measures throughput while doubling the
number of keygens per launch (a grid-stride loop) and settles on the value that gets
~99% of peak throughput, capped so a single launch stays short (bounded hit-detection
latency and responsive Ctrl-C). No flags required.

While running, an interactive terminal shows a **live dashboard** that updates in
place:

```
tor-v3-vanity · 8×GPU · 00:01:11
1.35 T keys · 19.21 G/s (avg 19.04 G/s)

PREFIX                  FOUND       CHANCE   E[NEXT]
shaonsenllc               0/1       0.004%     21.9d
paarthshah                0/1       0.120%     16.4h

GPU     ITERS         RATE
  0      8192     2.36 G/s
  1      8192     2.37 G/s
  2      8192     2.50 G/s
  ...
```

Each found key is printed as a permanent line above the dashboard and written to the
destination folder. Per-prefix ETAs are shown individually, so shorter prefixes that
will land soon are no longer hidden behind the hardest one. When stdout is **not** a
TTY (piped/redirected), it falls back to plain status lines every 30s.

## Required vs bonus prefixes (when does it stop?)

Positional prefixes are **required**: the run exits as soon as every one of them has
been found. Prefixes passed with **`--bonus`** are searched the whole time and saved
if they turn up, but they never keep the run alive on their own.

This lets you say "I mainly want `shaonsen`, but grab the longer `shaonsenllc` too if
it happens to appear while we're still searching":

```bash
t3v --dst mykeys/ shaonsen --bonus shaonsenllc
```

The moment `shaonsen` is found, t3v writes both keys it managed to collect, prints a
summary, and exits — it won't keep grinding for years just for the bonus. List
multiple of either kind comma-separated (`a,b,c`). Use **`--count N`** to collect N
matches of each prefix before considering it satisfied (default 1); `--count`
applies to bonus prefixes too, though they never gate the exit.

Prefixes are case-insensitive (onion addresses are lowercase) and may only use the
base32 alphabet `a–z` and `2–7` — `0`, `1`, `8` and `9` are never valid. Anything
else is rejected before the search starts.

Progress is reported as **CHANCE** — the probability a match has turned up by now —
and **E\[NEXT\]**, the expected wait for the next one. The search draws a fresh
random base every launch, so it is memoryless: `E[NEXT]` stays constant rather than
counting down, because running longer genuinely does not bring the next hit closer.

## Search algorithm (`--algo`)

There are two GPU kernels:

- **`--algo incremental`** (default) — each thread computes one `A = a·B`, then
  enumerates `A, A+B, A+2B, …` by point addition (the secret scalar for step `k` is
  just `a+k`), with batched Montgomery inversion over a window.
  **~200× faster** (≈19 G keys/s vs ≈95 M keys/s across 8×H100). Keys are stored as
  the raw scalar; Tor uses it un-clamped.
- **`--algo seed`** — the original reference path: hash a fresh random seed per
  candidate, which costs a full scalar multiplication every time.

The incremental algorithm is taken from [**mkp224o**](https://github.com/cathugger/mkp224o):
enumerate keys by repeated basepoint addition with batched modular inversion, rather than a
fresh scalar multiplication per candidate. mkp224o is the mature CPU implementation of this
idea; the `incremental` kernel is a GPU port of it.

The incremental path's field/point arithmetic is validated bit-for-bit against
curve25519-dalek (`cargo run --example field_oracle`, `--example gpu_selftest`),
and every produced key is independently re-derived and signed/verified
(`--example validate_key`). Each found key is also re-derived on the host before
writing, so a key whose address doesn't actually carry the prefix is discarded
rather than saved.

```bash
t3v --algo incremental --dst mykeys/ myprefix
```

### Un-clamped scalars: read this before using another tool on these keys

`--algo incremental` stores the secret **scalar** directly, un-clamped — which is
what Tor does with the on-disk key, and what makes the algorithm possible at all.
`--algo seed` stores a clamped, SHA-512-expanded scalar instead, so the two
algorithms produce structurally different (both valid) key files.

The consequence: **any tool that recovers the public key by clamping will derive
the wrong address from an incremental-generated key file, silently.** That includes
`ed25519-dalek`'s `ExpandedSecretKey → PublicKey`, which this repo's own `seed`
path uses. Tor itself is unaffected. To check a key independently:

```bash
cargo run --release --example validate_key -- mykeys/<address>.onion
```

### Why only one key per launch is kept

Every candidate within a single GPU launch is `base + k` for one random `base`, so
two keys harvested from the same launch have secret scalars that differ by a small,
searchable amount — whoever holds one could recover the other in seconds. t3v
therefore keeps at most one match per launch and re-seeds before the next, so every
key it writes is independent of every other. Discarded matches are counted and
reported in the run summary rather than dropped silently.

This costs nothing at realistic prefix lengths (a launch almost never contains two
matches), but it does mean `--count N` on a very short prefix collects roughly one
key per launch.

Within a launch, matches are written to a per-prefix ring of slots claimed with an
atomic increment, so concurrent matches never overwrite one another. Every key is
still re-derived on the host before being written.

Tuning knobs:

- **`T3V_ITERS`** (runtime env var) — pins keygens-per-launch and **skips autotune**.
  Useful for benchmarking a specific value, e.g. `T3V_ITERS=1024 t3v myprefix`.

- **`KERNEL_TARGET_CPU`** (build-time env var, default `sm_75`) — the *minimum*
  compute capability the PTX kernel targets. PTX is JIT-compiled to the installed GPU
  at load and the kernel uses no arch-specific instructions, so `sm_75` runs
  everywhere from Turing up (including H100) at full speed. It's a minimum, so don't
  raise it above your oldest GPU or the module won't load there.
