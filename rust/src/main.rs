use std::fmt::Write as _;
use std::io::{IsTerminal, Write};
use std::path::PathBuf;
use std::sync::atomic::{AtomicBool, AtomicU64, Ordering};
use std::sync::Arc;
use std::time::{Duration, Instant};

use anyhow::Result;
use clap::Parser;
use sha3::{Digest, Sha3_256};
use tor_v3_vanity_core as core;

pub fn pubkey_to_onion(pubkey: &[u8; 32]) -> String {
    let mut hasher = Sha3_256::new();
    hasher.update(b".onion checksum");
    hasher.update(pubkey);
    hasher.update(&[3]);
    let mut onion = [0; 35];
    onion[..32].clone_from_slice(pubkey);
    onion[32..34].clone_from_slice(&hasher.finalize()[..2]);
    onion[34] = 3;
    format!(
        "{}.onion",
        base32::encode(base32::Alphabet::RFC4648 { padding: false }, &onion).to_lowercase()
    )
}

/// Longest prefix the GPU can constrain: the public key is 32 bytes = 256 bits,
/// and 51 base32 characters pin down 255 of them. Character 52 would straddle the
/// key and the checksum, which the kernel never sees.
pub const MAX_PREFIX_CHARS: usize = 51;

/// Past this length a prefix is not reachable in any practical amount of GPU time
/// (each character is 32x the work), so warn rather than let it run silently.
const IMPRACTICAL_PREFIX_CHARS: usize = 13;

/// Lowercase and validate a user-supplied prefix.
///
/// Normalizing to lowercase matters for correctness, not just tidiness: the host
/// verification gate compares against `pubkey_to_onion`, which emits lowercase, so
/// an uppercase prefix would make every genuine match fail the gate and be thrown
/// away forever.
pub fn normalize_prefix(s: &str) -> std::result::Result<String, String> {
    let p = s.trim().to_ascii_lowercase();
    if p.is_empty() {
        return Err("prefix is empty".to_string());
    }
    if let Some(c) = p.chars().find(|c| !matches!(c, 'a'..='z' | '2'..='7')) {
        return Err(format!(
            "'{}' is not a base32 character — onion addresses use a-z and 2-7 only \
             (note: 0, 1, 8 and 9 are never valid)",
            c
        ));
    }
    if p.len() > MAX_PREFIX_CHARS {
        return Err(format!(
            "{} characters is longer than the {} the public key can constrain, \
             so it could never match",
            p.len(),
            MAX_PREFIX_CHARS
        ));
    }
    Ok(p)
}

pub struct BytePrefixOwned {
    pub byte_prefix: rustacuda::memory::DeviceBuffer<u8>,
    pub last_byte_idx: usize,
    pub last_byte_mask: u8,
    /// `core::OUT_SLOTS` x 32 bytes, one slot per match claimed this launch.
    pub out: rustacuda::memory::DeviceBuffer<u8>,
    /// Atomic claim counter, reset by the host after each launch.
    pub found: rustacuda::memory::DeviceBox<u32>,
}
impl BytePrefixOwned {
    /// `s` must already have passed [`normalize_prefix`].
    pub fn new(s: &str) -> Result<Self> {
        let byte_prefix = base32::decode(
            base32::Alphabet::RFC4648 { padding: false },
            &format!("{}aa", s),
        )
        .ok_or_else(|| anyhow::anyhow!("prefix {:?} is not base32", s))?;
        // Number of fully-constrained bytes, plus a mask for the partial trailing
        // byte when the prefix doesn't end on a byte boundary (0 = byte-aligned, e.g.
        // any prefix whose length is a multiple of 8).
        let last_byte_idx = 5 * s.len() / 8;
        let n_bits = (5 * s.len()) % 8;
        let last_byte_mask = if n_bits > 0 {
            ((1u8 << n_bits) - 1) << (8 - n_bits)
        } else {
            0
        };
        Ok(BytePrefixOwned {
            byte_prefix: rustacuda::memory::DeviceBuffer::from_slice(&byte_prefix)?,
            last_byte_idx,
            last_byte_mask,
            out: rustacuda::memory::DeviceBuffer::from_slice(
                &vec![0u8; core::OUT_SLOTS as usize * 32],
            )?,
            found: rustacuda::memory::DeviceBox::new(&0u32)?,
        })
    }
    pub fn as_byte_prefix(&mut self) -> core::BytePrefix {
        core::BytePrefix {
            byte_prefix: self.byte_prefix.as_device_ptr(),
            byte_prefix_len: self.byte_prefix.len(),
            last_byte_idx: self.last_byte_idx,
            last_byte_mask: self.last_byte_mask,
            out: self.out.as_device_ptr(),
            found: self.found.as_device_ptr(),
        }
    }

    /// Read back this launch's matches and reset the counter. Returns the stored
    /// results plus the total the GPU claimed, which is larger when the ring
    /// overflowed.
    pub fn drain(&mut self) -> Result<(Vec<[u8; 32]>, u32)> {
        use rustacuda::memory::CopyDestination;
        let mut claimed = 0u32;
        self.found.copy_to(&mut claimed)?;
        if claimed == 0 {
            return Ok((Vec::new(), 0));
        }
        self.found.copy_from(&0u32)?;
        let stored = claimed.min(core::OUT_SLOTS) as usize;
        let mut raw = vec![0u8; core::OUT_SLOTS as usize * 32];
        self.out.copy_to(&mut raw)?;
        let results = raw
            .chunks_exact(32)
            .take(stored)
            .map(|c| {
                let mut a = [0u8; 32];
                a.copy_from_slice(c);
                a
            })
            .collect();
        Ok((results, claimed))
    }
}

fn assert_crypto_rng<Rng: rand::CryptoRng>(rng: Rng) -> Rng {
    rng
}

/// Overwrite secret material in place. `write_volatile` keeps the compiler from
/// optimizing the wipe away as a dead store.
fn wipe(buf: &mut [u8]) {
    for b in buf.iter_mut() {
        unsafe { std::ptr::write_volatile(b, 0) };
    }
    std::sync::atomic::compiler_fence(Ordering::SeqCst);
}

/// Live, lock-free telemetry shared between the GPU worker threads and the reporter.
struct DeviceStat {
    /// Cumulative keys this device has tried.
    keys: AtomicU64,
    /// Keygens per thread per launch this device settled on (after autotune).
    iters: AtomicU64,
    /// True while the device is still autotuning `iters`.
    calibrating: AtomicBool,
}

struct Shared {
    start: Instant,
    prefixes: Vec<String>,
    /// Per-prefix: true if required (must be found to finish), false if bonus.
    required: Vec<bool>,
    /// Matches to collect per prefix before it's considered satisfied.
    target: u64,
    /// Per-prefix count of matches found so far.
    found: Vec<AtomicU64>,
    devices: Vec<DeviceStat>,
    /// Matches the GPU found but that were discarded to keep saved keys mutually
    /// independent (see `harvest`), plus any that overflowed a prefix's ring.
    dropped_correlated: AtomicU64,
    /// Set once all required prefixes are satisfied; tells the reporter to finalize.
    done: AtomicBool,
}

impl Shared {
    fn total_keys(&self) -> u64 {
        self.devices
            .iter()
            .map(|d| d.keys.load(Ordering::Relaxed))
            .sum()
    }

    fn satisfied(&self, i: usize) -> bool {
        self.found[i].load(Ordering::Relaxed) >= self.target
    }

    /// True once every *required* prefix has reached its target count.
    fn required_satisfied(&self) -> bool {
        (0..self.prefixes.len()).all(|i| !self.required[i] || self.satisfied(i))
    }
}

/// Read back one launch's matches and forward at most one of them.
///
/// Every candidate in a launch is `base + k` for a single random `base`, so two
/// keys harvested from the same launch have secret scalars differing by a known,
/// small amount: whoever holds one can recover the other in seconds. Keeping only
/// one match per launch — and re-seeding before the next — makes every saved key
/// independent of every other. The rest are counted, not silently dropped.
fn harvest(
    prefixes: &mut [BytePrefixOwned],
    shared: &Shared,
    found_tx: &crossbeam_channel::Sender<(usize, [u8; 32])>,
    rng: &mut impl rand::Rng,
) -> Result<()> {
    let mut candidates: Vec<(usize, [u8; 32])> = Vec::new();
    let mut claimed_total: u64 = 0;
    for (idx, p) in prefixes.iter_mut().enumerate() {
        let (results, claimed) = p.drain()?;
        claimed_total += claimed as u64;
        candidates.extend(results.into_iter().map(|r| (idx, r)));
    }
    if candidates.is_empty() {
        return Ok(());
    }
    let (idx, out) = candidates[rng.gen_range(0..candidates.len())];
    shared
        .dropped_correlated
        .fetch_add(claimed_total - 1, Ordering::Relaxed);
    // A closed channel just means the run already finished; not an error.
    let _ = found_tx.send((idx, out));
    Ok(())
}

/// Spawn one worker thread per CUDA device. Each thread autotunes its per-launch
/// `iters` for throughput, then loops generating keys until the run finishes.
fn cuda_try_loop(
    shared: Arc<Shared>,
    found_tx: crossbeam_channel::Sender<(usize, [u8; 32])>,
    err_tx: crossbeam_channel::Sender<String>,
    forced_iters: Option<u64>,
    algo: Algo,
) -> Result<()> {
    // The two algorithms are separate kernel entry points in the same PTX module.
    let entry: &'static [u8] = match algo {
        Algo::Seed => b"render\0",
        Algo::Incremental => b"render_incremental\0",
    };

    for (i, device) in rustacuda::device::Device::devices()?.enumerate() {
        let device = device?;
        let found_tx = found_tx.clone();
        let err_tx = err_tx.clone();
        let shared = shared.clone();
        std::thread::spawn(move || {
            // A worker that dies has to say why: otherwise the only symptom is an
            // unrelated `RecvError` in main once the last sender drops.
            if let Err(e) = run_device(i, device, shared, found_tx, forced_iters, entry) {
                let _ = err_tx.send(format!("GPU #{} stopped: {:#}", i, e));
            }
        });
    }
    Ok(())
}

/// Drive one CUDA device: autotune the batch size, then launch and harvest until
/// the run is done.
fn run_device(
    i: usize,
    device: rustacuda::device::Device,
    shared: Arc<Shared>,
    found_tx: crossbeam_channel::Sender<(usize, [u8; 32])>,
    forced_iters: Option<u64>,
    entry: &'static [u8],
) -> Result<()> {
    use rand::RngCore;
    use rustacuda::launch;
    use rustacuda::memory::DeviceBox;
    use rustacuda::prelude::*;
    use std::ffi::CString;

    let mut csprng = assert_crypto_rng(rand::thread_rng());
    let _context =
        Context::create_and_push(ContextFlags::MAP_HOST | ContextFlags::SCHED_AUTO, device)?;

    // Load PTX module
    let module_data = CString::new(include_str!(env!("KERNEL_PTX_PATH")))?;
    let kernel = Module::load_from_string(&module_data)?;
    let function = kernel.get_function(std::ffi::CStr::from_bytes_with_nul(entry)?)?;

    // Create a stream to submit work to
    let stream = Stream::new(StreamFlags::NON_BLOCKING, None)?;

    // Move seed and prefixes to device
    let mut seed = [0; 32];
    let mut gpu_seed = DeviceBuffer::from_slice(&seed)?;

    let mut byte_prefixes_owned: Vec<BytePrefixOwned> = shared
        .prefixes
        .iter()
        .map(|a| BytePrefixOwned::new(a))
        .collect::<Result<_>>()?;
    let byte_prefixes: Vec<_> = byte_prefixes_owned
        .iter_mut()
        .map(|bp| bp.as_byte_prefix())
        .collect();
    let mut gpu_byte_prefixes = DeviceBuffer::from_slice(&byte_prefixes)?;

    let mut params = DeviceBox::new(&core::KernelParams {
        seed: gpu_seed.as_device_ptr(),
        byte_prefixes: gpu_byte_prefixes.as_device_ptr(),
        byte_prefixes_len: gpu_byte_prefixes.len(),
        iters: 1,
    })?;

    // calculate threads and blocks (occupancy heuristic)
    let fn_max_threads =
        function.get_attribute(rustacuda::function::FunctionAttribute::MaxThreadsPerBlock)? as u32;
    let fn_registers =
        function.get_attribute(rustacuda::function::FunctionAttribute::NumRegisters)? as u32;
    let gpu_max_threads =
        device.get_attribute(rustacuda::device::DeviceAttribute::MaxThreadsPerBlock)? as u32;
    let gpu_max_registers =
        device.get_attribute(rustacuda::device::DeviceAttribute::MaxRegistersPerBlock)? as u32;
    let gpu_cores =
        device.get_attribute(rustacuda::device::DeviceAttribute::MultiprocessorCount)? as u32;

    // Block size: hard-cap at 256 — the incremental kernel is register-heavy
    // (256*255 < 64K regs/block) and old rustacuda misreports NumRegisters.
    let reg_threads = (((gpu_max_registers as f64 * 0.9) as u32) / fn_registers.max(1)) / 32 * 32;
    let threads = *[fn_max_threads, gpu_max_threads, reg_threads.max(32), 256]
        .iter()
        .min()
        .unwrap();
    let blocks = gpu_cores * gpu_max_threads / threads;
    let grid = threads as u64 * blocks as u64;

    let stat = &shared.devices[i];

    // Autotune iters: throughput rises with iters then plateaus, so climb by
    // doubling until the gain is <2% or a launch exceeds the wall-time cap.
    let iters = if let Some(forced) = forced_iters {
        forced
    } else {
        const CAP_MS: f64 = 750.0;
        const CEILING: u64 = 1 << 22;

        // Measure throughput (keys/s) and mean launch time (ms) at `iters`.
        let mut measure = |iters: u64, rounds: u32| -> Result<(f64, f64)> {
            params.copy_from(&core::KernelParams {
                seed: gpu_seed.as_device_ptr(),
                byte_prefixes: gpu_byte_prefixes.as_device_ptr(),
                byte_prefixes_len: gpu_byte_prefixes.len(),
                iters,
            })?;
            let t0 = Instant::now();
            for _ in 0..rounds {
                csprng.fill_bytes(&mut seed);
                gpu_seed.copy_from(&seed)?;
                unsafe {
                    launch!(function<<<blocks, threads, 0, stream>>>(params.as_device_ptr()))?;
                }
                stream.synchronize()?;
            }
            let secs = t0.elapsed().as_secs_f64();
            let kps = (grid * iters * rounds as u64) as f64 / secs;
            // Count the warm-up work toward the live total too.
            stat.keys
                .fetch_add(grid * iters * rounds as u64, Ordering::Relaxed);
            Ok((kps, secs / rounds as f64 * 1000.0))
        };

        let mut cur = 16u64;
        let (mut cur_kps, _) = measure(cur, 3)?;
        loop {
            let cand = cur.saturating_mul(2);
            if cand >= CEILING {
                break;
            }
            let (kps, ms) = measure(cand, 2)?;
            if ms > CAP_MS {
                if kps > cur_kps {
                    cur = cand;
                }
                break;
            }
            if kps > cur_kps * 1.02 {
                cur = cand;
                cur_kps = kps;
            } else {
                if kps > cur_kps {
                    cur = cand;
                }
                break;
            }
        }
        cur
    };

    params.copy_from(&core::KernelParams {
        seed: gpu_seed.as_device_ptr(),
        byte_prefixes: gpu_byte_prefixes.as_device_ptr(),
        byte_prefixes_len: gpu_byte_prefixes.len(),
        iters,
    })?;
    stat.iters.store(iters, Ordering::Relaxed);
    stat.calibrating.store(false, Ordering::Relaxed);

    // Autotune ran real launches, so it may have banked matches. Take one before
    // starting the search proper rather than letting them pile up in the ring.
    harvest(&mut byte_prefixes_owned, &shared, &found_tx, &mut csprng)?;

    let keys_per_launch = grid * iters;
    while !shared.done.load(Ordering::Relaxed) {
        csprng.fill_bytes(&mut seed);
        gpu_seed.copy_from(&seed)?;
        unsafe {
            launch!(function<<<blocks, threads, 0, stream>>>(params.as_device_ptr()))?;
        }

        // The kernel launch is asynchronous, so we wait for it to finish.
        stream.synchronize()?;

        harvest(&mut byte_prefixes_owned, &shared, &found_tx, &mut csprng)?;

        stat.keys.fetch_add(keys_per_launch, Ordering::Relaxed);
    }
    Ok(())
}

const FILE_PREFIX: &'static [u8] = b"== ed25519v1-secret: type0 ==\0\0\0";

/// Search algorithm / GPU kernel selection.
#[derive(Clone, Copy, PartialEq, Eq, clap::ValueEnum)]
enum Algo {
    /// Original: hash a fresh random seed per candidate (full scalarmult each).
    /// Kept as the reference path.
    Seed,
    /// mkp224o-style (default): one scalarmult per thread, then enumerate by point
    /// addition with batched inversion. ~30x+ faster; validated end-to-end (CPU
    /// oracle, GPU self-test, host verify-gate, and a real Tor load).
    Incremental,
}

#[derive(clap::Parser)]
#[command(name = "t3v")]
struct Args {
    /// Required prefix(es) (comma-separated). The run exits once all of these are found.
    #[arg(required = true, value_delimiter(','))]
    prefix: Vec<String>,

    /// Destination folder
    #[arg(short, long)]
    dst: Option<PathBuf>,

    /// Bonus prefix(es) (comma-separated): searched and saved if found, but never
    /// keep the run alive. Useful for a longer "nice-to-have" prefix you'll grab
    /// only if it happens to turn up while searching for the required ones.
    #[arg(long, value_delimiter(','))]
    bonus: Vec<String>,

    /// Stop collecting a prefix after this many matches.
    #[arg(long, default_value_t = 1)]
    count: u64,

    /// Search algorithm: `incremental` (mkp224o-style, default, ~30x+ faster) or
    /// `seed` (original reference path).
    #[arg(long, value_enum, default_value_t = Algo::Incremental)]
    algo: Algo,
}

fn main() {
    let args = Args::parse();
    let dst = args.dst.unwrap_or_else(|| std::env::current_dir().unwrap());
    assert!(dst.is_dir(), "dst must be a directory");
    assert!(args.count >= 1, "--count must be at least 1");
    let algo = args.algo;

    // Required prefixes first, then bonus; `required[i]` tracks which is which.
    // Normalize and validate before touching CUDA: a bad prefix caught here is one
    // line of output, but caught inside a worker thread it is two unrelated panics.
    let n_required = args.prefix.len();
    let mut prefixes: Vec<String> = Vec::with_capacity(n_required + args.bonus.len());
    for raw in args.prefix.iter().chain(args.bonus.iter()) {
        match normalize_prefix(raw) {
            Ok(p) => {
                if p.len() >= IMPRACTICAL_PREFIX_CHARS {
                    eprintln!(
                        "warning: {} characters is {} times the work of {} — this run may never finish",
                        p.len(),
                        1u64 << (5 * (p.len() - IMPRACTICAL_PREFIX_CHARS + 1)).min(63),
                        IMPRACTICAL_PREFIX_CHARS - 1
                    );
                }
                if p != *raw {
                    eprintln!("note: searching for \"{}\" (prefixes are lowercase)", p);
                }
                prefixes.push(p);
            }
            Err(e) => {
                eprintln!("error: prefix \"{}\": {}", raw, e);
                std::process::exit(2);
            }
        }
    }
    let required: Vec<bool> = (0..prefixes.len()).map(|i| i < n_required).collect();

    rustacuda::init(rustacuda::CudaFlags::empty()).expect("failed to init CUDA");
    let n_devices = rustacuda::device::Device::num_devices().expect("failed to enumerate devices");
    if n_devices == 0 {
        eprintln!("No cuda devices available.");
        std::process::exit(2);
    }

    let shared = Arc::new(Shared {
        start: Instant::now(),
        found: prefixes.iter().map(|_| AtomicU64::new(0)).collect(),
        prefixes,
        required,
        target: args.count,
        devices: (0..n_devices)
            .map(|_| DeviceStat {
                keys: AtomicU64::new(0),
                iters: AtomicU64::new(0),
                calibrating: AtomicBool::new(true),
            })
            .collect(),
        dropped_correlated: AtomicU64::new(0),
        done: AtomicBool::new(false),
    });

    // T3V_ITERS pins iters and skips autotune (handy for benchmarking).
    let forced_iters = std::env::var("T3V_ITERS")
        .ok()
        .and_then(|v| v.parse::<u64>().ok())
        .filter(|&v| v > 0);

    let (found_tx, found_rx) = crossbeam_channel::unbounded::<(usize, [u8; 32])>();
    let (err_tx, err_rx) = crossbeam_channel::unbounded::<String>();
    if let Err(e) = cuda_try_loop(shared.clone(), found_tx, err_tx, forced_iters, algo) {
        eprintln!("error: could not start GPU workers: {:#}", e);
        std::process::exit(1);
    }

    // Reporter owns all stdout (live dashboard on a TTY, periodic lines otherwise).
    let (log_tx, log_rx) = crossbeam_channel::unbounded::<String>();
    let reporter_handle = {
        let shared = shared.clone();
        std::thread::spawn(move || reporter(shared, log_rx))
    };

    // Persist one match and notify the reporter. Drops matches for prefixes we've
    // already collected enough of (idempotent past the target count).
    let save = |idx: usize, mut out: [u8; 32]| {
        if shared.satisfied(idx) {
            wipe(&mut out);
            return;
        }
        // Derive the address + secret-key file body from the trusted CPU path
        // (dalek), without writing yet. `out` is a seed (Seed) or a raw scalar
        // (Incremental).
        let (onion, mut body): (String, Vec<u8>) = match algo {
            Algo::Seed => {
                let esk: ed25519_dalek::ExpandedSecretKey =
                    (&ed25519_dalek::SecretKey::from_bytes(&out).unwrap()).into();
                let pk: ed25519_dalek::PublicKey = (&esk).into();
                (pubkey_to_onion(pk.as_bytes()), esk.to_bytes().to_vec())
            }
            Algo::Incremental => {
                // GPU returned the raw scalar (possibly >= L); reduce mod L and derive
                // scalar·B directly (Tor uses it un-clamped). File = scalar || nonce.
                use rand::RngCore;
                let scalar = curve25519_dalek::scalar::Scalar::from_bytes_mod_order(out);
                let pubkey = (&scalar * &curve25519_dalek::constants::ED25519_BASEPOINT_TABLE)
                    .compress()
                    .to_bytes();
                let mut nonce = [0u8; 32];
                assert_crypto_rng(rand::thread_rng()).fill_bytes(&mut nonce);
                let mut body = Vec::with_capacity(64);
                body.extend_from_slice(scalar.as_bytes());
                body.extend_from_slice(&nonce);
                wipe(&mut nonce);
                (pubkey_to_onion(&pubkey), body)
            }
        };

        // Correctness gate: the independently re-derived address MUST carry the
        // prefix the GPU claimed. A miss means the kernel's field math is wrong —
        // discard it rather than writing a bogus key or falsely satisfying a goal.
        if !onion.starts_with(&shared.prefixes[idx]) {
            log_tx
                .send(format!(
                    "⚠ DISCARDED unverified match for \"{}\" (re-derived {}) — kernel bug?",
                    shared.prefixes[idx], onion
                ))
                .ok();
            wipe(&mut body);
            wipe(&mut out);
            return;
        }

        // This is an onion service identity key: create it 0600 rather than
        // inheriting whatever the umask allows, refuse to clobber an existing file,
        // and get it on disk before we claim to have found it.
        let mut opts = std::fs::OpenOptions::new();
        opts.write(true).create_new(true);
        #[cfg(unix)]
        {
            use std::os::unix::fs::OpenOptionsExt;
            opts.mode(0o600);
        }
        let write = opts.open(dst.join(&onion)).and_then(|mut f| {
            f.write_all(FILE_PREFIX)?;
            f.write_all(&body)?;
            f.sync_all()
        });
        wipe(&mut body);
        wipe(&mut out);
        if let Err(e) = write {
            log_tx
                .send(format!("⚠ could not write {}: {}", onion, e))
                .ok();
            return;
        }

        let n = shared.found[idx].fetch_add(1, Ordering::Relaxed) + 1;
        let tag = if shared.required[idx] { "" } else { " (bonus)" };
        log_tx
            .send(format!(
                "✔ {}  matched \"{}\"{}  [{}/{}]",
                onion, shared.prefixes[idx], tag, n, shared.target
            ))
            .ok();
        if n >= shared.target {
            log_tx
                .send(format!("★ \"{}\" satisfied", shared.prefixes[idx]))
                .ok();
        }
    };

    // Writer loop: persist found keys and stop once every required prefix is
    // satisfied. Bonus prefixes are saved opportunistically but don't gate exit.
    loop {
        let (idx, seed) = match found_rx.recv() {
            Ok(v) => v,
            Err(_) => {
                // Every worker exited without the run finishing, so each one hit an
                // error. Surface those instead of an opaque channel failure.
                shared.done.store(true, Ordering::Relaxed);
                drop(log_tx);
                let _ = reporter_handle.join();
                eprintln!("error: every GPU worker stopped before the search finished");
                while let Ok(e) = err_rx.try_recv() {
                    eprintln!("  {}", e);
                }
                std::process::exit(1);
            }
        };
        save(idx, seed);

        if shared.required_satisfied() {
            // A bonus match found in the same GPU batch as the final required key
            // may be queued right behind it — save those before exiting.
            while let Ok((idx, seed)) = found_rx.try_recv() {
                save(idx, seed);
            }
            break;
        }
    }

    // Finalize: tell the reporter to render its last frame and exit, then print a
    // summary below it. Dropping our log sender unblocks the reporter immediately.
    // `save` borrows log_tx only by shared ref and isn't used past here, so the
    // borrow is already released; dropping log_tx now disconnects the channel.
    shared.done.store(true, Ordering::Relaxed);
    drop(log_tx);
    let _ = reporter_handle.join();

    let elapsed = shared.start.elapsed().as_secs_f64();
    println!(
        "Done — all required prefixes found in {} ({} keys tried).",
        fmt_clock(elapsed),
        human(shared.total_keys() as f64)
    );
    for (i, p) in shared.prefixes.iter().enumerate() {
        let n = shared.found[i].load(Ordering::Relaxed);
        let kind = if shared.required[i] { "required" } else { "bonus" };
        let mark = if n > 0 { "✔" } else { "·" };
        println!("  {} {:<18} {:>8}  found {}/{}", mark, p, kind, n, shared.target);
    }
    let dropped = shared.dropped_correlated.load(Ordering::Relaxed);
    if dropped > 0 {
        println!(
            "  ({} further match(es) discarded: keys from one launch share a base scalar, \
             so only one per launch is kept)",
            dropped
        );
    }
    println!("Keys written to {}", dst.display());
    if algo == Algo::Incremental {
        println!(
            "\nNote: these keys store an un-clamped scalar, which is what Tor uses. Tools that\n\
             re-derive the address by clamping (ed25519-dalek's ExpandedSecretKey among them)\n\
             will compute a different address from the same file — that is the tool being\n\
             wrong, not the key. Check any key with:\n\
             \x20   cargo run --release --example validate_key -- <keyfile>"
        );
    }
}

// ----------------------------- reporting -----------------------------

fn reporter(shared: Arc<Shared>, log_rx: crossbeam_channel::Receiver<String>) {
    if std::io::stdout().is_terminal() {
        reporter_tty(shared, log_rx);
    } else {
        reporter_plain(shared, log_rx);
    }
}

/// Live, in-place dashboard for interactive terminals.
fn reporter_tty(shared: Arc<Shared>, log_rx: crossbeam_channel::Receiver<String>) {
    let n = shared.devices.len();
    let mut out = std::io::stdout();
    let mut prev_lines = 0usize;

    let mut last_total = 0u64;
    let mut last_dev = vec![0u64; n];
    let mut last_t = Instant::now();
    let mut inst_rate = 0.0f64;
    let mut dev_rate = vec![0.0f64; n];

    loop {
        // Collect any found-key notifications to emit as permanent scrollback.
        let mut logs: Vec<String> = Vec::new();
        let mut finished = false;
        match log_rx.recv_timeout(Duration::from_millis(1000)) {
            Ok(m) => {
                logs.push(m);
                while let Ok(m) = log_rx.try_recv() {
                    logs.push(m);
                }
            }
            Err(crossbeam_channel::RecvTimeoutError::Timeout) => {}
            Err(crossbeam_channel::RecvTimeoutError::Disconnected) => finished = true,
        }
        finished |= shared.done.load(Ordering::Relaxed);

        // Rates on a multi-second window with EMA smoothing: keys arrive in large
        // per-launch chunks, so a short window aliases badly between GPUs.
        const WINDOW: f64 = 3.0;
        const ALPHA: f64 = 0.4;
        let now = Instant::now();
        let dt = now.duration_since(last_t).as_secs_f64();
        let total = shared.total_keys();
        if dt >= WINDOW {
            let ema = |prev: f64, raw: f64| if prev <= 0.0 { raw } else { ALPHA * raw + (1.0 - ALPHA) * prev };
            inst_rate = ema(inst_rate, total.saturating_sub(last_total) as f64 / dt);
            for i in 0..n {
                let k = shared.devices[i].keys.load(Ordering::Relaxed);
                dev_rate[i] = ema(dev_rate[i], k.saturating_sub(last_dev[i]) as f64 / dt);
                last_dev[i] = k;
            }
            last_total = total;
            last_t = now;
        }

        // Erase the previous status block, emit logs above the redrawn block.
        if prev_lines > 0 {
            let _ = write!(out, "\x1b[{}A\r\x1b[J", prev_lines);
        }
        for l in &logs {
            let _ = writeln!(out, "{}", l);
        }
        let status = render_status(&shared, total, inst_rate, &dev_rate);
        let _ = write!(out, "{}", status);
        prev_lines = status.matches('\n').count();
        let _ = out.flush();

        if finished {
            // Leave the final frame on screen and move below it for the summary.
            let _ = writeln!(out);
            return;
        }
    }
}

/// Non-TTY fallback: append-only status lines (safe to pipe to a file).
fn reporter_plain(shared: Arc<Shared>, log_rx: crossbeam_channel::Receiver<String>) {
    let mut last = Instant::now();
    let mut last_total = 0u64;
    loop {
        match log_rx.recv_timeout(Duration::from_secs(1)) {
            Ok(m) => {
                println!("{}", m);
                while let Ok(m) = log_rx.try_recv() {
                    println!("{}", m);
                }
            }
            Err(crossbeam_channel::RecvTimeoutError::Timeout) => {}
            Err(crossbeam_channel::RecvTimeoutError::Disconnected) => return,
        }
        if shared.done.load(Ordering::Relaxed) {
            return;
        }
        if last.elapsed() >= Duration::from_secs(30) {
            let elapsed = shared.start.elapsed().as_secs_f64();
            let total = shared.total_keys();
            let inst = total.saturating_sub(last_total) as f64 / last.elapsed().as_secs_f64();
            let avg = if elapsed > 0.0 { total as f64 / elapsed } else { 0.0 };
            println!(
                "[{}] {} keys · {} (avg {})",
                fmt_clock(elapsed),
                human(total as f64),
                human_rate(inst),
                human_rate(avg)
            );
            for (i, p) in shared.prefixes.iter().enumerate() {
                let (chance, eta) = odds(p.len(), total, avg);
                let (chance, eta) = if shared.satisfied(i) {
                    ("✔".to_string(), "done".to_string())
                } else {
                    (fmt_pct(chance), fmt_eta(eta))
                };
                println!(
                    "    {:<18} found {:>4}  {:>10}  E[next] {}",
                    p,
                    shared.found[i].load(Ordering::Relaxed),
                    chance,
                    eta
                );
            }
            last = Instant::now();
            last_total = total;
        }
    }
}

fn render_status(shared: &Shared, total: u64, inst_rate: f64, dev_rate: &[f64]) -> String {
    let elapsed = shared.start.elapsed().as_secs_f64();
    let avg = if elapsed > 0.0 { total as f64 / elapsed } else { 0.0 };
    let n = shared.devices.len();

    let mut s = String::new();
    let _ = writeln!(
        s,
        "\x1b[1mtor-v3-vanity\x1b[0m · {}×GPU · {}",
        n,
        fmt_clock(elapsed)
    );
    let _ = writeln!(
        s,
        "{} keys · {} (avg {})",
        human(total as f64),
        human_rate(inst_rate),
        human_rate(avg)
    );
    let _ = writeln!(s);
    let _ = writeln!(
        s,
        "\x1b[1m{:<22}{:>7}{:>13}{:>10}\x1b[0m",
        "PREFIX", "FOUND", "CHANCE", "E[NEXT]"
    );
    for (i, p) in shared.prefixes.iter().enumerate() {
        let found = shared.found[i].load(Ordering::Relaxed);
        let satisfied = shared.satisfied(i);
        let name = if shared.required[i] {
            trunc(p, 22)
        } else {
            trunc(&format!("{} (bonus)", p), 22)
        };
        let found_col = format!("{}/{}", found, shared.target);
        let (prog_col, eta_col) = if satisfied {
            ("✔".to_string(), "done".to_string())
        } else {
            let (chance, eta) = odds(p.len(), total, avg);
            (fmt_pct(chance), fmt_eta(eta))
        };
        let _ = writeln!(s, "{:<22}{:>7}{:>13}{:>10}", name, found_col, prog_col, eta_col);
    }
    let _ = writeln!(s);
    let _ = writeln!(s, "\x1b[1m{:>3}{:>10}{:>13}\x1b[0m", "GPU", "ITERS", "RATE");
    for i in 0..n {
        let st = &shared.devices[i];
        if st.calibrating.load(Ordering::Relaxed) {
            let _ = writeln!(s, "{:>3}{:>10}{:>13}", i, "—", "tuning…");
        } else {
            let _ = writeln!(
                s,
                "{:>3}{:>10}{:>13}",
                i,
                st.iters.load(Ordering::Relaxed),
                human_rate(dev_rate.get(i).copied().unwrap_or(0.0))
            );
        }
    }
    // No trailing newline: the line count then equals the cursor-up distance.
    if s.ends_with('\n') {
        s.pop();
    }
    s
}

fn human(n: f64) -> String {
    const UNITS: [&str; 7] = ["", "K", "M", "G", "T", "P", "E"];
    let mut v = n;
    let mut i = 0;
    while v >= 1000.0 && i < UNITS.len() - 1 {
        v /= 1000.0;
        i += 1;
    }
    if i == 0 {
        format!("{:.0}", v)
    } else {
        format!("{:.2} {}", v, UNITS[i])
    }
}

fn human_rate(kps: f64) -> String {
    format!("{}/s", human(kps))
}

/// Chance (%) that a prefix of `len` characters has turned up by now, and the
/// expected wait for the next one.
///
/// The search is memoryless: every launch draws a fresh random base, so it never
/// gets "closer" to a hit the way an exhaustive scan does. Counting keys down
/// against the 2^(5·len) keyspace would show a shrinking ETA that reaches zero and
/// keeps going, promising something the search cannot deliver. So report the two
/// quantities that are actually true — P(found) = 1 − e^(−tried/space), which never
/// exceeds 100%, and E[time to next hit] = space / rate, which is constant.
fn odds(len: usize, tried: u64, rate: f64) -> (f64, f64) {
    let space = 2f64.powi(5 * len as i32);
    let chance = (1.0 - (-(tried as f64) / space).exp()) * 100.0;
    let eta = if rate > 0.0 {
        space / rate
    } else {
        f64::INFINITY
    };
    (chance, eta)
}

fn fmt_pct(p: f64) -> String {
    if p >= 99.999 {
        ">99.99%".to_string()
    } else if p >= 0.001 {
        format!("{:.3}%", p)
    } else if p > 0.0 {
        format!("{:.2e}%", p)
    } else {
        "0%".to_string()
    }
}

fn fmt_eta(secs: f64) -> String {
    if !secs.is_finite() || secs <= 0.0 {
        return "—".to_string();
    }
    if secs < 90.0 {
        format!("{:.0}s", secs)
    } else if secs < 5400.0 {
        format!("{:.0}m", secs / 60.0)
    } else if secs < 172_800.0 {
        format!("{:.1}h", secs / 3600.0)
    } else if secs < 31_536_000.0 {
        format!("{:.1}d", secs / 86_400.0)
    } else {
        format!("{:.1}y", secs / 31_536_000.0)
    }
}

fn fmt_clock(secs: f64) -> String {
    let t = secs as u64;
    format!("{:02}:{:02}:{:02}", t / 3600, (t % 3600) / 60, t % 60)
}

fn trunc(s: &str, n: usize) -> String {
    if s.len() <= n {
        s.to_string()
    } else {
        format!("{}…", &s[..n - 1])
    }
}
