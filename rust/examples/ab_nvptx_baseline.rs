//! NVPTX half of the cuda-oxide A/B dump.
//!
//! Uses the **default** embedded PTX (`render_incremental`) with the same fixed
//! seed / 1×1 launch / 32-iter accept-all geometry as
//! `oxide_spike`'s `dump` mode, then prints sorted hex scalars.
//!
//! ```text
//! cargo +nightly run --release --example ab_nvptx_baseline --features cuda-oxide
//! ```
//!
//! Diff against: `(cd oxide_spike && cargo oxide run -- dump)`.

use rustacuda::launch;
use rustacuda::memory::{CopyDestination, DeviceBox, DeviceBuffer};
use rustacuda::prelude::*;
use std::ffi::CString;
use tor_v3_vanity_core as core;

/// Must match `oxide_spike::FIXED_SEED`.
const FIXED_SEED: [u8; 32] = [
    0x11, 0x22, 0x33, 0x44, 0x55, 0x66, 0x77, 0x88, 0x99, 0xaa, 0xbb, 0xcc, 0xdd, 0xee, 0xff, 0x01,
    0x02, 0x03, 0x04, 0x05, 0x06, 0x07, 0x08, 0x09, 0x0a, 0x0b, 0x0c, 0x0d, 0x0e, 0x0f, 0x10, 0x42,
];

fn hex(b: &[u8]) -> String {
    b.iter().map(|x| format!("{x:02x}")).collect()
}

fn main() {
    rustacuda::init(CudaFlags::empty()).unwrap();
    let device = Device::get_device(0).unwrap();
    let _ctx = Context::create_and_push(ContextFlags::MAP_HOST | ContextFlags::SCHED_AUTO, device)
        .unwrap();
    let module_data = CString::new(include_str!(env!("KERNEL_PTX_PATH"))).unwrap();
    let module = Module::load_from_string(&module_data).unwrap();
    let stream = Stream::new(StreamFlags::NON_BLOCKING, None).unwrap();

    let iters = 32u64;
    let mut gpu_seed = DeviceBuffer::from_slice(&FIXED_SEED).unwrap();
    // Accept-all prefix: last_byte_idx=0, mask=0.
    let mut prefix_bytes = DeviceBuffer::from_slice(&[0u8]).unwrap();
    let mut out_buf = DeviceBuffer::from_slice(&vec![0u8; core::OUT_SLOTS as usize * 32]).unwrap();
    let mut found = DeviceBox::new(&0u32).unwrap();

    let bp = core::BytePrefix {
        byte_prefix: prefix_bytes.as_device_ptr(),
        byte_prefix_len: 1,
        last_byte_idx: 0,
        last_byte_mask: 0,
        out: out_buf.as_device_ptr(),
        found: found.as_device_ptr(),
    };
    let mut gpu_bps = DeviceBuffer::from_slice(&[bp]).unwrap();
    let mut params = DeviceBox::new(&core::KernelParams {
        seed: gpu_seed.as_device_ptr(),
        byte_prefixes: gpu_bps.as_device_ptr(),
        byte_prefixes_len: 1,
        iters,
    })
    .unwrap();

    unsafe {
        launch!(module.render_incremental<<<1, 1, 0, stream>>>(params.as_device_ptr())).unwrap();
    }
    stream.synchronize().unwrap();

    let mut claimed = 0u32;
    found.copy_to(&mut claimed).unwrap();
    let mut ring = vec![0u8; core::OUT_SLOTS as usize * 32];
    out_buf.copy_to(&mut ring).unwrap();

    let stored = claimed.min(core::OUT_SLOTS) as usize;
    let mut lines: Vec<String> = ring.chunks_exact(32).take(stored).map(hex).collect();
    lines.sort();

    println!("backend=nvptx claimed={claimed}");
    for line in lines {
        println!("{line}");
    }
}
