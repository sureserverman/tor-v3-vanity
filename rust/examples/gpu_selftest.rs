//! Diagnostic: run the `selftest` kernel (compress(s·B) on the GPU) and compare
//! to curve25519-dalek. Isolates whether the GPU's field/point arithmetic agrees
//! with the CPU oracle (e.g. to catch u128 miscompilation on nvptx).
//!
//!   cargo +nightly run --release --example gpu_selftest

use curve25519_dalek::constants::ED25519_BASEPOINT_TABLE;
use curve25519_dalek::scalar::Scalar;
use rand::RngCore;
use rustacuda::launch;
use rustacuda::memory::{DeviceBox, DeviceBuffer};
use rustacuda::prelude::*;
use std::ffi::CString;
use tor_v3_vanity_core as core;

fn hex(b: &[u8]) -> String {
    b.iter().map(|x| format!("{:02x}", x)).collect()
}

fn main() {
    rustacuda::init(CudaFlags::empty()).unwrap();
    let device = Device::get_device(0).unwrap();
    let _ctx =
        Context::create_and_push(ContextFlags::MAP_HOST | ContextFlags::SCHED_AUTO, device).unwrap();
    let module_data = CString::new(include_str!(env!("KERNEL_PTX_PATH"))).unwrap();
    let module = Module::load_from_string(&module_data).unwrap();
    let stream = Stream::new(StreamFlags::NON_BLOCKING, None).unwrap();

    let walks: [u64; 8] = [0, 1, 2, 5, 100, 1000, 65537, 1_000_000];
    let mut fails = 0;
    for trial in 0..8 {
        let walk = walks[trial];
        let mut s = [0u8; 32];
        rand::thread_rng().fill_bytes(&mut s);

        let mut gpu_seed = DeviceBuffer::from_slice(&s).unwrap();
        let mut dummy = DeviceBuffer::from_slice(&[0u8; 1]).unwrap();
        let mut out_buf =
            DeviceBuffer::from_slice(&vec![0u8; core::OUT_SLOTS as usize * 32]).unwrap();
        let mut found = DeviceBox::new(&0u32).unwrap();

        let bp = core::BytePrefix {
            byte_prefix: dummy.as_device_ptr(),
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
            iters: walk,
        })
        .unwrap();

        unsafe {
            launch!(module.selftest<<<1, 1, 0, stream>>>(params.as_device_ptr())).unwrap();
        }
        stream.synchronize().unwrap();

        // The kernel writes into slot 0 of the ring; only that slot is of interest.
        let mut ring = vec![0u8; core::OUT_SLOTS as usize * 32];
        out_buf.copy_to(&mut ring).unwrap();
        let mut gpu_pub = [0u8; 32];
        gpu_pub.copy_from_slice(&ring[..32]);

        let scalar = Scalar::from_bytes_mod_order(s) + Scalar::from(walk);
        let want = (&scalar * &ED25519_BASEPOINT_TABLE).compress().to_bytes();

        if gpu_pub == want {
            println!("trial {trial}: walk={walk} PASS");
        } else {
            fails += 1;
            println!("trial {trial}: walk={walk} FAIL");
            println!("  gpu : {}", hex(&gpu_pub));
            println!("  want: {}", hex(&want));
        }
    }

    if fails == 0 {
        println!("\nGPU arithmetic matches curve25519-dalek — bug is elsewhere.");
    } else {
        println!("\n{fails}/8 mismatches — GPU field/point arithmetic is wrong on nvptx.");
    }
}
