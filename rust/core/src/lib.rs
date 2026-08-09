#![cfg_attr(target_arch = "nvptx64", no_std)]
#![cfg_attr(target_arch = "nvptx64", feature(abi_ptx, stdarch_nvptx))]

use rustacuda_core::DevicePointer;
use rustacuda_derive::DeviceCopy;

#[cfg(target_arch = "nvptx64")]
mod kernel;

#[derive(DeviceCopy, Clone)]
#[repr(C)]
pub struct KernelParams {
    pub seed: DevicePointer<u8>,
    pub byte_prefixes: DevicePointer<BytePrefix>,
    pub byte_prefixes_len: usize,
    /// Number of keygen iterations each thread performs per launch (grid-stride).
    /// Amortizes launch/sync/host round-trip overhead across many keys.
    pub iters: u64,
}

/// Matches recorded per prefix per launch. Threads claim a slot with an atomic
/// increment, so concurrent matches never overwrite each other. Matches past the
/// last slot still bump the counter, so the host can report how many it dropped
/// instead of silently pretending the launch found fewer keys than it did.
pub const OUT_SLOTS: u32 = 64;

#[derive(DeviceCopy, Clone)]
#[repr(C)]
pub struct BytePrefix {
    pub byte_prefix: DevicePointer<u8>,
    pub byte_prefix_len: usize,
    pub last_byte_idx: usize,
    pub last_byte_mask: u8,
    /// Ring of `OUT_SLOTS` 32-byte results, one per claimed match.
    pub out: DevicePointer<u8>,
    /// Atomic claim counter: total matches this launch, which may exceed `OUT_SLOTS`.
    pub found: DevicePointer<u32>,
}
impl BytePrefix {
    /// `last_byte_idx` is the number of fully-constrained leading bytes; when the
    /// prefix doesn't end on a byte boundary, `last_byte_mask` constrains the high
    /// bits of the next byte (mask 0 = byte-aligned prefix, no partial byte).
    #[inline(always)]
    pub fn matches(&self, data: &[u8]) -> bool {
        let slice =
            unsafe { core::slice::from_raw_parts(self.byte_prefix.as_raw(), self.byte_prefix_len) };
        // Manual byte compare instead of `data.starts_with(..)`, which lowers to a
        // memcmp call — too costly per candidate in the GPU hot loop.
        let mut i = 0;
        while i < self.last_byte_idx {
            if data[i] != slice[i] {
                return false;
            }
            i += 1;
        }
        self.last_byte_mask == 0
            || data[self.last_byte_idx] & self.last_byte_mask == slice[self.last_byte_idx]
    }

    /// Record a match. Claims a slot with an atomic increment so simultaneous
    /// matches from different threads can't tear each other's 32-byte writes.
    #[cfg(target_arch = "nvptx64")]
    #[inline(always)]
    pub fn record(&mut self, data: &[u8; 32]) {
        use core::sync::atomic::{AtomicU32, Ordering};
        let counter = unsafe { &*(self.found.as_raw() as *const AtomicU32) };
        let slot = counter.fetch_add(1, Ordering::Relaxed);
        if slot < OUT_SLOTS {
            let base = unsafe { self.out.as_raw_mut().add(slot as usize * 32) };
            unsafe { core::slice::from_raw_parts_mut(base, 32) }.clone_from_slice(data);
        }
    }
}
