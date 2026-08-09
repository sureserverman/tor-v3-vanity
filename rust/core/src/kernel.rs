use super::KernelParams;
use byteorder::{ByteOrder, LittleEndian};

#[inline]
fn add_u256(base: &[u8; 32], mut offset: u64) -> [u8; 32] {
    let mut res = [0; 32];
    for i in 0..4 {
        let start = i * 8;
        let end = (i + 1) * 8;
        let base = LittleEndian::read_u64(&base[start..end]);
        let (total, overflow) = base.overflowing_add(offset);
        LittleEndian::write_u64(&mut res[start..end], total);
        if overflow {
            offset = 1;
        } else {
            offset = 0;
        }
    }
    res
}

#[no_mangle]
pub extern "ptx-kernel" fn render(params_ptr: *mut KernelParams) {
    use core::arch::nvptx::*;

    let params = unsafe { &mut *params_ptr };
    // Global thread index and the total grid size (grid-stride loop step).
    let x = unsafe { _block_dim_x() * _block_idx_x() + _thread_idx_x() } as u64;
    let stride = unsafe { _block_dim_x() * _grid_dim_x() } as u64;

    let seed = unsafe { core::slice::from_raw_parts(params.seed.as_raw(), 32) }
        .try_into()
        .unwrap();

    let byte_prefixes = unsafe {
        core::slice::from_raw_parts_mut(params.byte_prefixes.as_raw_mut(), params.byte_prefixes_len)
    };

    // Grid-stride: each thread does `iters` candidates, stepping by the grid size,
    // so launches stay large and per-launch overhead is amortized.
    let mut idx = x;
    for _ in 0..params.iters {
        let cur_seed = add_u256(seed, idx);
        let s = ed25519_compact::Seed::new(cur_seed);
        let kp = ed25519_compact::KeyPair::from_seed(s);

        for byte_prefix in byte_prefixes.iter_mut() {
            if byte_prefix.matches(&*kp.pk) {
                byte_prefix.record(&cur_seed);
            }
        }
        idx = idx.wrapping_add(stride);
    }
}

// Incremental (mkp224o-style) search: each thread computes A = a·B once, then walks
// A, A+B, A+2B, ... by point addition; the secret scalar for step j is a+j, stored
// raw (Tor uses it un-clamped). Radix-2^51 ed25519, validated on CPU against
// curve25519-dalek (see examples/field_oracle.rs).

type Fe = [u64; 5];
const FE_MASK: u64 = (1u64 << 51) - 1;
const FE_ONE: Fe = [1, 0, 0, 0, 0];
const EDWARDS_D2: Fe = [
    1859910466990425,
    932731440258426,
    1072319116312658,
    1815898335770999,
    633789495995903,
];

#[derive(Clone, Copy)]
struct Pt {
    x: Fe,
    y: Fe,
    z: Fe,
    t: Fe,
}

const BASEPOINT: Pt = Pt {
    x: [
        1738742601995546,
        1146398526822698,
        2070867633025821,
        562264141797630,
        587772402128613,
    ],
    y: [
        1801439850948184,
        1351079888211148,
        450359962737049,
        900719925474099,
        1801439850948198,
    ],
    z: FE_ONE,
    t: [
        1841354044333475,
        16398895984059,
        755974180946558,
        900171276175154,
        1821297809914039,
    ],
};

const IDENTITY: Pt = Pt {
    x: [0; 5],
    y: FE_ONE,
    z: FE_ONE,
    t: [0; 5],
};

fn fe_reduce(mut l: [u64; 5]) -> Fe {
    let c0 = l[0] >> 51;
    let c1 = l[1] >> 51;
    let c2 = l[2] >> 51;
    let c3 = l[3] >> 51;
    let c4 = l[4] >> 51;
    l[0] &= FE_MASK;
    l[1] &= FE_MASK;
    l[2] &= FE_MASK;
    l[3] &= FE_MASK;
    l[4] &= FE_MASK;
    l[0] += c4 * 19;
    l[1] += c0;
    l[2] += c1;
    l[3] += c2;
    l[4] += c3;
    l
}

fn fe_add(a: &Fe, b: &Fe) -> Fe {
    fe_reduce([
        a[0] + b[0],
        a[1] + b[1],
        a[2] + b[2],
        a[3] + b[3],
        a[4] + b[4],
    ])
}

fn fe_sub(a: &Fe, b: &Fe) -> Fe {
    fe_reduce([
        (a[0] + 36028797018963664) - b[0],
        (a[1] + 36028797018963952) - b[1],
        (a[2] + 36028797018963952) - b[2],
        (a[3] + 36028797018963952) - b[3],
        (a[4] + 36028797018963952) - b[4],
    ])
}

fn fe_mul(a: &Fe, b: &Fe) -> Fe {
    let (a0, a1, a2, a3, a4) = (
        a[0] as u128,
        a[1] as u128,
        a[2] as u128,
        a[3] as u128,
        a[4] as u128,
    );
    let (b0, b1, b2, b3, b4) = (
        b[0] as u128,
        b[1] as u128,
        b[2] as u128,
        b[3] as u128,
        b[4] as u128,
    );
    let b1_19 = b1 * 19;
    let b2_19 = b2 * 19;
    let b3_19 = b3 * 19;
    let b4_19 = b4 * 19;

    let c0 = a0 * b0 + a1 * b4_19 + a2 * b3_19 + a3 * b2_19 + a4 * b1_19;
    let c1 = a0 * b1 + a1 * b0 + a2 * b4_19 + a3 * b3_19 + a4 * b2_19;
    let c2 = a0 * b2 + a1 * b1 + a2 * b0 + a3 * b4_19 + a4 * b3_19;
    let c3 = a0 * b3 + a1 * b2 + a2 * b1 + a3 * b0 + a4 * b4_19;
    let c4 = a0 * b4 + a1 * b3 + a2 * b2 + a3 * b1 + a4 * b0;

    let m = (1u128 << 51) - 1;
    let c1 = c1 + (c0 >> 51);
    let r0 = (c0 & m) as u64;
    let c2 = c2 + (c1 >> 51);
    let r1 = (c1 & m) as u64;
    let c3 = c3 + (c2 >> 51);
    let r2 = (c2 & m) as u64;
    let c4 = c4 + (c3 >> 51);
    let r3 = (c3 & m) as u64;
    let carry = (c4 >> 51) as u64;
    let r4 = (c4 & m) as u64;

    let mut r0 = r0 + carry * 19;
    let r1 = r1 + (r0 >> 51);
    r0 &= FE_MASK;
    [r0, r1, r2, r3, r4]
}

fn fe_sq(a: &Fe) -> Fe {
    // Dedicated squaring: fewer cross terms than fe_mul(a, a).
    let a0 = a[0] as u128;
    let a1 = a[1] as u128;
    let a2 = a[2] as u128;
    let a3 = a[3] as u128;
    let a4 = a[4] as u128;
    let a3_19 = 19 * a3;
    let a4_19 = 19 * a4;

    let c0 = a0 * a0 + 2 * (a1 * a4_19 + a2 * a3_19);
    let c1 = a3 * a3_19 + 2 * (a0 * a1 + a2 * a4_19);
    let c2 = a1 * a1 + 2 * (a0 * a2 + a4 * a3_19);
    let c3 = a4 * a4_19 + 2 * (a0 * a3 + a1 * a2);
    let c4 = a2 * a2 + 2 * (a0 * a4 + a1 * a3);

    let m = (1u128 << 51) - 1;
    let c1 = c1 + (c0 >> 51);
    let r0 = (c0 & m) as u64;
    let c2 = c2 + (c1 >> 51);
    let r1 = (c1 & m) as u64;
    let c3 = c3 + (c2 >> 51);
    let r2 = (c2 & m) as u64;
    let c4 = c4 + (c3 >> 51);
    let r3 = (c3 & m) as u64;
    let carry = (c4 >> 51) as u64;
    let r4 = (c4 & m) as u64;

    let mut r0 = r0 + carry * 19;
    let r1 = r1 + (r0 >> 51);
    r0 &= FE_MASK;
    [r0, r1, r2, r3, r4]
}

fn fe_pow2k(a: &Fe, k: u32) -> Fe {
    let mut r = *a;
    let mut i = 0;
    while i < k {
        r = fe_sq(&r);
        i += 1;
    }
    r
}

#[inline(always)]
fn fe_invert(z: &Fe) -> Fe {
    let t0 = fe_sq(z);
    let t1 = fe_sq(&fe_sq(&t0));
    let t2 = fe_mul(z, &t1);
    let t3 = fe_mul(&t0, &t2);
    let t4 = fe_sq(&t3);
    let t5 = fe_mul(&t2, &t4);
    let t6 = fe_pow2k(&t5, 5);
    let t7 = fe_mul(&t6, &t5);
    let t8 = fe_pow2k(&t7, 10);
    let t9 = fe_mul(&t8, &t7);
    let t10 = fe_pow2k(&t9, 20);
    let t11 = fe_mul(&t10, &t9);
    let t12 = fe_pow2k(&t11, 10);
    let t13 = fe_mul(&t12, &t7);
    let t14 = fe_pow2k(&t13, 50);
    let t15 = fe_mul(&t14, &t13);
    let t16 = fe_pow2k(&t15, 100);
    let t17 = fe_mul(&t16, &t15);
    let t18 = fe_pow2k(&t17, 50);
    let t19 = fe_mul(&t18, &t13);
    let t20 = fe_pow2k(&t19, 5);
    fe_mul(&t20, &t3)
}

fn fe_to_bytes(f: &Fe) -> [u8; 32] {
    let mut l = fe_reduce(*f);
    let mut q = (l[0] + 19) >> 51;
    q = (l[1] + q) >> 51;
    q = (l[2] + q) >> 51;
    q = (l[3] + q) >> 51;
    q = (l[4] + q) >> 51;
    l[0] += 19 * q;
    l[1] += l[0] >> 51;
    l[0] &= FE_MASK;
    l[2] += l[1] >> 51;
    l[1] &= FE_MASK;
    l[3] += l[2] >> 51;
    l[2] &= FE_MASK;
    l[4] += l[3] >> 51;
    l[3] &= FE_MASK;
    l[4] &= FE_MASK;

    let mut s = [0u8; 32];
    s[0] = l[0] as u8;
    s[1] = (l[0] >> 8) as u8;
    s[2] = (l[0] >> 16) as u8;
    s[3] = (l[0] >> 24) as u8;
    s[4] = (l[0] >> 32) as u8;
    s[5] = (l[0] >> 40) as u8;
    s[6] = ((l[0] >> 48) | (l[1] << 3)) as u8;
    s[7] = (l[1] >> 5) as u8;
    s[8] = (l[1] >> 13) as u8;
    s[9] = (l[1] >> 21) as u8;
    s[10] = (l[1] >> 29) as u8;
    s[11] = (l[1] >> 37) as u8;
    s[12] = ((l[1] >> 45) | (l[2] << 6)) as u8;
    s[13] = (l[2] >> 2) as u8;
    s[14] = (l[2] >> 10) as u8;
    s[15] = (l[2] >> 18) as u8;
    s[16] = (l[2] >> 26) as u8;
    s[17] = (l[2] >> 34) as u8;
    s[18] = (l[2] >> 42) as u8;
    s[19] = ((l[2] >> 50) | (l[3] << 1)) as u8;
    s[20] = (l[3] >> 7) as u8;
    s[21] = (l[3] >> 15) as u8;
    s[22] = (l[3] >> 23) as u8;
    s[23] = (l[3] >> 31) as u8;
    s[24] = (l[3] >> 39) as u8;
    s[25] = ((l[3] >> 47) | (l[4] << 4)) as u8;
    s[26] = (l[4] >> 4) as u8;
    s[27] = (l[4] >> 12) as u8;
    s[28] = (l[4] >> 20) as u8;
    s[29] = (l[4] >> 28) as u8;
    s[30] = (l[4] >> 36) as u8;
    s[31] = (l[4] >> 44) as u8;
    s
}

#[inline(always)]
fn pt_add(p: &Pt, q: &Pt) -> Pt {
    let a = fe_mul(&fe_sub(&p.y, &p.x), &fe_sub(&q.y, &q.x));
    let b = fe_mul(&fe_add(&p.y, &p.x), &fe_add(&q.y, &q.x));
    let c = fe_mul(&fe_mul(&p.t, &q.t), &EDWARDS_D2);
    let d = fe_mul(&fe_add(&p.z, &p.z), &q.z);
    let e = fe_sub(&b, &a);
    let f = fe_sub(&d, &c);
    let g = fe_add(&d, &c);
    let h = fe_add(&b, &a);
    Pt {
        x: fe_mul(&e, &f),
        y: fe_mul(&g, &h),
        t: fe_mul(&e, &h),
        z: fe_mul(&f, &g),
    }
}

/// Basepoint terms that are constant across the whole walk: (B.y-B.x), (B.y+B.x),
/// and B.t*2d. Computed once per thread, reused by every `+B` step.
fn base_cached() -> (Fe, Fe, Fe) {
    (
        fe_sub(&BASEPOINT.y, &BASEPOINT.x),
        fe_add(&BASEPOINT.y, &BASEPOINT.x),
        fe_mul(&BASEPOINT.t, &EDWARDS_D2),
    )
}

/// p + B using the precomputed base terms. Same as pt_add(p, B) but skips the
/// constant recomputation and the multiply by B.z (= 1): 7 muls instead of 9.
fn pt_add_base(p: &Pt, b_sub: &Fe, b_add: &Fe, b_d2t: &Fe) -> Pt {
    let a = fe_mul(&fe_sub(&p.y, &p.x), b_sub);
    let b = fe_mul(&fe_add(&p.y, &p.x), b_add);
    let c = fe_mul(&p.t, b_d2t);
    let d = fe_add(&p.z, &p.z); // 2 * p.z * B.z, and B.z = 1
    let e = fe_sub(&b, &a);
    let f = fe_sub(&d, &c);
    let g = fe_add(&d, &c);
    let h = fe_add(&b, &a);
    Pt {
        x: fe_mul(&e, &f),
        y: fe_mul(&g, &h),
        t: fe_mul(&e, &h),
        z: fe_mul(&f, &g),
    }
}

fn scalar_mul_base(scalar: &[u8; 32]) -> Pt {
    let mut r = IDENTITY;
    let mut i: i32 = 255;
    while i >= 0 {
        r = pt_add(&r, &r);
        if (scalar[(i >> 3) as usize] >> (i & 7)) & 1 == 1 {
            r = pt_add(&r, &BASEPOINT);
        }
        i -= 1;
    }
    r
}

fn compress(p: &Pt) -> [u8; 32] {
    let zinv = fe_invert(&p.z);
    let x = fe_mul(&p.x, &zinv);
    let y = fe_mul(&p.y, &zinv);
    let mut s = fe_to_bytes(&y);
    s[31] |= (fe_to_bytes(&x)[0] & 1) << 7;
    s
}

/// Montgomery batch inversion of `zs[..n]` in place: one real inversion plus ~3n
/// muls. `scratch` must hold `n` elements.
fn fe_batch_invert(zs: &mut [Fe], scratch: &mut [Fe], n: usize) {
    let mut acc = FE_ONE;
    let mut i = 0;
    while i < n {
        scratch[i] = acc;
        acc = fe_mul(&acc, &zs[i]);
        i += 1;
    }
    acc = fe_invert(&acc);
    let mut i = n;
    while i > 0 {
        i -= 1;
        let inv = fe_mul(&acc, &scratch[i]);
        acc = fe_mul(&acc, &zs[i]);
        zs[i] = inv;
    }
}

#[no_mangle]
pub extern "ptx-kernel" fn render_incremental(params_ptr: *mut KernelParams) {
    use core::arch::nvptx::*;

    let params = unsafe { &mut *params_ptr };
    let tid = unsafe { _block_dim_x() * _block_idx_x() + _thread_idx_x() } as u64;
    let iters = params.iters;
    // Each thread owns a disjoint range [offset, offset+iters) of scalars.
    let offset = tid.wrapping_mul(iters);

    let base: &[u8; 32] = unsafe { core::slice::from_raw_parts(params.seed.as_raw(), 32) }
        .try_into()
        .unwrap();

    // A = (base + offset)·B, computed once; thereafter only point additions.
    let a_bytes = add_u256(base, offset);
    let mut p = scalar_mul_base(&a_bytes);
    let (b_sub, b_add, b_d2t) = base_cached();

    let byte_prefixes = unsafe {
        core::slice::from_raw_parts_mut(params.byte_prefixes.as_raw_mut(), params.byte_prefixes_len)
    };

    // Walk in windows of WINDOW, one batched inversion per window. Only Y,Z are
    // needed: the prefix lives in the low bytes of affine y (host re-derives the key).
    const WINDOW: usize = 32;
    let mut ys = [[0u64; 5]; WINDOW];
    let mut zs = [[0u64; 5]; WINDOW];
    let mut scratch = [[0u64; 5]; WINDOW];

    let mut j: u64 = 0;
    while j < iters {
        let remaining = iters - j;
        let w = if remaining < WINDOW as u64 {
            remaining as usize
        } else {
            WINDOW
        };

        // Snapshot Y,Z for the window and advance the walk by one point each step.
        let mut i = 0;
        while i < w {
            ys[i] = p.y;
            zs[i] = p.z;
            p = pt_add_base(&p, &b_sub, &b_add, &b_d2t);
            i += 1;
        }

        // One inversion for the whole window: zs[i] becomes 1/Z_i.
        fe_batch_invert(&mut zs, &mut scratch, w);

        let mut i = 0;
        while i < w {
            let y_aff = fe_mul(&ys[i], &zs[i]); // affine y = Y / Z
            let yb = fe_to_bytes(&y_aff);
            for byte_prefix in byte_prefixes.iter_mut() {
                if byte_prefix.matches(&yb) {
                    // Store the raw scalar (base + offset + j + i); host reduces mod L.
                    let scalar = add_u256(base, offset + j + i as u64);
                    byte_prefix.record(&scalar);
                }
            }
            i += 1;
        }

        j += w as u64;
    }
}

/// Diagnostic: write compress(seed·B) into byte_prefixes[0].out so the host can
/// compare GPU arithmetic against curve25519-dalek. Launch with one thread.
#[no_mangle]
pub extern "ptx-kernel" fn selftest(params_ptr: *mut KernelParams) {
    let params = unsafe { &mut *params_ptr };
    let base: &[u8; 32] = unsafe { core::slice::from_raw_parts(params.seed.as_raw(), 32) }
        .try_into()
        .unwrap();
    // Walk params.iters steps from base·B, mirroring render_incremental, so the
    // host can check the walk lands on (base + iters)·B.
    let mut p = scalar_mul_base(base);
    let (b_sub, b_add, b_d2t) = base_cached();
    let mut step: u64 = 0;
    while step < params.iters {
        p = pt_add_base(&p, &b_sub, &b_add, &b_d2t);
        step += 1;
    }
    let c = compress(&p);

    let byte_prefixes = unsafe {
        core::slice::from_raw_parts_mut(params.byte_prefixes.as_raw_mut(), params.byte_prefixes_len)
    };
    byte_prefixes[0].record(&c);
}

#[panic_handler]
fn panic(_: &::core::panic::PanicInfo) -> ! {
    use core::arch::nvptx::*;

    unsafe { trap() }
}
