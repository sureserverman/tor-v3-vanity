//! Step 1 of the GPU rewrite: a from-scratch radix-2^51 ed25519 field + point
//! implementation, validated against curve25519-dalek.
//!
//! This is the *exact arithmetic* that will be ported to the nvptx64 kernel
//! (curve25519-dalek can't compile to nvptx no_std). Everything here is plain
//! `[u64;5]` / `u128` with no external crypto, so it ports almost verbatim. We
//! prove it on CPU — where we can diff against a trusted reference at the
//! compressed-point level — before any of it touches the GPU.
//!
//! Validates:
//!   • field inversion: x · x⁻¹ == 1
//!   • fixed-base scalar mult: a·B matches curve25519-dalek for random a
//!   • incremental point addition: (a·B) + k·B == (a+k)·B for many k
//!
//! Run on the node:  cargo +nightly run --release --example field_oracle

use curve25519_dalek::constants::ED25519_BASEPOINT_TABLE;
use curve25519_dalek::scalar::Scalar;
use rand::RngCore;

type Fe = [u64; 5];
const MASK: u64 = (1u64 << 51) - 1;

const ONE: Fe = [1, 0, 0, 0, 0];

// 2*d, the Edwards curve constant (curve25519-dalek EDWARDS_D2).
const D2: Fe = [
    1859910466990425,
    932731440258426,
    1072319116312658,
    1815898335770999,
    633789495995903,
];

// ed25519 basepoint B in extended coordinates (X:Y:Z:T), Z=1.
const BX: Fe = [
    1738742601995546,
    1146398526822698,
    2070867633025821,
    562264141797630,
    587772402128613,
];
const BY: Fe = [
    1801439850948184,
    1351079888211148,
    450359962737049,
    900719925474099,
    1801439850948198,
];
const BT: Fe = [
    1841354044333475,
    16398895984059,
    755974180946558,
    900171276175154,
    1821297809914039,
];

#[derive(Clone)]
struct Pt {
    x: Fe,
    y: Fe,
    z: Fe,
    t: Fe,
}

fn identity() -> Pt {
    Pt { x: [0; 5], y: ONE, z: ONE, t: [0; 5] }
}
fn basepoint() -> Pt {
    Pt { x: BX, y: BY, z: ONE, t: BT }
}

// Single-pass weak reduction: limbs in -> limbs < ~2^51.
fn fe_reduce(mut l: [u64; 5]) -> Fe {
    let c0 = l[0] >> 51;
    let c1 = l[1] >> 51;
    let c2 = l[2] >> 51;
    let c3 = l[3] >> 51;
    let c4 = l[4] >> 51;
    l[0] &= MASK;
    l[1] &= MASK;
    l[2] &= MASK;
    l[3] &= MASK;
    l[4] &= MASK;
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

// a - b, computed as a + 16p - b so limbs never underflow, then reduced.
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
    r0 &= MASK;
    [r0, r1, r2, r3, r4]
}

fn fe_sq(a: &Fe) -> Fe {
    fe_mul(a, a)
}
fn fe_pow2k(a: &Fe, k: u32) -> Fe {
    let mut r = *a;
    for _ in 0..k {
        r = fe_sq(&r);
    }
    r
}

// x^(p-2) via the standard curve25519-dalek addition chain.
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
    // Canonicalize (subtract p if needed).
    let mut q = (l[0] + 19) >> 51;
    q = (l[1] + q) >> 51;
    q = (l[2] + q) >> 51;
    q = (l[3] + q) >> 51;
    q = (l[4] + q) >> 51;
    l[0] += 19 * q;
    l[1] += l[0] >> 51;
    l[0] &= MASK;
    l[2] += l[1] >> 51;
    l[1] &= MASK;
    l[3] += l[2] >> 51;
    l[2] &= MASK;
    l[4] += l[3] >> 51;
    l[3] &= MASK;
    l[4] &= MASK;

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

fn fe_from_bytes(b: &[u8; 32]) -> Fe {
    let load8 = |i: usize| -> u64 {
        let mut x = 0u64;
        for j in 0..8 {
            x |= (b[i + j] as u64) << (8 * j);
        }
        x
    };
    [
        load8(0) & MASK,
        (load8(6) >> 3) & MASK,
        (load8(12) >> 6) & MASK,
        (load8(19) >> 1) & MASK,
        (load8(24) >> 12) & MASK,
    ]
}

// Unified extended-coordinate addition (Hisil–Wong–Carter–Dawson), valid for
// all inputs including doubling.
fn pt_add(p: &Pt, q: &Pt) -> Pt {
    let a = fe_mul(&fe_sub(&p.y, &p.x), &fe_sub(&q.y, &q.x));
    let b = fe_mul(&fe_add(&p.y, &p.x), &fe_add(&q.y, &q.x));
    let c = fe_mul(&fe_mul(&p.t, &q.t), &D2);
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

// Double-and-add fixed-base scalar mult (MSB first). One per thread on the GPU.
fn scalar_mul(point: &Pt, scalar: &[u8; 32]) -> Pt {
    let mut r = identity();
    for i in (0..256).rev() {
        r = pt_add(&r, &r);
        if (scalar[i >> 3] >> (i & 7)) & 1 == 1 {
            r = pt_add(&r, point);
        }
    }
    r
}

// Affine y-encoding with the x sign bit — the 32-byte public key / address bytes.
fn compress(p: &Pt) -> [u8; 32] {
    let zinv = fe_invert(&p.z);
    let x = fe_mul(&p.x, &zinv);
    let y = fe_mul(&p.y, &zinv);
    let mut s = fe_to_bytes(&y);
    s[31] |= (fe_to_bytes(&x)[0] & 1) << 7;
    s
}

fn main() {
    let mut rng = rand::thread_rng();

    // 1) Field inversion sanity over random elements.
    for _ in 0..1000 {
        let mut b = [0u8; 32];
        rng.fill_bytes(&mut b);
        let x = fe_from_bytes(&b);
        let prod = fe_mul(&x, &fe_invert(&x));
        assert_eq!(
            fe_to_bytes(&prod),
            fe_to_bytes(&ONE),
            "x * x^-1 != 1 — field inversion/mul is wrong"
        );
    }

    // 2) Fixed-base scalar mult matches curve25519-dalek.
    let mut seed = [0u8; 32];
    rng.fill_bytes(&mut seed);
    let a = Scalar::from_bytes_mod_order(seed);
    let a_bytes = a.to_bytes();

    let mine = scalar_mul(&basepoint(), &a_bytes);
    let mine_c = compress(&mine);
    let theirs_c = (&a * &ED25519_BASEPOINT_TABLE).compress().to_bytes();
    assert_eq!(
        mine_c, theirs_c,
        "a·B disagrees with curve25519-dalek — field/point/B-constant bug"
    );

    // 3) Incremental walk: A + k·B must equal (a+k)·B for every step.
    const STEPS: u64 = 2000;
    let b = basepoint();
    let mut point = mine;
    for k in 0..STEPS {
        let scalar_k = a + Scalar::from(k);
        let expected = (&scalar_k * &ED25519_BASEPOINT_TABLE).compress().to_bytes();
        assert_eq!(
            compress(&point),
            expected,
            "step {k}: incremental A+{k}·B != (a+{k})·B"
        );
        point = pt_add(&point, &b);
    }

    println!("OK: from-scratch radix-2^51 ed25519 validated against curve25519-dalek.");
    println!("  • 1000 random field inversions correct (x·x⁻¹ == 1)");
    println!("  • fixed-base a·B matches the reference");
    println!("  • {STEPS} incremental point additions match (a+k)·B exactly");
    println!("  → this arithmetic is ready to port to the nvptx64 kernel.");
}
