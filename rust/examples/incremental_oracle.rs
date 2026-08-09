//! CPU reference / correctness oracle for the mkp224o-style *incremental* search.
//!
//! The GPU win is to stop hashing a fresh seed per candidate (which forces a full
//! scalar multiplication each time) and instead enumerate keys algebraically:
//! pick one random scalar `a`, compute `A = a·B` once, then walk
//! `A, A+B, A+2B, …` by repeated point **addition** (cheap), where the secret
//! scalar for step `k` is simply `a + k`.
//!
//! IMPORTANT correctness note this oracle established: Tor (via ed25519-donna,
//! same as mkp224o relies on) uses the stored 32-byte scalar **raw / un-clamped**.
//! `ed25519-dalek`'s ExpandedSecretKey→PublicKey instead *clamps* the bits, so it
//! is NOT a faithful model here and must not be used for the incremental path.
//! Everything below therefore uses raw curve25519-dalek group ops and hand-rolled
//! RFC8032 signing to mirror what Tor actually does with the on-disk key.
//!
//! Run on a machine with the toolchain (e.g. the GPU node):
//!   cargo +nightly run --release --example incremental_oracle

use curve25519_dalek::constants::{ED25519_BASEPOINT_POINT, ED25519_BASEPOINT_TABLE};
use curve25519_dalek::edwards::EdwardsPoint;
use curve25519_dalek::scalar::Scalar;
use rand::RngCore;
use sha2::{Digest as _, Sha512};
use sha3::Sha3_256;

const STEPS: u64 = 5000;
const TOR_SECRET_HEADER: &[u8] = b"== ed25519v1-secret: type0 ==\0\0\0";

fn sha512_wide(parts: &[&[u8]]) -> [u8; 64] {
    let mut h = Sha512::new();
    for p in parts {
        h.update(p);
    }
    let mut out = [0u8; 64];
    out.copy_from_slice(&h.finalize());
    out
}

fn pubkey_to_onion(pubkey: &[u8; 32]) -> String {
    use sha3::Digest as _;
    let mut hasher = Sha3_256::new();
    hasher.update(b".onion checksum");
    hasher.update(pubkey);
    hasher.update(&[3]);
    let mut onion = [0u8; 35];
    onion[..32].copy_from_slice(pubkey);
    onion[32..34].copy_from_slice(&hasher.finalize()[..2]);
    onion[34] = 3;
    format!(
        "{}.onion",
        base32::encode(base32::Alphabet::RFC4648 { padding: false }, &onion).to_lowercase()
    )
}

/// RFC8032 signing using the raw scalar `a` and prefix/nonce, exactly as Tor's
/// donna does with the stored expanded key (no clamping). Returns (R||S).
fn raw_sign(a: &Scalar, nonce: &[u8; 32], a_pub: &[u8; 32], msg: &[u8]) -> [u8; 64] {
    let r = Scalar::from_bytes_mod_order_wide(&sha512_wide(&[nonce, msg]));
    let big_r = (&r * &ED25519_BASEPOINT_TABLE).compress().to_bytes();
    let k = Scalar::from_bytes_mod_order_wide(&sha512_wide(&[&big_r, a_pub, msg]));
    let s = r + k * a;
    let mut sig = [0u8; 64];
    sig[..32].copy_from_slice(&big_r);
    sig[32..].copy_from_slice(s.as_bytes());
    sig
}

/// Standard ed25519 verification: check S·B == R + k·A.
fn verify(a_point: &EdwardsPoint, a_pub: &[u8; 32], msg: &[u8], sig: &[u8; 64]) -> bool {
    let big_r = match curve25519_dalek::edwards::CompressedEdwardsY::from_slice(&sig[..32]).decompress()
    {
        Some(p) => p,
        None => return false,
    };
    let mut s_bytes = [0u8; 32];
    s_bytes.copy_from_slice(&sig[32..]);
    let s = match Scalar::from_canonical_bytes(s_bytes) {
        Some(s) => s,
        None => return false,
    };
    let k = Scalar::from_bytes_mod_order_wide(&sha512_wide(&[&sig[..32], a_pub, msg]));
    (&s * &ED25519_BASEPOINT_TABLE) == (big_r + &k * a_point)
}

fn main() {
    let mut rng = rand::thread_rng();

    // Random per-"thread" starting scalar a, and an arbitrary nonce.
    let mut seed_bytes = [0u8; 32];
    rng.fill_bytes(&mut seed_bytes);
    let a = Scalar::from_bytes_mod_order(seed_bytes);
    let mut nonce = [0u8; 32];
    rng.fill_bytes(&mut nonce);

    // A = a·B computed once; thereafter we only add B (the incremental hot path).
    let mut point = &a * &ED25519_BASEPOINT_TABLE;
    let basepoint = ED25519_BASEPOINT_POINT;

    let msg = b"tor-v3-vanity incremental oracle";
    let mut sample_onion = String::new();

    for k in 0..STEPS {
        // Secret scalar for this step (Scalar arithmetic is mod L, so this stays
        // canonical even if a + k wrapped past the group order).
        let scalar_k = a + Scalar::from(k);

        // Public key two ways: (1) the incrementally-added point, and (2) a fresh
        // raw scalar-mult of (a+k). With Tor's raw-scalar semantics these MUST be
        // equal — this is the core invariant the GPU walk depends on.
        let pub_from_walk = point.compress().to_bytes();
        let pub_from_scalar = (&scalar_k * &ED25519_BASEPOINT_TABLE).compress().to_bytes();
        assert_eq!(
            pub_from_walk, pub_from_scalar,
            "step {k}: A+{k}·B != (a+{k})·B — incremental enumeration is WRONG"
        );

        // The stored raw scalar must produce a key that signs & verifies (this is
        // what makes it a usable Tor HS key, not just a matching address).
        let sig = raw_sign(&scalar_k, &nonce, &pub_from_walk, msg);
        assert!(
            verify(&point, &pub_from_walk, msg, &sig),
            "step {k}: raw-scalar signature failed to verify"
        );

        if k == 0 {
            sample_onion = pubkey_to_onion(&pub_from_walk);

            // The exact on-disk file the GPU host path will write: 32-byte header
            // + raw scalar (32) + nonce (32). Reload it and confirm the scalar
            // reproduces the same address under raw-scalar derivation.
            let mut file = Vec::new();
            file.extend_from_slice(TOR_SECRET_HEADER);
            file.extend_from_slice(scalar_k.as_bytes());
            file.extend_from_slice(&nonce);
            assert_eq!(file.len(), 32 + 64, "unexpected secret key file length");

            let mut reloaded = [0u8; 32];
            reloaded.copy_from_slice(&file[32..64]);
            let scalar_reloaded =
                Scalar::from_canonical_bytes(reloaded).expect("stored scalar is canonical");
            let pub_reloaded = (&scalar_reloaded * &ED25519_BASEPOINT_TABLE).compress().to_bytes();
            assert_eq!(
                pubkey_to_onion(&pub_reloaded),
                sample_onion,
                "on-disk key file did not round-trip to the same address"
            );
        }

        point = point + basepoint;
    }

    println!("OK: {STEPS} incremental keys validated.");
    println!("  • every A+k·B == (a+k)·B  (incremental walk is sound)");
    println!("  • every raw-scalar key signed + verified (RFC8032, un-clamped, Tor semantics)");
    println!("  • the header+scalar+nonce key file round-tripped to the same address");
    println!("  sample address (k=0): {sample_onion}");
}
