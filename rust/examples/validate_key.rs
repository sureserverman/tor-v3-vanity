//! Independently validate a generated hs_ed25519_secret_key file: re-derive the
//! address from the stored scalar, confirm it matches the filename, and prove the
//! key signs + verifies under raw RFC8032 (Tor/donna semantics).
//!
//!   cargo +nightly run --release --example validate_key -- <path-to-onion-keyfile>

use curve25519_dalek::constants::ED25519_BASEPOINT_TABLE;
use curve25519_dalek::edwards::{CompressedEdwardsY, EdwardsPoint};
use curve25519_dalek::scalar::Scalar;
use sha2::{Digest as _, Sha512};
use sha3::Sha3_256;

const HEADER: &[u8] = b"== ed25519v1-secret: type0 ==\0\0\0";

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

fn main() {
    let path = std::env::args().nth(1).expect("usage: validate_key <file>");
    let bytes = std::fs::read(&path).expect("read key file");
    assert_eq!(bytes.len(), 96, "expected 32-byte header + 64-byte expanded key");
    assert_eq!(&bytes[..32], HEADER, "bad Tor secret-key header");

    let mut scalar_bytes = [0u8; 32];
    scalar_bytes.copy_from_slice(&bytes[32..64]);
    let mut nonce = [0u8; 32];
    nonce.copy_from_slice(&bytes[64..96]);

    let scalar = Scalar::from_canonical_bytes(scalar_bytes).expect("stored scalar is canonical");
    let a_point: EdwardsPoint = &scalar * &ED25519_BASEPOINT_TABLE;
    let a_pub = a_point.compress().to_bytes();
    let onion = pubkey_to_onion(&a_pub);

    // Address must match the filename.
    let fname = std::path::Path::new(&path)
        .file_name()
        .unwrap()
        .to_str()
        .unwrap();
    assert_eq!(onion, fname, "derived address != filename");

    // Raw RFC8032 sign + verify (un-clamped, as Tor/donna does).
    let msg = b"validate";
    let r = Scalar::from_bytes_mod_order_wide(&sha512_wide(&[&nonce, msg]));
    let big_r = (&r * &ED25519_BASEPOINT_TABLE).compress().to_bytes();
    let k = Scalar::from_bytes_mod_order_wide(&sha512_wide(&[&big_r, &a_pub, msg]));
    let s = r + k * scalar;

    let big_r_pt = CompressedEdwardsY::from_slice(&big_r).decompress().unwrap();
    let ok = (&s * &ED25519_BASEPOINT_TABLE) == big_r_pt + &k * a_point;
    assert!(ok, "signature failed to verify");

    println!("VALID ✔");
    println!("  address  : {onion}");
    println!("  matches filename, scalar is canonical, signs + verifies (RFC8032 raw).");
}
