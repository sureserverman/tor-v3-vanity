//! Build the PTX kernel using the official llvm-bitcode-linker (no ptx-linker).
//!
//! Requires nightly and: rustup component add llvm-bitcode-linker llvm-tools rust-src --toolchain nightly

use std::env;
use std::path::PathBuf;
use std::process::Command;

/// One-time setup command to print when the kernel build fails (e.g. missing rust-src).
const SETUP_CMD: &str = "rustup install nightly && rustup target add nvptx64-nvidia-cuda --toolchain nightly && rustup component add llvm-bitcode-linker llvm-tools rust-src --toolchain nightly";

/// CUDA lib path: CUDA_LIB_DIR (exact), or CUDA_HOME/lib64 or CUDA_PATH/lib64, or default.
/// For cross-compiling: AARCH64_CUDA_LIB_DIR (target aarch64), X86_64_CUDA_LIB_DIR (target x86_64 on aarch64 host).
fn cuda_lib_dir() -> String {
    let target = env::var("TARGET").unwrap_or_default();
    if target.contains("aarch64") {
        if let Ok(dir) = env::var("AARCH64_CUDA_LIB_DIR") {
            println!("cargo:rerun-if-env-changed=AARCH64_CUDA_LIB_DIR");
            return dir;
        }
    }
    if target == "x86_64-unknown-linux-gnu" {
        if let Ok(dir) = env::var("X86_64_CUDA_LIB_DIR") {
            println!("cargo:rerun-if-env-changed=X86_64_CUDA_LIB_DIR");
            return dir;
        }
    }
    if let Ok(dir) = env::var("CUDA_LIB_DIR") {
        println!("cargo:rerun-if-env-changed=CUDA_LIB_DIR");
        return dir;
    }
    if let Ok(home) = env::var("CUDA_HOME") {
        println!("cargo:rerun-if-env-changed=CUDA_HOME");
        return format!("{}/lib64", home.trim_end_matches('/'));
    }
    if let Ok(path) = env::var("CUDA_PATH") {
        println!("cargo:rerun-if-env-changed=CUDA_PATH");
        return format!("{}/lib64", path.trim_end_matches('/'));
    }
    "/usr/local/cuda/lib64".into()
}

fn main() {
    let cuda_lib = cuda_lib_dir();
    println!("cargo:rustc-link-search=native={}", cuda_lib);
    // Fallback for distro-installed CUDA by host arch
    let target = env::var("TARGET").unwrap_or_default();
    if target.contains("aarch64") {
        println!("cargo:rustc-link-search=native=/usr/lib/aarch64-linux-gnu");
    } else {
        println!("cargo:rustc-link-search=native=/usr/lib/x86_64-linux-gnu");
    }
    println!("cargo:rerun-if-changed=core");
    println!("cargo:rerun-if-env-changed=KERNEL_PTX_PATH");
    println!("cargo:rerun-if-env-changed=KERNEL_TARGET_CPU");

    // Compute capability the kernel PTX targets. Defaults to sm_75: the driver JITs
    // PTX to the installed GPU at load time and we use no arch-specific instructions,
    // so a low target runs everywhere from Turing up (incl. Hopper) at full speed.
    // The PTX `.target` is a *minimum*, so don't raise it above your oldest GPU.
    let target_cpu = env::var("KERNEL_TARGET_CPU").unwrap_or_else(|_| "sm_75".into());
    let target_cpu_flag = format!("-Ctarget-cpu={}", target_cpu);

    let manifest_dir = PathBuf::from(env::var("CARGO_MANIFEST_DIR").unwrap());
    let target_dir = env::var("CARGO_TARGET_DIR")
        .map(PathBuf::from)
        .unwrap_or_else(|_| manifest_dir.join("target"));
    // Use a separate target dir for the kernel build to avoid deadlock: the outer
    // cargo holds the main target lock; inner cargo would wait forever for it.
    let kernel_target_dir = target_dir.join("nvptx-kernel");
    let nvptx_dir = kernel_target_dir
        .join("nvptx64-nvidia-cuda")
        .join("release");

    eprintln!("Building PTX kernel for nvptx64-nvidia-cuda (first time can take several minutes)...");

    let status = Command::new(env::var("CARGO").unwrap_or_else(|_| "cargo".into()))
        .args([
            "rustc",
            "-p",
            "tor-v3-vanity-core",
            "--target",
            "nvptx64-nvidia-cuda",
            "-Z",
            "build-std=core,panic_abort",
            "--release",
            "--",
            "-Z",
            "unstable-options",
            "-C",
            "link-self-contained=+linker",
            "-C",
            "linker-flavor=llbc",
            target_cpu_flag.as_str(),
        ])
        .env("CARGO_TARGET_DIR", &kernel_target_dir)
        .current_dir(&manifest_dir)
        .status();

    let status = match status {
        Ok(s) => s,
        Err(e) => {
            eprintln!(
                "\n\nFailed to run kernel build: {}.\n\nRun this once: {}\n\nThen: cargo +nightly build --release\n",
                e, SETUP_CMD
            );
            panic!("kernel build failed: {}", e);
        }
    };

    if !status.success() {
        eprintln!(
            "\n\nKernel build failed. Run this once (reproducible on any machine):\n\n  {}\n\nThen: cargo +nightly build --release\n",
            SETUP_CMD
        );
        panic!("kernel build failed");
    }

    // Find the generated .ptx (llbc produces PTX for cdylib)
    let deps = nvptx_dir.join("deps");
    let pattern = deps.join("*.ptx");
    let pattern = pattern.to_str().expect("path is valid UTF-8");
    let ptx_path = glob::glob(pattern)
        .ok()
        .and_then(|mut g| g.next())
        .and_then(|e| e.ok());

    let ptx_path = match ptx_path {
        Some(p) => p,
        None => {
            eprintln!("cargo:warning=No .ptx found under {}", deps.display());
            panic!("kernel build did not produce a .ptx file under {}", deps.display());
        }
    };

    let ptx_path = ptx_path.canonicalize().unwrap_or(ptx_path);
    println!("cargo:rustc-env=KERNEL_PTX_PATH={}", ptx_path.display());
}
