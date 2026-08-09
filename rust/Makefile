BINARY := t3v
DEB_DIR ?= $(CURDIR)/../deb
# Host arch for cross-compile detection (uname -m)
HOST_ARCH := $(shell uname -m)

CUDA_LIB_DIR ?= /usr/local/cuda/lib64
CUDA_LIB_FALLBACK_X64 := /usr/lib/x86_64-linux-gnu
CUDA_LIB_FALLBACK_ARM := /usr/lib/aarch64-linux-gnu
AARCH64_CUDA_LIB_DIR ?= $(CURDIR)/cuda-arm64-libs
X86_64_CUDA_LIB_DIR ?= $(CURDIR)/cuda-x86-64-libs
AARCH64_GCC := $(firstword $(shell command -v aarch64-linux-gnu-gcc 2>/dev/null || command -v aarch64-linux-gnu-gcc-14 2>/dev/null || command -v aarch64-linux-gnu-gcc-13 2>/dev/null || echo ""))
X86_64_GCC := $(firstword $(shell command -v x86_64-linux-gnu-gcc 2>/dev/null || command -v x86_64-linux-gnu-gcc-14 2>/dev/null || command -v x86_64-linux-gnu-gcc-13 2>/dev/null || echo ""))

DOCSRC := $(CURDIR)/../docs
MANDIR := $(DEB_DIR)/package/usr/share/man/man1
PKGDOCDIR := $(DEB_DIR)/package/usr/share/doc/$(BINARY)

.PHONY: all setup amd64 arm64 debdocs install-cuda check-cuda setup-cross-arm64-deps fetch-aarch64-cuda-libs setup-cross-x86-64-deps fetch-x86-64-cuda-libs clean
all: amd64

# Stage the man page, copyright file and changelog into the .deb skeleton.
# deb/ is gitignored, so these are generated at build time from the tracked
# sources in docs/, the same way the per-arch binaries are. Without them
# lintian reports no-changelog and no-copyright-file as errors.
# gzip -9n: lintian wants these compressed, and -n keeps the output
# byte-identical across builds.
debdocs: $(DOCSRC)/t3v.1 $(DOCSRC)/copyright $(DOCSRC)/changelog
	mkdir -p $(MANDIR) $(PKGDOCDIR)
	gzip -9nc $(DOCSRC)/t3v.1 > $(MANDIR)/t3v.1.gz
	gzip -9nc $(DOCSRC)/changelog > $(PKGDOCDIR)/changelog.gz
	cp $(DOCSRC)/copyright $(PKGDOCDIR)/copyright

# Install CUDA toolkit via apt (Debian/Ubuntu only). Run once if check-cuda fails.
install-cuda:
	@if [ -f "/etc/debian_version" ] && command -v apt-get >/dev/null 2>&1; then \
		echo "Installing CUDA toolkit (nvidia-cuda-toolkit) via apt..."; \
		sudo apt-get update && sudo apt-get install -y nvidia-cuda-toolkit; \
		echo "Done. Run: make check-cuda"; \
	else \
		echo "install-cuda is only for Debian/Ubuntu (apt)."; \
		echo "On other distros: https://developer.nvidia.com/cuda-downloads"; \
		exit 1; \
	fi

setup:
	@echo "Ensuring nightly toolchain and PTX build deps..."
	rustup install nightly && \
	rustup target add nvptx64-nvidia-cuda --toolchain nightly && \
	rustup component add llvm-bitcode-linker llvm-tools rust-src --toolchain nightly

check-cuda:
	@FALLBACK="$(CUDA_LIB_FALLBACK_X64)"; [ "$(CUDA_ARCH)" = "arm64" ] && FALLBACK="$(CUDA_LIB_FALLBACK_ARM)"; \
	if [ -f "$(CUDA_LIB_DIR)/libcuda.so" ] || [ -f "$$FALLBACK/libcuda.so" ]; then \
		echo "CUDA libs found."; \
	else \
		echo ""; \
		echo "CUDA libraries not found. Build will fail at link."; \
		echo "  make install-cuda  (Debian/Ubuntu)  or  CUDA_LIB_DIR=/path make amd64"; \
		echo ""; \
		exit 1; \
	fi

# --- amd64: native on x86_64, cross-compile when host is aarch64 ---
setup-cross-x86-64-deps:
	@$(CURDIR)/scripts/setup-cross-x86-64.sh

fetch-x86-64-cuda-libs:
	@$(CURDIR)/scripts/fetch-x86-64-cuda-libs.sh

amd64: setup debdocs
	@if [ "$(HOST_ARCH)" = "aarch64" ]; then \
		$(MAKE) setup-cross-x86-64-deps; \
		if [ -z "$(X86_64_GCC)" ]; then \
			echo "x86_64 gcc not found. Run: ./scripts/setup-cross-x86-64.sh"; exit 1; \
		fi; \
		if [ ! -f "$(X86_64_CUDA_LIB_DIR)/libcuda.so" ] && [ ! -f "$(X86_64_CUDA_LIB_DIR)/libcuda.so.1" ]; then \
			echo ""; \
			echo "X86_64_CUDA_LIB_DIR ($(X86_64_CUDA_LIB_DIR)) must contain libcuda.so (x86_64)."; \
			echo "  make fetch-x86-64-cuda-libs   then   FETCH_DRIVER=1 make fetch-x86-64-cuda-libs"; \
			echo "  Or copy libcuda.so from an x86_64 machine. See docs/amd64-build.md"; \
			echo ""; exit 1; \
		fi; \
		CARGO_TARGET_DIR=target/amd64 \
		CARGO_TARGET_X86_64_UNKNOWN_LINUX_GNU_LINKER="$(X86_64_GCC)" \
		X86_64_CUDA_LIB_DIR="$(X86_64_CUDA_LIB_DIR)" \
		  cargo +nightly build --release --target x86_64-unknown-linux-gnu; \
	else \
		$(MAKE) check-cuda; \
		CARGO_TARGET_DIR=target/amd64 \
		  cargo +nightly build --release --target x86_64-unknown-linux-gnu; \
	fi
	mkdir -p $(DEB_DIR)/amd64
	mv target/amd64/x86_64-unknown-linux-gnu/release/$(BINARY) $(DEB_DIR)/amd64/$(BINARY)
	@echo "Binary: $(DEB_DIR)/amd64/$(BINARY)"

# --- arm64: native on aarch64, cross-compile when host is amd64 ---
setup-cross-arm64-deps:
	@$(CURDIR)/scripts/setup-cross-arm64.sh

fetch-aarch64-cuda-libs:
	@$(CURDIR)/scripts/fetch-aarch64-cuda-libs.sh

arm64: setup debdocs
	rustup target add aarch64-unknown-linux-gnu --toolchain nightly
	@if [ "$(HOST_ARCH)" = "aarch64" ]; then \
		$(MAKE) check-cuda CUDA_ARCH=arm64; \
		CARGO_TARGET_DIR=target/arm64 cargo +nightly build --release --target aarch64-unknown-linux-gnu; \
	else \
		$(MAKE) setup-cross-arm64-deps; \
		if [ -z "$(AARCH64_GCC)" ]; then \
			echo "aarch64 gcc not found. Run: ./scripts/setup-cross-arm64.sh"; exit 1; \
		fi; \
		if [ ! -f "$(AARCH64_CUDA_LIB_DIR)/libcuda.so" ] && [ ! -f "$(AARCH64_CUDA_LIB_DIR)/libcuda.so.1" ]; then \
			echo ""; \
			echo "AARCH64_CUDA_LIB_DIR ($(AARCH64_CUDA_LIB_DIR)) must contain libcuda.so (aarch64)."; \
			echo "  make fetch-aarch64-cuda-libs   then   FETCH_DRIVER=1 make fetch-aarch64-cuda-libs"; \
			echo "  Or copy libcuda.so from an arm64 machine. See docs/arm64-build.md"; \
			echo ""; exit 1; \
		fi; \
		CARGO_TARGET_DIR=target/arm64 \
		CARGO_TARGET_AARCH64_UNKNOWN_LINUX_GNU_LINKER="$(AARCH64_GCC)" \
		AARCH64_CUDA_LIB_DIR="$(AARCH64_CUDA_LIB_DIR)" \
		  cargo +nightly build --release --target aarch64-unknown-linux-gnu; \
	fi
	mkdir -p $(DEB_DIR)/arm64
	mv target/arm64/aarch64-unknown-linux-gnu/release/$(BINARY) $(DEB_DIR)/arm64/$(BINARY)
	@echo "Binary: $(DEB_DIR)/arm64/$(BINARY)"

clean:
	rm -rf target $(DEB_DIR)/amd64/$(BINARY) $(DEB_DIR)/arm64/$(BINARY) $(MANDIR)/t3v.1.gz $(PKGDOCDIR)
