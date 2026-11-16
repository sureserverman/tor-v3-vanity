#!/usr/bin/env bash
# Grep cuda-oxide-emitted PTX for .local memory use (ld.local / st.local /
# .local declarations). Related to NVIDIA/cuda-rust#1815: constant-trip loops
# that stay rolled at -O2 can silently spill register arrays to .local.
#
# Usage (from rust/oxide_spike/):
#   ./scripts/check-ptx-local.sh [path-to.ptx]
# Env:
#   PTX_LOCAL_STRICT=1  — exit 1 if any .local traffic is found (default: warn)
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"

PTX="${1:-}"
if [[ -z "$PTX" ]]; then
  # Prefer the artifact cargo oxide leaves next to the crate after build.
  if [[ -f oxide_spike.ptx ]]; then
    PTX=oxide_spike.ptx
  else
    # Fall back to newest .ptx under .oxide-artifacts / target
    PTX="$(find .oxide-artifacts target -name '*.ptx' -type f 2>/dev/null | head -1 || true)"
  fi
fi

if [[ -z "${PTX}" || ! -f "${PTX}" ]]; then
  echo "PTX .local check: no .ptx found. Run: cargo oxide build --arch sm_75" >&2
  exit 2
fi

# Count declarations and traffic. Use separate patterns so a .local in a comment
# or symbol name is less likely to inflate the decl count.
decl_count=$(rg -c '^\s*\.local\b' "$PTX" || true)
# rg -c prints nothing when zero matches with || true from empty — normalize
decl_count=${decl_count:-0}
ld_count=$(rg -c '\bld\.local\b' "$PTX" || true)
ld_count=${ld_count:-0}
st_count=$(rg -c '\bst\.local\b' "$PTX" || true)
st_count=${st_count:-0}

# Some PTX dumps put multiple ops per line; also count occurrences not just lines.
ld_occ=$(rg -o '\bld\.local\b' "$PTX" | wc -l | tr -d ' ')
st_occ=$(rg -o '\bst\.local\b' "$PTX" | wc -l | tr -d ' ')
decl_occ=$(rg -o '^\s*\.local\b' "$PTX" | wc -l | tr -d ' ')

echo "PTX .local check ($PTX)"
echo "  .local declarations (lines/occ): ${decl_count}/${decl_occ}"
echo "  ld.local (lines/occ):            ${ld_count}/${ld_occ}"
echo "  st.local (lines/occ):            ${st_count}/${st_occ}"

total=$((decl_occ + ld_occ + st_occ))
if [[ "$total" -gt 0 ]]; then
  echo "WARNING: PTX uses .local memory (${total} declaration/traffic hits)." >&2
  echo "  See NVIDIA/cuda-rust#1815 — unexpected spills of register arrays are a red flag." >&2
  echo "  Windowed Fe arrays in render_incremental are expected to live in .local;" >&2
  echo "  re-check after kernel changes that touch hot loops." >&2
  if [[ "${PTX_LOCAL_STRICT:-0}" == "1" ]]; then
    echo "PTX_LOCAL_STRICT=1: failing on .local traffic." >&2
    exit 1
  fi
else
  echo "PTX .local check PASS: no .local declarations or ld/st.local traffic."
fi
