#!/usr/bin/env bash
# setup-intel-gpu.sh — Install Intel GPU compute stack on Debian 13 (Trixie)
# Targets: Intel Arc B-series (Battlemage) discrete GPUs
# Tested on: Intel Arc Pro B50 (Battlemage G21), kernel 6.12, Debian 13.4
#
# Prerequisites: xe kernel driver loaded (check: lspci -k | grep -A2 "VGA.*Intel")
# Usage: sudo ./setup-intel-gpu.sh

set -euo pipefail

WORKDIR=$(mktemp -d /tmp/intel-gpu-debs.XXXX)
trap 'rm -rf "$WORKDIR"' EXIT

# Versions — update these when upgrading
COMPUTE_RUNTIME_VER="26.14.37833.4"
IGC_VER="2.32.7"
IGC_BUILD="21184"
LEVEL_ZERO_VER="1.28.0"
IGDGMM_VER="22.9.0"
XPU_SMI_VER="1.3.6"
XPU_SMI_BUILD="20260206.143628.1004f6cb"

echo "=== Intel GPU Compute Stack Installer ==="
echo "Compute Runtime: $COMPUTE_RUNTIME_VER"
echo "IGC: $IGC_VER"
echo "Level Zero: $LEVEL_ZERO_VER"
echo "Working directory: $WORKDIR"
echo

# Check for xe driver
if ! lspci -k 2>/dev/null | grep -q "Kernel driver in use: xe"; then
    echo "ERROR: No Intel GPU with xe driver found. Is the card installed?"
    exit 1
fi

# ── 1. Add Intel GPU apt repository (for intel-gsc and future updates) ──
echo ">>> Adding Intel GPU apt repository..."
wget -qO - https://repositories.intel.com/gpu/intel-graphics.key | \
    gpg --yes --dearmor --output /usr/share/keyrings/intel-graphics.gpg

echo "deb [arch=amd64 signed-by=/usr/share/keyrings/intel-graphics.gpg] https://repositories.intel.com/gpu/ubuntu noble unified" \
    > /etc/apt/sources.list.d/intel-gpu.list

apt-get update -qq

# ── 2. Install intel-gsc (Graphics System Controller) from Intel repo ──
echo ">>> Installing intel-gsc and dependencies..."
apt-get install -y intel-gsc

# ── 3. Download compute runtime packages from GitHub releases ──
echo ">>> Downloading Intel compute runtime packages..."
cd "$WORKDIR"

CR_BASE="https://github.com/intel/compute-runtime/releases/download/${COMPUTE_RUNTIME_VER}"
wget -q "${CR_BASE}/libigdgmm12_${IGDGMM_VER}_amd64.deb"
wget -q "${CR_BASE}/intel-opencl-icd_${COMPUTE_RUNTIME_VER}-0_amd64.deb"
wget -q "${CR_BASE}/libze-intel-gpu1_${COMPUTE_RUNTIME_VER}-0_amd64.deb"
wget -q "${CR_BASE}/intel-ocloc_${COMPUTE_RUNTIME_VER}-0_amd64.deb"

echo ">>> Downloading Intel Graphics Compiler..."
IGC_BASE="https://github.com/intel/intel-graphics-compiler/releases/download/v${IGC_VER}"
wget -q "${IGC_BASE}/intel-igc-core-2_${IGC_VER}%2B${IGC_BUILD}_amd64.deb"
wget -q "${IGC_BASE}/intel-igc-opencl-2_${IGC_VER}%2B${IGC_BUILD}_amd64.deb"

echo ">>> Downloading Level Zero..."
LZ_BASE="https://github.com/oneapi-src/level-zero/releases/download/v${LEVEL_ZERO_VER}"
wget -q "${LZ_BASE}/level-zero_${LEVEL_ZERO_VER}%2Bu24.04_amd64.deb"
wget -q "${LZ_BASE}/level-zero-devel_${LEVEL_ZERO_VER}%2Bu24.04_amd64.deb"

echo ">>> Downloading xpu-smi..."
wget -q "https://github.com/intel/xpumanager/releases/download/v${XPU_SMI_VER}/xpu-smi_${XPU_SMI_VER}_${XPU_SMI_BUILD}.u24.04_amd64.deb"

# ── 4. Handle libze1 conflict ──
# Debian's libze1 package conflicts with Intel's level-zero package
# (both provide libze_loader.so.1). Remove libze1 and create a shim.
if dpkg -l libze1 2>/dev/null | grep -q "^ii.*1\.20"; then
    echo ">>> Removing conflicting Debian libze1..."
    dpkg -r --force-depends libze1 libze-dev 2>/dev/null || true
fi

# ── 5. Install packages in dependency order ──
echo ">>> Installing packages..."
dpkg -i "libigdgmm12_${IGDGMM_VER}_amd64.deb"
dpkg -i "intel-igc-core-2_${IGC_VER}+${IGC_BUILD}_amd64.deb" \
        "intel-igc-opencl-2_${IGC_VER}+${IGC_BUILD}_amd64.deb"
dpkg -i "level-zero_${LEVEL_ZERO_VER}+u24.04_amd64.deb" \
        "level-zero-devel_${LEVEL_ZERO_VER}+u24.04_amd64.deb"
dpkg -i "libze-intel-gpu1_${COMPUTE_RUNTIME_VER}-0_amd64.deb" \
        "intel-opencl-icd_${COMPUTE_RUNTIME_VER}-0_amd64.deb"
dpkg -i "intel-ocloc_${COMPUTE_RUNTIME_VER}-0_amd64.deb"

# ── 6. Create libze1 shim to satisfy libhwloc-plugins dependency ──
if ! dpkg -l libze1 2>/dev/null | grep -q "^ii"; then
    echo ">>> Creating libze1 shim package..."
    SHIM_DIR=$(mktemp -d)
    mkdir -p "${SHIM_DIR}/DEBIAN"
    cat > "${SHIM_DIR}/DEBIAN/control" <<CTRL
Package: libze1
Version: ${LEVEL_ZERO_VER}-shim
Architecture: amd64
Maintainer: local <local@local>
Depends: level-zero
Description: Shim - level-zero replaces libze1
Section: libs
Priority: optional
CTRL
    dpkg-deb --root-owner-group --build "$SHIM_DIR" "${WORKDIR}/libze1_shim.deb"
    dpkg -i "${WORKDIR}/libze1_shim.deb"
    rm -rf "$SHIM_DIR"
fi

# ── 7. Install xpu-smi ──
echo ">>> Installing xpu-smi..."
dpkg -i "xpu-smi_${XPU_SMI_VER}_${XPU_SMI_BUILD}.u24.04_amd64.deb"

# ── 8. Install verification tools ──
echo ">>> Installing clinfo..."
apt-get install -y clinfo

# ── 9. Fix any remaining dependency issues ──
apt --fix-broken install -y

# ── 10. Verify ──
echo
echo "=== Verification ==="
echo
echo "--- xpu-smi discovery ---"
xpu-smi discovery
echo
echo "--- OpenCL devices ---"
clinfo --list
echo
echo "--- Health check ---"
xpu-smi health -d 0
echo
echo "=== Intel GPU compute stack installed successfully ==="
