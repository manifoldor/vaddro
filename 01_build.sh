#!/usr/bin/env bash
# ==============================================================================
# Vaddro Native CUDA + PTX Search Engine Builder
# ==============================================================================
set -euo pipefail

# Ensure standard CUDA binary paths are included in PATH
export PATH="/usr/local/cuda-12.8/bin:/usr/local/cuda/bin:$PATH"

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
CUDA_SOURCE_DIR="$SCRIPT_DIR/cuda"

echo "=================================================="
echo "Vaddro — Vanity Address Online"
echo "Seed Offline. Compute Online."
echo "🐧 Building Native CUDA Search Engine"
echo "=================================================="

# 1. Package & Tool Check
if ! command -v nvcc &> /dev/null; then
    echo "ERROR: nvcc compiler not found in PATH." >&2
    echo "Please ensure CUDA Toolkit 12.8+ is installed and added to PATH." >&2
    exit 1
fi

if ! command -v make &> /dev/null; then
    echo "--> Installing make..."
    if command -v apt-get &> /dev/null; then
        apt-get update && apt-get install -y make build-essential
    fi
fi

NPROC=$(nproc 2>/dev/null || echo 4)

# 2. Detect host GPU architecture for ultra-fast single-target compile
ARCH_FLAG=""
TARGET_ARCH="sm_120"
MIN_CUDA_REQUIRED="12.8"

if command -v nvidia-smi &> /dev/null; then
    GPU_NAME=$(nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null | head -n 1 || true)
    if [[ "$GPU_NAME" =~ "5090"|"5080"|"5070"|"Blackwell" ]]; then
        ARCH_FLAG="ARCH=sm_120"
        TARGET_ARCH="sm_120"
        MIN_CUDA_REQUIRED="12.8"
        echo "⚡ Detected NVIDIA Blackwell GPU ($GPU_NAME) -> Targeting sm_120 natively."
    elif [[ "$GPU_NAME" =~ "4090"|"4080"|"4070"|"Ada"|"L4"|"L40" ]]; then
        ARCH_FLAG="ARCH=sm_89"
        TARGET_ARCH="sm_89"
        MIN_CUDA_REQUIRED="11.8"
        echo "⚡ Detected NVIDIA Ada Lovelace GPU ($GPU_NAME) -> Targeting sm_89 natively."
    elif [[ "$GPU_NAME" =~ "3090"|"3080"|"3070"|"A10"|"A5000"|"A6000" ]]; then
        ARCH_FLAG="ARCH=sm_86"
        TARGET_ARCH="sm_86"
        MIN_CUDA_REQUIRED="11.1"
        echo "⚡ Detected NVIDIA Ampere GPU ($GPU_NAME) -> Targeting sm_86 natively."
    elif [[ "$GPU_NAME" =~ "A100"|"A30" ]]; then
        ARCH_FLAG="ARCH=sm_80"
        TARGET_ARCH="sm_80"
        MIN_CUDA_REQUIRED="11.0"
        echo "⚡ Detected NVIDIA A100 GPU ($GPU_NAME) -> Targeting sm_80 natively."
    elif [[ "$GPU_NAME" =~ "H100"|"H800" ]]; then
        ARCH_FLAG="ARCH=sm_90"
        TARGET_ARCH="sm_90"
        MIN_CUDA_REQUIRED="11.8"
        echo "⚡ Detected NVIDIA Hopper GPU ($GPU_NAME) -> Targeting sm_90 natively."
    fi
fi

# 3. Check NVCC Version against target architecture requirement
NVCC_VERSION=$(nvcc --version | sed -nE 's/.*release ([0-9]+\.[0-9]+).*/\1/p' | head -n 1)
NVCC_MAJOR="${NVCC_VERSION%%.*}"
NVCC_MINOR="${NVCC_VERSION#*.}"
NVCC_MINOR="${NVCC_MINOR%%.*}"
if [[ -z "$NVCC_VERSION" || -z "$NVCC_MAJOR" || -z "$NVCC_MINOR" ]]; then
    echo "ERROR: could not determine nvcc version." >&2
    exit 1
fi

if [[ "$TARGET_ARCH" == "sm_120" ]]; then
    if (( NVCC_MAJOR < 12 || (NVCC_MAJOR == 12 && NVCC_MINOR < 8) )); then
        echo "ERROR: CUDA $NVCC_VERSION detected, but CUDA 12.8+ is required for Blackwell sm_120." >&2
        exit 1
    fi
elif [[ "$TARGET_ARCH" == "sm_89" || "$TARGET_ARCH" == "sm_90" ]]; then
    if (( NVCC_MAJOR < 11 || (NVCC_MAJOR == 11 && NVCC_MINOR < 8) )); then
        echo "ERROR: CUDA $NVCC_VERSION detected, but CUDA 11.8+ is required for $TARGET_ARCH." >&2
        exit 1
    fi
fi

# 4. Compile Native CUDA Engine
echo ""
echo "--> Verifying embedded secp256k1 precomputation table..."
PYTHON_BIN="$(command -v python3 || command -v python || true)"
if [[ -z "$PYTHON_BIN" && -x "/root/miniconda3/bin/python3" ]]; then
    PYTHON_BIN="/root/miniconda3/bin/python3"
elif [[ -z "$PYTHON_BIN" && -x "/root/miniconda3/bin/python" ]]; then
    PYTHON_BIN="/root/miniconda3/bin/python"
fi

if [[ -f "$SCRIPT_DIR/scripts/verify_precomp.py" ]]; then
    if [[ -n "$PYTHON_BIN" ]]; then
        "$PYTHON_BIN" "$SCRIPT_DIR/scripts/verify_precomp.py"
    else
        echo "WARN: No python interpreter found to verify precomp. Skipping precomp verification."
    fi
else
    echo "ERROR: missing scripts/verify_precomp.py; refusing to build unchecked source." >&2
    exit 1
fi

echo "--> Compiling Vaddro Native CUDA + PTX Engine..."
cd "$CUDA_SOURCE_DIR"
make clean || true
if make -j"$NPROC" $ARCH_FLAG; then
    CUDA_EXE="$CUDA_SOURCE_DIR/bin/vaddro-cuda"
    if [[ -f "$CUDA_EXE" ]]; then
        chmod +x "$CUDA_EXE"
        echo "✓ Vaddro Native CUDA Engine build successful: $CUDA_EXE"
        echo "--> Running GPU cryptographic self-test..."
        "$CUDA_EXE" --self-test
    fi
else
    echo "ERROR: Native CUDA compilation failed." >&2
    exit 1
fi
cd "$SCRIPT_DIR"

echo ""
echo "=================================================="
echo "✓ Build completed successfully (Vaddro CUDA Ready)"
echo "=================================================="

# Run doctor check
if [[ -x "$SCRIPT_DIR/00_doctor.sh" ]]; then
    echo ""
    "$SCRIPT_DIR/00_doctor.sh" || true
fi

echo ""
echo "Next steps:"
echo "  - Test GPU speed:   ./benchmark.sh"
echo "  - Start searching:  ./02_start_search.sh"
echo "=================================================="
