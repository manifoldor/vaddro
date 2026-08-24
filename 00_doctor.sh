#!/usr/bin/env bash
# ==============================================================================
# Vaddro Linux Diagnostic Doctor (RTX 5090 / CUDA / Environment)
# ==============================================================================
set -u

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# Ensure standard CUDA binary paths are included in PATH.
export PATH="/usr/local/cuda-12.8/bin:/usr/local/cuda/bin:$PATH"

echo "=================================================="
echo "Vaddro — Vanity Address Online"
echo "Seed Offline. Compute Online."
echo "🩺 Linux GPU & Environment Doctor"
echo "=================================================="

# 1. System Information
echo "--> 1. System Information:"
if [[ -f /etc/os-release ]]; then
    # shellcheck disable=SC1091
    . /etc/os-release
    echo "  OS:           ${PRETTY_NAME:-Linux}"
else
    echo "  OS:           $(uname -s) $(uname -r)"
fi
echo "  Kernel:       $(uname -r) ($(uname -m))"

# 2. NVIDIA GPU & Driver Check
echo ""
echo "--> 2. NVIDIA Driver & GPU Check:"
if command -v nvidia-smi &> /dev/null; then
    NVIDIA_DRIVER_VERSION=$(nvidia-smi --query-gpu=driver_version --format=csv,noheader | head -n 1 | tr -d '[:space:]')
    echo "  NVIDIA Driver: $NVIDIA_DRIVER_VERSION"

    GPU_INDEX=0
    nvidia-smi --query-gpu=index,name,memory.total,power.limit,temperature.gpu --format=csv,noheader | while IFS=',' read -r idx name mem pwr temp; do
        idx=$(echo "$idx" | tr -d '[:space:]')
        name=$(echo "$name" | sed 's/^[ \t]*//')
        mem=$(echo "$mem" | tr -d '[:space:]')
        pwr=$(echo "$pwr" | tr -d '[:space:]')
        temp=$(echo "$temp" | tr -d '[:space:]')
        echo "  GPU $idx:       $name ($mem, $pwr, $temp°C)"

        # Check 5090 specific requirement (NVIDIA Driver >= 570)
        if [[ "$name" =~ 5090|5080|5070|Blackwell|GB20 ]]; then
            echo "  ⚡ Blackwell Architecture detected ($name)"
            DRIVER_MAJOR=$(echo "$NVIDIA_DRIVER_VERSION" | cut -d'.' -f1)
            if [[ "$DRIVER_MAJOR" -lt 570 ]]; then
                echo "  ⚠️  WARNING: $name requires NVIDIA Driver >= 570.xx (current: $NVIDIA_DRIVER_VERSION)."
                echo "     Please install a current production driver compatible with CUDA 12.8."
            else
                echo "  ✓ Driver version ($NVIDIA_DRIVER_VERSION) is compatible with $name."
            fi
        fi
    done
else
    echo "  ⚠️  nvidia-smi not found. If this machine has an NVIDIA GPU, please install the proprietary driver (>= 570.xx for RTX 5090)."
fi

# 3. CUDA Toolkit & NVCC Check
echo ""
echo "--> 3. Native CUDA Toolkit Check:"
if command -v nvcc &> /dev/null; then
    NVCC_VER=$(nvcc --version | sed -nE 's/.*release ([0-9]+\.[0-9]+).*/\1/p' | head -n 1)
    echo "  ✓ NVCC Compiler found: CUDA $NVCC_VER"
    if [[ -n "$NVCC_VER" ]]; then
        NVCC_MAJOR="${NVCC_VER%%.*}"
        NVCC_MINOR="${NVCC_VER#*.}"
        NVCC_MINOR="${NVCC_MINOR%%.*}"
        if (( NVCC_MAJOR < 12 || (NVCC_MAJOR == 12 && NVCC_MINOR < 8) )); then
            echo "  ℹ️   CUDA $NVCC_VER detected. (Note: RTX 5090 sm_120 requires CUDA 12.8+; RTX 4090 sm_89 requires CUDA 11.8+)"
        fi
    fi
else
    echo "  ⚠️  NVCC not found in PATH. Please install CUDA Toolkit (CUDA 12.8+ for RTX 5090, or CUDA 11.8+ for RTX 4090)."
fi

# 4. Compiled Engine Binary Check
echo ""
echo "--> 4. Available Search Engine:"
CUDA_EXE="$SCRIPT_DIR/cuda/bin/vaddro-cuda"

if [[ -f "$CUDA_EXE" ]]; then
    echo "  ✓ Native CUDA + PTX Engine: $CUDA_EXE (qualified RTX 5090 production search: 5.20-5.26 GH/s)"
else
    echo "  - Native CUDA Engine: Not compiled yet (run ./01_build.sh)"
fi

echo ""
echo "=================================================="
echo "💡 Usage Guide for NVIDIA GPU:"
echo "   - Build Engine:     ./01_build.sh"
echo "   - Benchmark:        ./benchmark.sh"
echo "   - Start Search:     ./02_start_search.sh"
echo "=================================================="
