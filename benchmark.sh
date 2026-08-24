#!/usr/bin/env bash
# ==============================================================================
# Vaddro Linux GPU Benchmark Tool (Native CUDA + PTX)
# ==============================================================================
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
CUDA_EXE="$SCRIPT_DIR/cuda/bin/vaddro-cuda"
CONFIG_DIR="$SCRIPT_DIR/config"
PUBKEY_PATH="$CONFIG_DIR/public-key.txt"
ENV_FILE="$CONFIG_DIR/miner.env"
CONFIG_LOADER="$SCRIPT_DIR/scripts/config_loader.sh"

echo "=================================================="
echo "Vaddro — Vanity Address Online"
echo "Seed Offline. Compute Online."
echo "⚡ Linux CUDA GPU Benchmark"
echo "=================================================="

# Load miner.env
if [[ ! -f "$ENV_FILE" ]]; then
    echo "ERROR: Missing configuration file: $ENV_FILE" >&2
    echo "Create it first: cp config/miner.env.example config/miner.env" >&2
    exit 1
fi

if [[ ! -f "$CONFIG_LOADER" ]]; then
    echo "ERROR: Missing safe configuration loader: $CONFIG_LOADER" >&2
    exit 1
fi
# shellcheck disable=SC1090
source "$CONFIG_LOADER"
load_vaddro_config "$ENV_FILE"

PUBLIC_KEY=$(echo -n "${PUBLIC_KEY:-}" | tr -d '[:space:]')
if [[ ! "$PUBLIC_KEY" =~ ^[0-9a-fA-F]{128}$ ]]; then
    echo "ERROR: Invalid PUBLIC_KEY in config/miner.env (must be 128 hex chars)." >&2
    exit 1
fi

DEMO_PUBLIC_KEY="426af53a55b0b1c06ea5a1e41ad1ef3ab87d83cafabbd97325bc67e9255f887987aa1c322948f193dde80defbbb4e6c1997519a1111d3326b22ce329c4b67adf"
if [[ "$PUBLIC_KEY" == "$DEMO_PUBLIC_KEY" && "${ALLOW_DEMO_KEY:-0}" != "1" ]]; then
    echo "ERROR: config/miner.env still contains the README demo public key." >&2
    echo "       Replace PUBLIC_KEY with your real split public key, or set ALLOW_DEMO_KEY=1." >&2
    exit 1
fi

WORK_SIZE="${WORK_SIZE:-224}"
INVERSE_MULTIPLE="${INVERSE_MULTIPLE:-32768}"
INVERSE_SIZE="${INVERSE_SIZE:-255}"
STREAMS="${STREAMS:-8}"

if [[ ! "$WORK_SIZE" =~ ^[0-9]+$ ]]; then
    echo "ERROR: WORK_SIZE must be an integer." >&2
    exit 1
fi
if [[ ! "$INVERSE_SIZE" =~ ^[0-9]+$ ]]; then
    echo "ERROR: INVERSE_SIZE must be an integer." >&2
    exit 1
fi
if [[ ! "$INVERSE_MULTIPLE" =~ ^[0-9]+$ ]]; then
    echo "ERROR: INVERSE_MULTIPLE must be an integer." >&2
    exit 1
fi
if [[ ! "$STREAMS" =~ ^[0-9]+$ ]]; then
    echo "ERROR: STREAMS must be an integer." >&2
    exit 1
fi

if [[ "$WORK_SIZE" -lt 32 || "$WORK_SIZE" -gt 256 || $((WORK_SIZE % 32)) -ne 0 ]]; then
    echo "ERROR: WORK_SIZE must be a multiple of 32 between 32 and 256." >&2
    exit 1
fi
if [[ "$INVERSE_SIZE" -ne 255 ]]; then
    echo "ERROR: this build only supports INVERSE_SIZE=255." >&2
    exit 1
fi
if [[ "$INVERSE_MULTIPLE" -lt 2 || $((INVERSE_MULTIPLE % 2)) -ne 0 ]]; then
    echo "ERROR: INVERSE_MULTIPLE must be a positive even number." >&2
    exit 1
fi
if [[ "$STREAMS" -lt 1 || "$STREAMS" -gt 8 || $((INVERSE_MULTIPLE % STREAMS)) -ne 0 ]]; then
    echo "ERROR: STREAMS must be 1..8 and divide INVERSE_MULTIPLE." >&2
    exit 1
fi

if [[ ! -f "$CUDA_EXE" || ! -x "$CUDA_EXE" ]]; then
    echo "ERROR: Native CUDA engine not found ($CUDA_EXE). Please run ./01_build.sh first." >&2
    exit 1
fi

echo "--> Benchmarking Native CUDA + PTX Engine..."
set +e
"$CUDA_EXE" --benchmark -z "$PUBLIC_KEY" -w "$WORK_SIZE" -I "$INVERSE_MULTIPLE" -i "$INVERSE_SIZE" -s "$STREAMS"
set -e

echo ""
echo "=================================================="
echo "✓ Benchmark complete."
echo "=================================================="
