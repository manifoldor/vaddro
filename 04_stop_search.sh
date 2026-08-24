#!/usr/bin/env bash
# Stop only Vaddro engine PIDs recorded by this installation.
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
RUNTIME_DIR="$SCRIPT_DIR/run"
CUDA_EXE="$SCRIPT_DIR/cuda/bin/vaddro-cuda"
GPU_FILTER=""

while [[ $# -gt 0 ]]; do
    case "$1" in
        -g|--gpu) GPU_FILTER="${2:?missing value for $1}"; shift 2 ;;
        *) echo "Usage: $0 [--gpu N]" >&2; exit 1 ;;
    esac
done
if [[ -n "$GPU_FILTER" && ! "$GPU_FILTER" =~ ^[0-9]+$ ]]; then
    echo "ERROR: GPU index must be a non-negative integer." >&2
    exit 1
fi

echo "Vaddro — Vanity Address Online"
echo "Seed Offline. Compute Online."

is_expected_engine() {
    local pid="$1" actual
    [[ "$pid" =~ ^[0-9]+$ ]] || return 1
    [[ -d "/proc/$pid" ]] || return 1
    actual=$(readlink "/proc/$pid/exe" 2>/dev/null || true)
    actual="${actual% (deleted)}"
    [[ "$actual" == "$CUDA_EXE" ]]
}

stop_pid_file() {
    local pid_file="$1" pid gpu waited
    gpu="${pid_file##*/gpu-}"
    gpu="${gpu%.pid}"
    if [[ -L "$pid_file" || ! -f "$pid_file" ]]; then
        return
    fi
    pid=$(tr -dc '0-9' < "$pid_file" 2>/dev/null || true)
    if ! is_expected_engine "$pid"; then
        echo "GPU $gpu: stale or mismatched PID file; no process was signalled."
        rm -f "$pid_file"
        return
    fi

    echo "GPU $gpu: sending SIGTERM to Vaddro engine PID $pid..."
    kill -TERM "$pid"
    waited=0
    while is_expected_engine "$pid" && [[ "$waited" -lt 30 ]]; do
        sleep 1
        waited=$((waited + 1))
    done
    if is_expected_engine "$pid"; then
        echo "GPU $gpu: graceful timeout; sending SIGKILL to PID $pid." >&2
        kill -KILL "$pid"
    else
        echo "GPU $gpu: stopped cleanly."
    fi
}

if [[ ! -d "$RUNTIME_DIR" ]]; then
    echo "No Vaddro runtime directory exists; nothing to stop."
    exit 0
fi

PID_FILES=()
if [[ -n "$GPU_FILTER" ]]; then
    PID_FILES+=("$RUNTIME_DIR/gpu-${GPU_FILTER}.pid")
else
    shopt -s nullglob
    PID_FILES=("$RUNTIME_DIR"/gpu-*.pid)
    shopt -u nullglob
fi

if [[ ${#PID_FILES[@]} -eq 0 ]]; then
    echo "No active Vaddro PID files found."
    exit 0
fi
for pid_file in "${PID_FILES[@]}"; do
    stop_pid_file "$pid_file"
done
