#!/usr/bin/env bash
# Human-readable status for the latest run on each GPU.
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

state_value() {
    local file="$1" key="$2"
    awk -F '\t' -v wanted="$key" '$1 == wanted {print substr($0, length($1) + 2); exit}' "$file"
}

is_expected_engine() {
    local pid="$1" actual
    [[ "$pid" =~ ^[0-9]+$ ]] || return 1
    [[ -d "/proc/$pid" ]] || return 1
    actual=$(readlink "/proc/$pid/exe" 2>/dev/null || true)
    actual="${actual% (deleted)}"
    [[ "$actual" == "$CUDA_EXE" ]]
}

show_state() {
    local state_file="$1" gpu pid worker run_id shard_index shard_count started result_dir json_log lifecycle
    local hashrate="n/a" elapsed="n/a" hits="n/a" canary="never" reason="n/a"
    gpu=$(state_value "$state_file" gpu)
    pid=$(state_value "$state_file" engine_pid)
    worker=$(state_value "$state_file" worker_id)
    run_id=$(state_value "$state_file" run_id)
    shard_index=$(state_value "$state_file" shard_index)
    shard_count=$(state_value "$state_file" shard_count)
    shard_index="${shard_index:-n/a}"
    shard_count="${shard_count:-n/a}"
    started=$(state_value "$state_file" started_utc)
    result_dir=$(state_value "$state_file" result_dir)
    json_log=$(state_value "$state_file" json_log)

    if is_expected_engine "$pid"; then lifecycle="RUNNING"; else lifecycle="STOPPED"; fi

    local python_bin=""
    python_bin=$(command -v python3 || command -v python || true)
    if [[ -z "$python_bin" && -x /root/miniconda3/bin/python3 ]]; then
        python_bin=/root/miniconda3/bin/python3
    elif [[ -z "$python_bin" && -x /root/miniconda3/bin/python ]]; then
        python_bin=/root/miniconda3/bin/python
    fi

    if [[ -f "$json_log" && -n "$python_bin" ]]; then
        STATUS_FIELDS=$("$python_bin" - "$json_log" <<'PY'
import json
import sys

status = {}
canary = {}
completed = {}
try:
    with open(sys.argv[1], "r", encoding="utf-8") as handle:
        for line in handle:
            try:
                event = json.loads(line)
            except (ValueError, TypeError):
                continue
            if event.get("event") == "status":
                status = event
            elif event.get("event") == "canary":
                canary = event
            elif event.get("event") == "completed":
                completed = event
except OSError:
    pass

values = (
    status.get("hashrate", "n/a"),
    status.get("elapsed", completed.get("elapsed", "n/a")),
    status.get("hit_progress", str(completed.get("hits", "n/a"))),
    canary.get("last_canary_utc", status.get("last_canary_utc", "never")),
    completed.get("reason", "n/a"),
)
print("\t".join(str(value).replace("\t", " ").replace("\n", " ") for value in values))
PY
)
        IFS=$'\t' read -r hashrate elapsed hits canary reason <<< "$STATUS_FIELDS"
    fi

    echo "------------------------------------------------------------"
    echo "GPU $gpu  $lifecycle"
    echo "Worker:       $worker"
    echo "Run ID:       $run_id"
    echo "Shard:        $shard_index/$shard_count"
    echo "Engine PID:   $pid"
    echo "Started UTC:  $started"
    echo "Hashrate:     $hashrate"
    echo "Elapsed:      $elapsed"
    echo "Hits:         $hits"
    echo "Last canary:  $canary"
    echo "Exit reason:  $reason"
    echo "Results:      $result_dir"
    echo "JSONL log:    $json_log"
    if command -v nvidia-smi >/dev/null 2>&1; then
        GPU_HEALTH=$(nvidia-smi -i "$gpu" --query-gpu=temperature.gpu,power.draw,memory.used,utilization.gpu --format=csv,noheader 2>/dev/null || true)
        [[ -n "$GPU_HEALTH" ]] && echo "GPU health:   $GPU_HEALTH"
    fi
}

if [[ ! -d "$RUNTIME_DIR" ]]; then
    echo "No Vaddro run state found."
    exit 0
fi

STATE_FILES=()
if [[ -n "$GPU_FILTER" ]]; then
    STATE_FILES+=("$RUNTIME_DIR/gpu-${GPU_FILTER}.state")
else
    shopt -s nullglob
    STATE_FILES=("$RUNTIME_DIR"/gpu-*.state)
    shopt -u nullglob
fi

if [[ ${#STATE_FILES[@]} -eq 0 ]]; then
    echo "No Vaddro run state found."
    exit 0
fi
DISPLAYED=0
for state_file in "${STATE_FILES[@]}"; do
    if [[ -f "$state_file" && ! -L "$state_file" ]]; then
        show_state "$state_file"
        DISPLAYED=1
    fi
done
if [[ "$DISPLAYED" -eq 0 ]]; then
    echo "No Vaddro run state found for the requested GPU."
fi
