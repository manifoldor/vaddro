#!/usr/bin/env bash
# Vaddro Linux search runner: one locked worker per GPU with flat verified results.
set -euo pipefail

export PATH="/usr/local/cuda-12.8/bin:/usr/local/cuda/bin:$PATH"

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
CUDA_EXE="$SCRIPT_DIR/cuda/bin/vaddro-cuda"
CONFIG_DIR="$SCRIPT_DIR/config"
RESULT_ROOT="$SCRIPT_DIR/result"
LOG_ROOT="$SCRIPT_DIR/logs"
RUNTIME_DIR="$SCRIPT_DIR/run"
ACTIVITY_LOG="$LOG_ROOT/activity.log"
ENV_FILE="$CONFIG_DIR/miner.env"
CONFIG_LOADER="$SCRIPT_DIR/scripts/config_loader.sh"

if [[ ! -f "$CUDA_EXE" || ! -x "$CUDA_EXE" ]]; then
    echo "ERROR: Native CUDA search engine not found ($CUDA_EXE). Run ./01_build.sh first." >&2
    exit 1
fi
if [[ ! -f "$ENV_FILE" ]]; then
    echo "ERROR: Missing configuration file: $ENV_FILE" >&2
    echo "Create it first: cp config/miner.env.example config/miner.env" >&2
    exit 1
fi
if [[ ! -f "$CONFIG_LOADER" ]]; then
    echo "ERROR: Missing safe configuration loader: $CONFIG_LOADER" >&2
    exit 1
fi
if ! command -v flock >/dev/null 2>&1; then
    echo "ERROR: flock is required for per-GPU single-instance locking (package: util-linux)." >&2
    exit 1
fi

# shellcheck disable=SC1090
source "$CONFIG_LOADER"
load_vaddro_config "$ENV_FILE"

PUBLIC_KEY=$(printf '%s' "${PUBLIC_KEY:-}" | tr -d '[:space:]')
PATTERN_RAW=$(printf '%s' "${PATTERN:-}" | tr -d '[:space:]')
SAME_ENDS_CFG="${SAME_CHAR_ENDS:-0}"
WORK_SIZE="${WORK_SIZE:-224}"
INVERSE_MULTIPLE="${INVERSE_MULTIPLE:-32768}"
INVERSE_SIZE="${INVERSE_SIZE:-255}"
STREAMS="${STREAMS:-8}"
TARGET_GPU="${GPU_INDEX:-0}"
USER_EXTRA_FLAGS="${EXTRA_FLAGS:-}"
CANARY_INTERVAL="${CANARY_INTERVAL:-10000}"
MAX_HITS="${MAX_HITS:-1}"
WORKER_ID_CFG="${WORKER_ID:-}"
SHARD_INDEX="${SHARD_INDEX:-0}"
SHARD_COUNT="${SHARD_COUNT:-1}"
CLI_EXTRA_ARGS=()

while [[ $# -gt 0 ]]; do
    case "$1" in
        -w|--work) WORK_SIZE="${2:?missing value for $1}"; shift 2 ;;
        -I|--inverse-multiple) INVERSE_MULTIPLE="${2:?missing value for $1}"; shift 2 ;;
        -i|--inverse-size) INVERSE_SIZE="${2:?missing value for $1}"; shift 2 ;;
        -s|--streams) STREAMS="${2:?missing value for $1}"; shift 2 ;;
        -E|--same-char-ends) SAME_ENDS_CFG="${2:?missing value for $1}"; PATTERN_RAW=""; shift 2 ;;
        -m|--matching) PATTERN_RAW="${2:?missing value for $1}"; SAME_ENDS_CFG=0; shift 2 ;;
        -g|--gpu) TARGET_GPU="${2:?missing value for $1}"; shift 2 ;;
        --max-hits) MAX_HITS="${2:?missing value for $1}"; shift 2 ;;
        --worker-id) WORKER_ID_CFG="${2:?missing value for $1}"; shift 2 ;;
        --shard-index) SHARD_INDEX="${2:?missing value for $1}"; shift 2 ;;
        --shard-count) SHARD_COUNT="${2:?missing value for $1}"; shift 2 ;;
        *) CLI_EXTRA_ARGS+=("$1"); shift ;;
    esac
done

if [[ ! "$PUBLIC_KEY" =~ ^[0-9a-fA-F]{128}$ ]]; then
    echo "ERROR: PUBLIC_KEY must contain exactly 128 hexadecimal characters." >&2
    exit 1
fi
DEMO_PUBLIC_KEY="426af53a55b0b1c06ea5a1e41ad1ef3ab87d83cafabbd97325bc67e9255f887987aa1c322948f193dde80defbbb4e6c1997519a1111d3326b22ce329c4b67adf"
if [[ "$PUBLIC_KEY" == "$DEMO_PUBLIC_KEY" && "${ALLOW_DEMO_KEY:-0}" != "1" ]]; then
    echo "ERROR: config/miner.env contains the README demo public key." >&2
    exit 1
fi
if [[ ! "$TARGET_GPU" =~ ^[0-9]+$ ]]; then
    echo "ERROR: GPU_INDEX must be a non-negative integer." >&2
    exit 1
fi

SAME_ENDS="$SAME_ENDS_CFG"
if [[ ! "$SAME_ENDS" =~ ^[0-9]+$ ]] || [[ "$SAME_ENDS" -gt 8 ]]; then
    echo "ERROR: SAME_CHAR_ENDS must be an integer from 0 through 8." >&2
    exit 1
fi
if [[ "$SAME_ENDS" -gt 0 && -n "$PATTERN_RAW" ]]; then
    echo "ERROR: PATTERN and SAME_CHAR_ENDS are mutually exclusive." >&2
    exit 1
fi
if [[ "$SAME_ENDS" -eq 0 && -z "$PATTERN_RAW" ]]; then
    echo "ERROR: Configure PATTERN or SAME_CHAR_ENDS before starting." >&2
    exit 1
fi
if [[ "$SAME_ENDS" -eq 0 ]]; then
    if [[ ! "$PATTERN_RAW" =~ ^[0-9a-fA-FXx.]{40}$ ]]; then
        echo "ERROR: PATTERN must be a 40-character hexadecimal/wildcard template." >&2
        exit 1
    fi
    SCORE=$(printf '%s' "$PATTERN_RAW" | grep -o -i '[0-9a-f]' | wc -l | tr -d ' ')
    if [[ "$SCORE" -lt 1 || "$SCORE" -gt 40 ]]; then
        echo "ERROR: PATTERN must contain at least one fixed hexadecimal character." >&2
        exit 1
    fi
fi

if [[ ! "$WORK_SIZE" =~ ^[0-9]+$ ]] || [[ "$WORK_SIZE" -lt 32 || "$WORK_SIZE" -gt 256 || $((WORK_SIZE % 32)) -ne 0 ]]; then
    echo "ERROR: WORK_SIZE must be a multiple of 32 between 32 and 256." >&2
    exit 1
fi
if [[ "$INVERSE_SIZE" -ne 255 ]]; then
    echo "ERROR: INVERSE_SIZE must be 255." >&2
    exit 1
fi
if [[ ! "$INVERSE_MULTIPLE" =~ ^[0-9]+$ ]] || [[ "$INVERSE_MULTIPLE" -lt 2 ]]; then
    echo "ERROR: INVERSE_MULTIPLE must be an integer of at least 2." >&2
    exit 1
fi
if [[ ! "$STREAMS" =~ ^[0-9]+$ ]] || [[ "$STREAMS" -lt 1 || "$STREAMS" -gt 8 || $((INVERSE_MULTIPLE % STREAMS)) -ne 0 ]]; then
    echo "ERROR: STREAMS must be 1..8 and divide INVERSE_MULTIPLE." >&2
    exit 1
fi
if [[ ! "$CANARY_INTERVAL" =~ ^[0-9]+$ ]] || [[ "$CANARY_INTERVAL" -ne 0 && "$CANARY_INTERVAL" -lt 20 ]]; then
    echo "ERROR: CANARY_INTERVAL must be 0 or at least 20 rounds." >&2
    exit 1
fi
if [[ ! "$MAX_HITS" =~ ^[0-9]+$ ]] || [[ "$MAX_HITS" -gt 4294967295 ]]; then
    echo "ERROR: MAX_HITS must be 0..4294967295." >&2
    exit 1
fi
if [[ ! "$SHARD_COUNT" =~ ^[1-9][0-9]*$ ]] || [[ ${#SHARD_COUNT} -gt 5 ]] || [[ "$SHARD_COUNT" -gt 65536 ]] ||
   (( (SHARD_COUNT & (SHARD_COUNT - 1)) != 0 )); then
    echo "ERROR: SHARD_COUNT must be a power of two from 1 through 65536." >&2
    exit 1
fi
if [[ ! "$SHARD_INDEX" =~ ^(0|[1-9][0-9]*)$ ]] || [[ ${#SHARD_INDEX} -gt 5 ]] || [[ "$SHARD_INDEX" -ge "$SHARD_COUNT" ]]; then
    echo "ERROR: SHARD_INDEX must be an integer from 0 through SHARD_COUNT-1." >&2
    exit 1
fi

HOST_LABEL=$(hostname -s 2>/dev/null || hostname 2>/dev/null || printf 'worker')
HOST_LABEL=$(printf '%s' "$HOST_LABEL" | tr -cs 'A-Za-z0-9._-' '-')
WORKER_ID="${WORKER_ID_CFG:-${HOST_LABEL}-gpu${TARGET_GPU}}"
if [[ ! "$WORKER_ID" =~ ^[A-Za-z0-9._-]{1,64}$ ]]; then
    echo "ERROR: WORKER_ID must use 1..64 letters, digits, dots, underscores, or hyphens." >&2
    exit 1
fi

mkdir -p "$RESULT_ROOT" "$LOG_ROOT" "$RUNTIME_DIR"
chmod 700 "$RUNTIME_DIR"
USER_ID=$(id -u)
GLOBAL_LOCK_DIR="${XDG_RUNTIME_DIR:-/tmp}/vaddro-${USER_ID}"
mkdir -p "$GLOBAL_LOCK_DIR"
chmod 700 "$GLOBAL_LOCK_DIR"
LOCK_FILE="$GLOBAL_LOCK_DIR/gpu-${TARGET_GPU}.lock"
PID_FILE="$RUNTIME_DIR/gpu-${TARGET_GPU}.pid"
STATE_FILE="$RUNTIME_DIR/gpu-${TARGET_GPU}.state"
exec 9>"$LOCK_FILE"
if ! flock -n 9; then
    ACTIVE_PID=""
    if [[ -f "$PID_FILE" && ! -L "$PID_FILE" ]]; then
        ACTIVE_PID=$(tr -dc '0-9' < "$PID_FILE" 2>/dev/null || true)
    fi
    echo "ERROR: GPU $TARGET_GPU already has an active Vaddro worker${ACTIVE_PID:+ (PID $ACTIVE_PID)}." >&2
    echo "Inspect it with: ./05_status.sh --gpu $TARGET_GPU" >&2
    exit 2
fi

RUN_ID="$(date -u '+%Y%m%dT%H%M%SZ')-$$"
STARTED_UTC="$(date -u '+%Y-%m-%dT%H:%M:%SZ')"
RESULT_OUTPUT_DIR="$RESULT_ROOT"
RUN_LOG_DIR="$LOG_ROOT/$WORKER_ID/$RUN_ID"
JSON_LOG="$RUN_LOG_DIR/events.jsonl"
mkdir -p "$RESULT_OUTPUT_DIR" "$RUN_LOG_DIR"

if [[ -f "$SCRIPT_DIR/BUILD_COMMIT" ]]; then
    BUILD_COMMIT=$(tr -dc 'A-Za-z0-9._-' < "$SCRIPT_DIR/BUILD_COMMIT" | cut -c1-64)
elif command -v git >/dev/null 2>&1; then
    BUILD_COMMIT=$(git -C "$SCRIPT_DIR" rev-parse --verify HEAD 2>/dev/null || true)
else
    BUILD_COMMIT=""
fi
BUILD_COMMIT="${BUILD_COMMIT:-local-build}"
DRIVER_VERSION="unknown"
if command -v nvidia-smi >/dev/null 2>&1; then
    DRIVER_VERSION=$(nvidia-smi -i "$TARGET_GPU" --query-gpu=driver_version --format=csv,noheader 2>/dev/null | head -n 1 | tr -d '[:space:]' || true)
    DRIVER_VERSION="${DRIVER_VERSION:-unknown}"
fi

ENGINE_PID=""
cleanup_runtime() {
    if [[ -n "$ENGINE_PID" && -f "$PID_FILE" ]]; then
        RECORDED_PID=$(tr -dc '0-9' < "$PID_FILE" 2>/dev/null || true)
        if [[ "$RECORDED_PID" == "$ENGINE_PID" ]]; then
            rm -f "$PID_FILE"
        fi
    fi
}
forward_signal() {
    if [[ -n "$ENGINE_PID" && "$ENGINE_PID" =~ ^[0-9]+$ ]] && kill -0 "$ENGINE_PID" 2>/dev/null; then
        kill -TERM "$ENGINE_PID" 2>/dev/null || true
    fi
}
trap cleanup_runtime EXIT
trap forward_signal INT TERM

PROF_ARGS=("-w" "$WORK_SIZE" "-I" "$INVERSE_MULTIPLE" "-i" "$INVERSE_SIZE" "-s" "$STREAMS" "--canary-interval" "$CANARY_INTERVAL" "--max-hits" "$MAX_HITS" "--shard-index" "$SHARD_INDEX" "--shard-count" "$SHARD_COUNT" "-g" "$TARGET_GPU")
PROF_ARGS+=("--worker-id" "$WORKER_ID" "--run-id" "$RUN_ID" "--build-commit" "$BUILD_COMMIT" "--driver-version" "$DRIVER_VERSION" "--json-log" "$JSON_LOG")
if [[ -n "$USER_EXTRA_FLAGS" ]]; then
    # shellcheck disable=SC2206
    PROF_ARGS+=($USER_EXTRA_FLAGS)
fi
if [[ ${#CLI_EXTRA_ARGS[@]} -gt 0 ]]; then
    PROF_ARGS+=("${CLI_EXTRA_ARGS[@]}")
fi

echo "[$(date '+%Y-%m-%d %H:%M:%S')] Startup self-test: GPU=$TARGET_GPU Worker=$WORKER_ID Run=$RUN_ID" | tee -a "$ACTIVITY_LOG"
"$CUDA_EXE" --self-test -g "$TARGET_GPU" | tee -a "$ACTIVITY_LOG"

if [[ "$SAME_ENDS" -gt 0 ]]; then
    MODE_DESC="same-char-ends:$SAME_ENDS"
    ENGINE_ARGS=(--same-char-ends "$SAME_ENDS" -z "$PUBLIC_KEY" -o "$RESULT_OUTPUT_DIR" "${PROF_ARGS[@]}")
else
    MODE_DESC="pattern:$PATTERN_RAW"
    ENGINE_ARGS=(--matching "$PATTERN_RAW" --match-score "$SCORE" -z "$PUBLIC_KEY" -o "$RESULT_OUTPUT_DIR" "${PROF_ARGS[@]}")
fi

echo "[$(date '+%Y-%m-%d %H:%M:%S')] Search start: GPU=$TARGET_GPU Worker=$WORKER_ID Run=$RUN_ID Shard=$SHARD_INDEX/$SHARD_COUNT Mode=$MODE_DESC MaxHits=$MAX_HITS Results=$RESULT_OUTPUT_DIR" >> "$ACTIVITY_LOG"
set +e
"$CUDA_EXE" "${ENGINE_ARGS[@]}" &
ENGINE_PID=$!
printf '%s\n' "$ENGINE_PID" > "$PID_FILE.tmp.$$"
chmod 600 "$PID_FILE.tmp.$$"
mv -f "$PID_FILE.tmp.$$" "$PID_FILE"
{
    printf 'gpu\t%s\n' "$TARGET_GPU"
    printf 'launcher_pid\t%s\n' "$$"
    printf 'engine_pid\t%s\n' "$ENGINE_PID"
    printf 'worker_id\t%s\n' "$WORKER_ID"
    printf 'run_id\t%s\n' "$RUN_ID"
    printf 'shard_index\t%s\n' "$SHARD_INDEX"
    printf 'shard_count\t%s\n' "$SHARD_COUNT"
    printf 'started_utc\t%s\n' "$STARTED_UTC"
    printf 'result_dir\t%s\n' "$RESULT_OUTPUT_DIR"
    printf 'log_dir\t%s\n' "$RUN_LOG_DIR"
    printf 'json_log\t%s\n' "$JSON_LOG"
} > "$STATE_FILE.tmp.$$"
chmod 600 "$STATE_FILE.tmp.$$"
mv -f "$STATE_FILE.tmp.$$" "$STATE_FILE"

wait "$ENGINE_PID"
EXIT_CODE=$?
if kill -0 "$ENGINE_PID" 2>/dev/null; then
    wait "$ENGINE_PID"
    EXIT_CODE=$?
fi
set -e
echo "[$(date '+%Y-%m-%d %H:%M:%S')] Search stop: GPU=$TARGET_GPU Worker=$WORKER_ID Run=$RUN_ID ExitCode=$EXIT_CODE" >> "$ACTIVITY_LOG"
exit "$EXIT_CODE"
