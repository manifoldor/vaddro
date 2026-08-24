#!/usr/bin/env bash
set -euo pipefail

if [[ "$(uname -s)" != "Linux" ]] || ! command -v cc >/dev/null 2>&1 || ! command -v flock >/dev/null 2>&1; then
    echo "Runtime script test: SKIP (requires Linux, cc, and flock)"
    exit 0
fi

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
TEST_ROOT="$(mktemp -d)"
trap 'rm -rf "$TEST_ROOT"' EXIT

mkdir -p "$TEST_ROOT/cuda/bin" "$TEST_ROOT/config" "$TEST_ROOT/scripts"
cp "$PROJECT_ROOT/02_start_search.sh" "$PROJECT_ROOT/04_stop_search.sh" "$PROJECT_ROOT/05_status.sh" "$TEST_ROOT/"
cp "$PROJECT_ROOT/scripts/config_loader.sh" "$TEST_ROOT/scripts/"

cat > "$TEST_ROOT/fake_engine.c" <<'C'
#include <signal.h>
#include <string.h>
#include <time.h>
#include <unistd.h>

static volatile sig_atomic_t running = 1;
static void stop_handler(int sig) { (void)sig; running = 0; }

int main(int argc, char **argv) {
    int i;
    for (i = 1; i < argc; ++i) {
        if (strcmp(argv[i], "--self-test") == 0) return 0;
    }
    signal(SIGTERM, stop_handler);
    signal(SIGINT, stop_handler);
    while (running) {
        struct timespec delay = {0, 100000000};
        nanosleep(&delay, 0);
    }
    return 0;
}
C
cc "$TEST_ROOT/fake_engine.c" -o "$TEST_ROOT/cuda/bin/vaddro-cuda"

cat > "$TEST_ROOT/config/miner.env" <<'ENV'
PUBLIC_KEY="79be667ef9dcbbac55a06295ce870b07029bfcdb2dce28d959f2815b16f81798483ada7726a3c4655da4fbfc0e1108a8fd17b448a68554199c47d08ffb10d4b8"
PATTERN="0XXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXX"
SAME_CHAR_ENDS=0
MAX_HITS=1
WORKER_ID="runtime-test"
SHARD_INDEX=2
SHARD_COUNT=4
WORK_SIZE=32
INVERSE_MULTIPLE=2
INVERSE_SIZE=255
STREAMS=1
CANARY_INTERVAL=20
GPU_INDEX=0
EXTRA_FLAGS=""
ENV

(cd "$TEST_ROOT" && ./02_start_search.sh > start.log 2>&1) &
LAUNCHER_PID=$!
for _ in $(seq 1 100); do
    [[ -f "$TEST_ROOT/run/gpu-0.pid" ]] && break
    sleep 0.05
done
[[ -f "$TEST_ROOT/run/gpu-0.pid" ]]
ENGINE_PID=$(tr -dc '0-9' < "$TEST_ROOT/run/gpu-0.pid")
kill -0 "$ENGINE_PID"

set +e
(cd "$TEST_ROOT" && ./02_start_search.sh > duplicate.log 2>&1)
DUPLICATE_RC=$?
set -e
[[ "$DUPLICATE_RC" -eq 2 ]]
grep -q "already has an active Vaddro worker" "$TEST_ROOT/duplicate.log"

(cd "$TEST_ROOT" && ./05_status.sh --gpu 0) > "$TEST_ROOT/status-running.log"
grep -q "GPU 0  RUNNING" "$TEST_ROOT/status-running.log"
grep -q "runtime-test" "$TEST_ROOT/status-running.log"
grep -q "Shard:        2/4" "$TEST_ROOT/status-running.log"
grep -Fq "result_dir"$'\t'"$TEST_ROOT/result" "$TEST_ROOT/run/gpu-0.state"
grep -Fq "shard_index"$'\t'"2" "$TEST_ROOT/run/gpu-0.state"
grep -Fq "shard_count"$'\t'"4" "$TEST_ROOT/run/gpu-0.state"

(cd "$TEST_ROOT" && ./04_stop_search.sh --gpu 0) > "$TEST_ROOT/stop.log"
wait "$LAUNCHER_PID"
[[ ! -f "$TEST_ROOT/run/gpu-0.pid" ]]
(cd "$TEST_ROOT" && ./05_status.sh --gpu 0) > "$TEST_ROOT/status-stopped.log"
grep -q "GPU 0  STOPPED" "$TEST_ROOT/status-stopped.log"

# A PID belonging to another executable must never be signalled.
printf '%s\n' "$$" > "$TEST_ROOT/run/gpu-2.pid"
(cd "$TEST_ROOT" && ./04_stop_search.sh --gpu 2) > "$TEST_ROOT/stop-mismatch.log"
kill -0 "$$"
grep -q "no process was signalled" "$TEST_ROOT/stop-mismatch.log"

echo "Runtime script lock/PID/status test: OK"
