#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
TEST_DIR="$(mktemp -d)"
trap 'rm -rf "$TEST_DIR"' EXIT
MARKER="$TEST_DIR/should-not-exist"
CONFIG="$TEST_DIR/miner.env"

printf 'PUBLIC_KEY="$(touch %s)"\n' "$MARKER" > "$CONFIG"
printf '%s\n' \
    'PATTERN="000000XXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXX"' \
    'SAME_CHAR_ENDS=0' \
    'MAX_HITS=3' \
    'WORKER_ID=worker-01' \
    'SHARD_INDEX=2' \
    'SHARD_COUNT=4' >> "$CONFIG"

# shellcheck disable=SC1091
source "$PROJECT_ROOT/scripts/config_loader.sh"
load_vaddro_config "$CONFIG"

[[ "$PUBLIC_KEY" == "\$(touch $MARKER)" ]]
[[ "$MAX_HITS" == "3" ]]
[[ "$WORKER_ID" == "worker-01" ]]
[[ "$SHARD_INDEX" == "2" ]]
[[ "$SHARD_COUNT" == "4" ]]
[[ ! -e "$MARKER" ]]

printf '%s\n' 'UNSUPPORTED_KEY=value' > "$CONFIG"
if load_vaddro_config "$CONFIG" 2>/dev/null; then
    echo "ERROR: unknown configuration key was accepted" >&2
    exit 1
fi

printf '%s\n' 'STREAMS=8' 'STREAMS=4' > "$CONFIG"
if load_vaddro_config "$CONFIG" 2>/dev/null; then
    echo "ERROR: duplicate configuration key was accepted" >&2
    exit 1
fi

echo "Configuration loader safety test: OK"
