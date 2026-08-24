#!/usr/bin/env bash
# Linux Result Check Script
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
RESULT_DIR="$SCRIPT_DIR/result"
LOG_DIR="$SCRIPT_DIR/logs"
ACTIVITY_LOG="$LOG_DIR/activity.log"
ENV_FILE="$SCRIPT_DIR/config/miner.env"
CONFIG_LOADER="$SCRIPT_DIR/scripts/config_loader.sh"

FOUND_COUNT=0
INVALID_COUNT=0
PUBLIC_VERIFIER="$SCRIPT_DIR/verify_result.py"
MAX_HITS=1

echo "Vaddro — Vanity Address Online"
echo "Seed Offline. Compute Online."
echo "Result verification"
echo "=================================================="

if [[ -f "$ENV_FILE" && -f "$CONFIG_LOADER" ]]; then
    # shellcheck disable=SC1090
    source "$CONFIG_LOADER"
    load_vaddro_config "$ENV_FILE"
    MAX_HITS="${MAX_HITS:-1}"
fi

if [[ ! "$MAX_HITS" =~ ^[0-9]+$ ]] || [[ "$MAX_HITS" -gt 4294967295 ]]; then
    echo "WARNING: invalid MAX_HITS in config; showing target as unknown." >&2
    MAX_HITS="unknown"
fi

RESULT_FILES=()
if [[ -d "$RESULT_DIR" ]]; then
    while IFS= read -r result_file; do
        RESULT_FILES+=("$result_file")
    done < <(find "$RESULT_DIR" -type f -name 'split-result-*.txt' -print | sort)
    # result-share.txt is the canonical latest-result alias. Only inspect it
    # when there are no per-address files, so one hit is never counted twice.
    if [[ ${#RESULT_FILES[@]} -eq 0 ]]; then
        while IFS= read -r result_file; do
            RESULT_FILES+=("$result_file")
        done < <(find "$RESULT_DIR" -type f -name 'result-share.txt' -print | sort)
    fi
fi

if [[ ${#RESULT_FILES[@]} -gt 0 ]]; then
    for f in "${RESULT_FILES[@]}"; do
            FOUND_COUNT=$((FOUND_COUNT + 1))
            echo "=================================================="
            echo "🎉 MATCH RESULT: $(basename "$f")"
            echo "=================================================="
            cat "$f"
            echo ""
            if [[ -f "$PUBLIC_VERIFIER" ]]; then
                if python3 "$PUBLIC_VERIFIER" "$f"; then
                    echo "✓ Public-data verification passed."
                else
                    echo "ERROR: Public-data verification failed for $f" >&2
                    INVALID_COUNT=$((INVALID_COUNT + 1))
                fi
            else
                echo "WARNING: verify_result.py is unavailable; result was not independently checked." >&2
            fi
    done
fi

if [[ $FOUND_COUNT -gt 0 ]]; then
    if [[ $INVALID_COUNT -gt 0 ]]; then
        echo "$INVALID_COUNT result(s) failed verification. Do not combine or fund them." >&2
        exit 2
    fi
    if [[ "$MAX_HITS" == "0" ]]; then
        echo "Verified hits: $FOUND_COUNT / unlimited. Copy them back to your safe Mac for offline combination."
    elif [[ "$MAX_HITS" == "unknown" ]]; then
        echo "Verified hits: $FOUND_COUNT / unknown target. Copy them back to your safe Mac for offline combination."
    else
        echo "Verified hits: $FOUND_COUNT / $MAX_HITS. Copy them back to your safe Mac for offline combination."
    fi
    exit 0
fi

echo "No saved match files in result/ directory yet."
if [[ -f "$ACTIVITY_LOG" ]]; then
    echo "Latest activity log entries:"
    tail -n 15 "$ACTIVITY_LOG"
else
    echo "No activity log found. Run ./02_start_search.sh to start."
fi
