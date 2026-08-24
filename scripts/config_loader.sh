#!/usr/bin/env bash
# Parse config/miner.env as data, never as executable shell code.

load_vaddro_config() {
    local env_file="$1"
    local raw key value first last
    local seen_keys="|"

    while IFS= read -r raw || [[ -n "$raw" ]]; do
        raw="${raw#"${raw%%[![:space:]]*}"}"
        raw="${raw%"${raw##*[![:space:]]}"}"
        [[ -z "$raw" || "${raw:0:1}" == "#" ]] && continue
        if [[ "$raw" != *=* ]]; then
            echo "ERROR: Invalid configuration line in $env_file: $raw" >&2
            return 1
        fi

        key="${raw%%=*}"
        value="${raw#*=}"
        key="${key//[[:space:]]/}"
        value="${value#"${value%%[![:space:]]*}"}"
        value="${value%"${value##*[![:space:]]}"}"

        case "$key" in
            PUBLIC_KEY|PATTERN|SAME_CHAR_ENDS|MAX_HITS|WORKER_ID|SHARD_INDEX|SHARD_COUNT|WORK_SIZE|INVERSE_MULTIPLE|INVERSE_SIZE|STREAMS|GPU_INDEX|EXTRA_FLAGS|CANARY_INTERVAL|ALLOW_DEMO_KEY)
                ;;
            *)
                echo "ERROR: Unknown configuration key '$key' in $env_file" >&2
                return 1
                ;;
        esac

        case "$seen_keys" in
            *"|$key|"*)
                echo "ERROR: Duplicate configuration key '$key' in $env_file" >&2
                return 1
                ;;
        esac
        seen_keys="${seen_keys}${key}|"

        if [[ ${#value} -ge 2 ]]; then
            first="${value:0:1}"
            last="${value: -1}"
            if [[ ("$first" == '"' && "$last" == '"') || ("$first" == "'" && "$last" == "'") ]]; then
                value="${value:1:${#value}-2}"
            fi
        fi
        printf -v "$key" '%s' "$value"
        export "$key"
    done < "$env_file"
}
