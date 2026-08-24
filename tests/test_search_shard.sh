#!/usr/bin/env bash
set -euo pipefail

if ! command -v c++ >/dev/null 2>&1; then
    echo "Search shard partition test: SKIP (requires c++)"
    exit 0
fi

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
TEST_ROOT="$(mktemp -d)"
trap 'rm -rf "$TEST_ROOT"' EXIT

c++ -std=c++14 -Wall -Wextra -Werror \
    -I"$PROJECT_ROOT/cuda" \
    "$PROJECT_ROOT/tests/test_search_shard.cpp" \
    -o "$TEST_ROOT/test-search-shard"

"$TEST_ROOT/test-search-shard"
