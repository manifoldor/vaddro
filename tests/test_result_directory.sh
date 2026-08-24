#!/usr/bin/env bash
set -euo pipefail

if ! command -v c++ >/dev/null 2>&1; then
    echo "Result directory recovery test: SKIP (requires c++)"
    exit 0
fi

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
TEST_ROOT="$(mktemp -d)"
trap 'rm -rf "$TEST_ROOT"' EXIT
TEST_ROOT="$(cd "$TEST_ROOT" && pwd -P)"

c++ -std=c++14 -Wall -Wextra -Werror \
    -I"$PROJECT_ROOT/cuda" \
    "$PROJECT_ROOT/tests/test_result_directory.cpp" \
    -o "$TEST_ROOT/test-result-directory"

"$TEST_ROOT/test-result-directory" "$TEST_ROOT/runtime"
