#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
TEST_BINARY=$(mktemp /tmp/vaddro-split-variant-hash.XXXXXX)
trap 'rm -f "$TEST_BINARY"' EXIT

"${CXX:-c++}" -O3 -std=c++14 -Wall -Wextra -pedantic -Wno-unknown-pragmas \
    "$SCRIPT_DIR/test_split_variant_hash.cpp" -o "$TEST_BINARY"
"$TEST_BINARY"
