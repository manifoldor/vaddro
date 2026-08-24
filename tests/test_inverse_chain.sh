#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
TEST_BINARY_OFF=$(mktemp /tmp/vaddro-inverse-chain-off.XXXXXX)
TEST_BINARY_ON=$(mktemp /tmp/vaddro-inverse-chain-on.XXXXXX)
trap 'rm -f "$TEST_BINARY_OFF" "$TEST_BINARY_ON"' EXIT

"${CXX:-c++}" -O3 -std=c++14 -Wall -Wextra -pedantic -Wno-unknown-pragmas \
    -DVADDRO_INV_ADDITION_CHAIN=0 \
    "$SCRIPT_DIR/test_inverse_chain.cpp" -o "$TEST_BINARY_OFF"
"$TEST_BINARY_OFF"

"${CXX:-c++}" -O3 -std=c++14 -Wall -Wextra -pedantic -Wno-unknown-pragmas \
    -DVADDRO_INV_ADDITION_CHAIN=1 \
    "$SCRIPT_DIR/test_inverse_chain.cpp" -o "$TEST_BINARY_ON"
"$TEST_BINARY_ON"
