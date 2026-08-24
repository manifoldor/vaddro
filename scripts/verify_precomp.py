#!/usr/bin/env python3
"""Verify every secp256k1 point embedded in cuda/precomp.cu."""

from __future__ import annotations

import hashlib
import re
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from combine import GENERATOR, point_add, scalar_multiply  # noqa: E402

PRECOMP_PATH = PROJECT_ROOT / "cuda" / "precomp.cu"
CHECKSUM_PATH = PROJECT_ROOT / "cuda" / "precomp.cu.sha256"
POINT_COUNT = 32 * 255


def words_to_int(words: list[int]) -> int:
    return sum(word << (32 * index) for index, word in enumerate(words))


def main() -> int:
    raw = PRECOMP_PATH.read_bytes()
    expected_digest = CHECKSUM_PATH.read_text(encoding="ascii").split()[0].lower()
    actual_digest = hashlib.sha256(raw).hexdigest()
    if actual_digest != expected_digest:
        raise SystemExit(
            f"precomp.cu SHA-256 mismatch:\n  expected {expected_digest}\n  actual   {actual_digest}"
        )

    tokens = [int(value, 16) for value in re.findall(rb"0x([0-9a-fA-F]{1,8})", raw)]
    if len(tokens) != POINT_COUNT * 16:
        raise SystemExit(f"Expected {POINT_COUNT * 16} limbs, found {len(tokens)}")

    points = []
    for offset in range(0, len(tokens), 16):
        points.append((words_to_int(tokens[offset:offset + 8]), words_to_int(tokens[offset + 8:offset + 16])))

    for byte_index in range(32):
        base = scalar_multiply(1 << (8 * byte_index), GENERATOR)
        if base is None:
            raise SystemExit(f"Unexpected point at infinity for byte {byte_index}")
        expected = base
        for value in range(1, 256):
            table_index = byte_index * 255 + value - 1
            if points[table_index] != expected:
                raise SystemExit(f"Precomputation mismatch at byte={byte_index}, value={value}")
            expected = point_add(expected, base)

    print(f"Precomputation verification: {POINT_COUNT} points OK ({actual_digest})")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
