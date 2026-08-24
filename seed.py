#!/usr/bin/env python3
"""Vaddro - Cross-platform offline EVM split-key seed generator.

Author: yishan <linyishan@gmail.com>
License: MIT
"""

from __future__ import annotations

__author__ = "yishan <linyishan@gmail.com>"
__version__ = "1.0.0"

import argparse
import os
import secrets
import stat
import sys
from pathlib import Path
from typing import Optional, Tuple

# secp256k1 curve parameters
FIELD_P = 0xFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFEFFFFFC2F
GROUP_N = 0xFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFEBAAEDCE6AF48A03BBFD25E8CD0364141
GENERATOR = (
    0x79BE667EF9DCBBAC55A06295CE870B07029BFCDB2DCE28D959F2815B16F81798,
    0x483ADA7726A3C4655DA4FBFC0E1108A8FD17B448A68554199C47D08FFB10D4B8,
)
Point = Optional[Tuple[int, int]]


class SeedGenError(Exception):
    """An error during seed generation or file write."""


def inverse_mod(value: int, modulus: int) -> int:
    return pow(value % modulus, modulus - 2, modulus)


def point_add(left: Point, right: Point) -> Point:
    if left is None:
        return right
    if right is None:
        return left

    x1, y1 = left
    x2, y2 = right
    if x1 == x2 and (y1 + y2) % FIELD_P == 0:
        return None

    if left == right:
        slope = (3 * x1 * x1) * inverse_mod(2 * y1, FIELD_P) % FIELD_P
    else:
        slope = (y2 - y1) * inverse_mod(x2 - x1, FIELD_P) % FIELD_P

    x3 = (slope * slope - x1 - x2) % FIELD_P
    y3 = (slope * (x1 - x3) - y1) % FIELD_P
    return x3, y3


def scalar_multiply(scalar: int, point: Point = GENERATOR) -> Point:
    scalar %= GROUP_N
    result: Point = None
    addend = point
    while scalar:
        if scalar & 1:
            result = point_add(result, addend)
        addend = point_add(addend, addend)
        scalar >>= 1
    return result


def generate_seed() -> Tuple[int, str]:
    """Generate a cryptographically secure 256-bit scalar and uncompressed public key."""
    # Ensure scalar is in [1, GROUP_N - 1]
    private_key = secrets.randbelow(GROUP_N - 1) + 1
    public_point = scalar_multiply(private_key)
    if public_point is None:
        raise SeedGenError("Generated point at infinity")
    x, y = public_point
    pub_hex = f"{x:064x}{y:064x}"
    return private_key, pub_hex


def _fsync_directory(path: Path) -> None:
    """Best-effort directory sync after an atomic private-file update."""
    if os.name != "posix":
        return
    try:
        descriptor = os.open(path, os.O_RDONLY)
        try:
            os.fsync(descriptor)
        finally:
            os.close(descriptor)
    except OSError:
        # Some filesystems do not support fsync on directories.
        pass


def _atomic_write_private(path: Path, content: str, force: bool = False) -> None:
    """Write private material with mode 0600 and an atomic final update."""
    try:
        path.parent.mkdir(mode=0o700, parents=True, exist_ok=True)
    except OSError as exc:
        raise SeedGenError(f"Failed to create seed directory '{path.parent}': {exc}") from exc

    try:
        existing = path.lstat()
    except FileNotFoundError:
        existing = None

    if existing is not None:
        if stat.S_ISLNK(existing.st_mode):
            raise SeedGenError(f"Refusing to write through symbolic link: '{path}'")
        if not force:
            raise SeedGenError(
                f"File already exists: '{path}'.\n"
                f"Use --force to overwrite, or specify a different output path with --output <path>."
            )
        if not stat.S_ISREG(existing.st_mode):
            raise SeedGenError(f"Refusing to replace non-regular file: '{path}'")

    temp_path = path.parent / f".{path.name}.{os.getpid()}.{secrets.token_hex(8)}.tmp"
    descriptor = None
    try:
        flags = os.O_WRONLY | os.O_CREAT | os.O_EXCL
        if hasattr(os, "O_NOFOLLOW"):
            flags |= os.O_NOFOLLOW
        descriptor = os.open(temp_path, flags, 0o600)
        with os.fdopen(descriptor, "w", encoding="ascii", newline="\n") as handle:
            descriptor = None
            handle.write(content)
            handle.flush()
            os.fsync(handle.fileno())
        if os.name == "posix":
            os.chmod(temp_path, 0o600)

        if force:
            os.replace(temp_path, path)
        else:
            # Hard-linking provides an atomic no-overwrite commit on POSIX and Windows.
            os.link(temp_path, path)
            os.unlink(temp_path)
        _fsync_directory(path.parent)
    except FileExistsError as exc:
        raise SeedGenError(f"File already exists: '{path}'. Use --force to overwrite it.") from exc
    except OSError as exc:
        raise SeedGenError(f"Failed to write seed file '{path}': {exc}") from exc
    finally:
        if descriptor is not None:
            os.close(descriptor)
        try:
            temp_path.unlink()
        except FileNotFoundError:
            pass


def write_seed_file(path: Path, private_key: int, public_hex: str, force: bool = False) -> None:
    if not (1 <= private_key < GROUP_N):
        raise SeedGenError("Seed private key must be in the secp256k1 scalar range")
    if len(public_hex) != 128 or any(c not in "0123456789abcdefABCDEF" for c in public_hex):
        raise SeedGenError("Seed public key must contain exactly 128 hexadecimal characters")

    content = (
        "Split-Key Seed v1\n"
        f"Seed Private Key: {private_key:064x}\n"
        f"Seed Public Key: {public_hex.lower()}\n"
    )
    try:
        _atomic_write_private(path, content, force=force)
    except SeedGenError:
        raise


def self_test() -> None:
    if scalar_multiply(1) != GENERATOR:
        raise SeedGenError("secp256k1 generator test failed")
    if scalar_multiply(GROUP_N) is not None:
        raise SeedGenError("secp256k1 group order test failed")
    priv, pub = generate_seed()
    if len(pub) != 128 or not (1 <= priv < GROUP_N):
        raise SeedGenError("Seed sanity check failed")


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Cross-platform offline EVM Split-Key seed generator (pure Python standard library)."
    )
    parser.add_argument("--version", action="version", version=f"%(prog)s {__version__}")
    parser.add_argument(
        "--output", "-o",
        type=Path,
        default=Path("seeds/split-seed.txt"),
        help="Target seed file path (default: seeds/split-seed.txt)",
    )
    parser.add_argument(
        "--force", "-f",
        action="store_true",
        help="Overwrite target file if it already exists",
    )
    parser.add_argument(
        "--self-test",
        action="store_true",
        help="Run cryptographic verification tests and exit",
    )
    args = parser.parse_args()

    try:
        self_test()
        if args.self_test:
            print("Cryptographic Self-test: OK")
            return 0

        priv, pub = generate_seed()

        write_seed_file(args.output, priv, pub, force=args.force)

        print("==================================================")
        print("Vaddro — Vanity Address Online")
        print("Seed Offline. Compute Online.")
        print("🔑 Split-Key Seed Generated Successfully!")
        print("==================================================")
        print(f"📁 Seed file saved to:  {args.output}")
        if os.name == "posix":
            print("🔒 File permissions:    0600 (read/write by owner only)")
        print("\n⚡ Seed Public Key (Safe to copy to worker config/miner.env):")
        print("--------------------------------------------------")
        print(pub)
        print("--------------------------------------------------")
        print("\n⚠️  SECURITY NOTICE:")
        print(f"1. Keep '{args.output}' SAFE and OFFLINE on this machine.")
        print("2. DO NOT upload or copy the seed file to the mining workers.")
        print("3. Copy ONLY the 128-hex Public Key above into the worker's config/miner.env (PUBLIC_KEY).")
        print("==================================================")
        return 0

    except SeedGenError as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
