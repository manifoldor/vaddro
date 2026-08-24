#!/usr/bin/env python3
"""Vaddro - Cross-platform offline EVM split-key combiner & verifier.

Author: yishan <linyishan@gmail.com>
License: MIT
"""

from __future__ import annotations

__author__ = "yishan <linyishan@gmail.com>"
__version__ = "1.0.0"

import argparse
import os
import re
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

# GLV endomorphism scalar lambda_glv where lambda_glv^3 = 1 mod N
LAMBDA_GLV = 0x5363AD4CC05C30E0A5261C028812645A122E22EA20816678DF02967C1B23BD72

# secp256k1 GLV beta coordinates (for reference)
BETA = 0x7AE96A2B657C07106E64479EAC3434E99CF0497512F58995C1396C28719501EE

KECCAK_ROUND_CONSTANTS = (
    0x0000000000000001, 0x0000000000008082,
    0x800000000000808A, 0x8000000080008000,
    0x000000000000808B, 0x0000000080000001,
    0x8000000080008081, 0x8000000000008009,
    0x000000000000008A, 0x0000000000000088,
    0x0000000080008009, 0x000000008000000A,
    0x000000008000808B, 0x800000000000008B,
    0x8000000000008089, 0x8000000000008003,
    0x8000000000008002, 0x8000000000000080,
    0x000000000000800A, 0x800000008000000A,
    0x8000000080008081, 0x8000000000008080,
    0x0000000080000001, 0x8000000080008008,
)

KECCAK_ROTATIONS = (
    (0, 36, 3, 41, 18),
    (1, 44, 10, 45, 2),
    (62, 6, 43, 15, 61),
    (28, 55, 25, 21, 56),
    (27, 20, 39, 8, 14),
)

MASK64 = (1 << 64) - 1
MAX_INPUT_BYTES = 64 * 1024
HEX_40_RE = re.compile(r"^[0-9a-fA-F]{40}$")
HEX_64_RE = re.compile(r"^[0-9a-fA-F]{64}$")
HEX_128_RE = re.compile(r"^[0-9a-fA-F]{128}$")


class CombineError(Exception):
    """A validation or safe-output error."""


def _strip_0x(value: str) -> str:
    return value[2:] if value.lower().startswith("0x") else value


def rotate_left_64(value: int, amount: int) -> int:
    if amount == 0:
        return value & MASK64
    return ((value << amount) | (value >> (64 - amount))) & MASK64


def keccak_f1600(state: list[int]) -> None:
    for round_constant in KECCAK_ROUND_CONSTANTS:
        column = [
            state[x] ^ state[x + 5] ^ state[x + 10] ^ state[x + 15] ^ state[x + 20]
            for x in range(5)
        ]
        delta = [column[(x - 1) % 5] ^ rotate_left_64(column[(x + 1) % 5], 1) for x in range(5)]
        for y in range(5):
            for x in range(5):
                state[x + 5 * y] ^= delta[x]

        moved = [0] * 25
        for y in range(5):
            for x in range(5):
                moved[y + 5 * ((2 * x + 3 * y) % 5)] = rotate_left_64(
                    state[x + 5 * y], KECCAK_ROTATIONS[x][y]
                )

        for y in range(5):
            row = moved[5 * y:5 * y + 5]
            for x in range(5):
                state[x + 5 * y] = row[x] ^ ((~row[(x + 1) % 5]) & row[(x + 2) % 5])
                state[x + 5 * y] &= MASK64

        state[0] ^= round_constant


def keccak_256(data: bytes) -> bytes:
    """Ethereum's legacy Keccak-256."""
    rate = 136
    padded = bytearray(data)
    padded.append(0x01)
    padded.extend(b"\x00" * ((rate - 1 - len(padded)) % rate))
    padded.append(0x80)

    state = [0] * 25
    for block_start in range(0, len(padded), rate):
        block = padded[block_start:block_start + rate]
        for lane in range(rate // 8):
            state[lane] ^= int.from_bytes(block[lane * 8:lane * 8 + 8], "little")
        keccak_f1600(state)

    output = b"".join(lane.to_bytes(8, "little") for lane in state)
    return output[:32]


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


def evm_address(public_point: Tuple[int, int]) -> str:
    x, y = public_point
    raw_public = x.to_bytes(32, "big") + y.to_bytes(32, "big")
    digest = keccak_256(raw_public)
    return "0x" + digest[-20:].hex()


def checksum_address(address: str) -> str:
    """Return an EIP-55 checksummed address."""
    normalized = _strip_0x(address.lower())
    if not HEX_40_RE.fullmatch(normalized):
        raise CombineError("Address must contain exactly 40 hexadecimal characters")
    digest = keccak_256(normalized.encode("ascii")).hex()
    encoded = "".join(
        char.upper() if char in "abcdef" and int(digest[index], 16) >= 8 else char
        for index, char in enumerate(normalized)
    )
    return "0x" + encoded


def _safe_read_text(path: Path, file_type: str = "file") -> str:
    # 1. Detect macOS AppleDouble metadata files (e.g. ._split-result-...)
    if path.name.startswith("._"):
        real_file = path.parent / path.name[2:]
        if real_file.is_file():
            print(f"[*] Auto-detected macOS AppleDouble metadata file '{path.name}'.")
            print(f"    Redirecting to actual text file: '{real_file.name}'")
            path = real_file
        else:
            raise CombineError(
                f"'{path.name}' is a macOS binary resource fork (AppleDouble) file, not the actual text {file_type}.\n"
                f"Please select the real text file '{path.name[2:]}' (without the '._' prefix)."
            )

    if not path.is_file():
        raise CombineError(f"{file_type.capitalize()} not found: '{path}'")

    try:
        size = path.stat().st_size
    except OSError as exc:
        raise CombineError(f"Failed to inspect {file_type} '{path}': {exc}") from exc
    if size > MAX_INPUT_BYTES:
        raise CombineError(
            f"Refusing oversized {file_type} '{path}' ({size} bytes; limit is {MAX_INPUT_BYTES})"
        )

    try:
        raw_bytes = path.read_bytes()
    except OSError as exc:
        raise CombineError(f"Failed to read {file_type} '{path}': {exc}") from exc
    if len(raw_bytes) > MAX_INPUT_BYTES:
        raise CombineError(f"Refusing oversized {file_type} '{path}'")

    # Check for AppleDouble / AppleSingle magic headers (0x00051607 / 0x00051600)
    if raw_bytes.startswith(b"\x00\x05\x16\x07") or raw_bytes.startswith(b"\x00\x05\x16\x00"):
        real_file = path.parent / path.name[2:] if path.name.startswith("._") else None
        if real_file and real_file.is_file():
            return _safe_read_text(real_file, file_type)
        raise CombineError(
            f"'{path.name}' is a macOS binary metadata file (AppleDouble), not a valid text {file_type}.\n"
            f"Please copy and use the real text file from your USB drive / directory."
        )

    try:
        return raw_bytes.decode("utf-8-sig")
    except UnicodeDecodeError:
        raise CombineError(f"{file_type.capitalize()} is not valid UTF-8 text")


def _parse_key_value_text(text: str, path: Path, file_type: str) -> dict[str, str]:
    data: dict[str, str] = {}
    for raw_line in text.splitlines():
        line = raw_line.strip()
        if not line or ":" not in line:
            continue
        key, value = line.split(":", 1)
        key = key.strip()
        if key in data:
            raise CombineError(f"Duplicate '{key}' field in {file_type} '{path}'")
        data[key] = value.strip()
    return data


def _parse_public_key(public_hex: str) -> Tuple[int, int]:
    normalized = _strip_0x(public_hex.lower())
    if not HEX_128_RE.fullmatch(normalized):
        raise CombineError("Seed Public Key must contain exactly 128 hexadecimal characters")
    point = int(normalized[:64], 16), int(normalized[64:], 16)
    x, y = point
    if x >= FIELD_P or y >= FIELD_P or (y * y - x * x * x - 7) % FIELD_P != 0:
        raise CombineError("Seed Public Key is not a valid secp256k1 affine point")
    return point


def parse_seed_file(path: Path | str) -> Tuple[int, str]:
    path = Path(path)
    text = _safe_read_text(path, "seed file")
    data = _parse_key_value_text(text, path, "seed file")
    priv_hex = data.get("Seed Private Key")
    pub_hex = _strip_0x(data.get("Seed Public Key", "").lower())

    if not priv_hex or not pub_hex:
        raise CombineError(f"Invalid seed file format in '{path}'")

    if not HEX_64_RE.fullmatch(priv_hex) or not HEX_128_RE.fullmatch(pub_hex):
        raise CombineError(f"Invalid seed key encoding in '{path}'")

    priv_int = int(priv_hex, 16)
    if not (1 <= priv_int < GROUP_N):
        raise CombineError("Seed private key is outside the secp256k1 scalar range")
    _parse_public_key(pub_hex)
    derived_pub = scalar_multiply(priv_int)
    if derived_pub is None:
        raise CombineError("Invalid seed private key (derived point at infinity)")

    dx, dy = derived_pub
    if f"{dx:064x}{dy:064x}".lower() != pub_hex:
        raise CombineError("Seed private key does not match seed public key in seed file")

    return priv_int, pub_hex


def parse_result_file(path: Path | str) -> dict:
    path = Path(path)
    text = _safe_read_text(path, "result file")
    data = _parse_key_value_text(text, path, "result file")

    required = ["Address", "Seed Public Key", "Search Share"]
    for r in required:
        if r not in data:
            raise CombineError(
                f"Missing '{r}' in result file '{path}'.\n"
                f"Please ensure you specified the real result text file, not a corrupted or metadata file."
            )

    variant_text = data.get("Variant", "0")
    if not re.fullmatch(r"[0-5]", variant_text):
        raise CombineError("Variant must be an integer from 0 through 5")
    data["Variant"] = int(variant_text)
    data["Seed Public Key"] = _strip_0x(data["Seed Public Key"].lower())
    data["Search Share"] = _strip_0x(data["Search Share"].lower())
    address_hex = _strip_0x(data["Address"].lower())
    if not HEX_128_RE.fullmatch(data["Seed Public Key"]):
        raise CombineError("Seed Public Key must contain exactly 128 hexadecimal characters")
    if not HEX_64_RE.fullmatch(data["Search Share"]):
        raise CombineError("Search Share must contain exactly 64 hexadecimal characters")
    if not HEX_40_RE.fullmatch(address_hex):
        raise CombineError("Address must contain exactly 40 hexadecimal characters")
    _parse_public_key(data["Seed Public Key"])
    share_scalar = int(data["Search Share"], 16)
    if not (0 <= share_scalar < GROUP_N):
        raise CombineError("Search Share is outside the secp256k1 scalar range")

    has_shard_index = "Shard Index" in data
    has_shard_count = "Shard Count" in data
    if has_shard_index != has_shard_count:
        raise CombineError("Shard Index and Shard Count must appear together")
    if has_shard_index:
        shard_index_text = data["Shard Index"]
        shard_count_text = data["Shard Count"]
        if not re.fullmatch(r"0|[1-9][0-9]*", shard_index_text):
            raise CombineError("Shard Index must be a non-negative integer")
        if not re.fullmatch(r"[1-9][0-9]*", shard_count_text):
            raise CombineError("Shard Count must be a positive integer")
        shard_index = int(shard_index_text)
        shard_count = int(shard_count_text)
        if (
            shard_count > 65536
            or shard_count & (shard_count - 1)
            or shard_index >= shard_count
        ):
            raise CombineError(
                "Shard Count must be a power of two up to 65536 and Shard Index must be in range"
            )
        shard_width = (1 << 48) // shard_count
        share_high = share_scalar >> 192
        shard_begin = shard_width * shard_index
        if not shard_begin <= share_high < shard_begin + shard_width:
            raise CombineError("Search Share is outside its declared shard")
        data["Shard Index"] = shard_index
        data["Shard Count"] = shard_count
    data["Address"] = "0x" + address_hex
    return data


def apply_variant(base_scalar: int, variant: int) -> int:
    """Applies GLV endomorphism & point negation to the private scalar."""
    if variant == 0:
        return base_scalar % GROUP_N
    elif variant == 1:
        return (base_scalar * LAMBDA_GLV) % GROUP_N
    elif variant == 2:
        return (base_scalar * pow(LAMBDA_GLV, 2, GROUP_N)) % GROUP_N
    elif variant == 3:
        return (GROUP_N - (base_scalar % GROUP_N)) % GROUP_N
    elif variant == 4:
        return (GROUP_N - ((base_scalar * LAMBDA_GLV) % GROUP_N)) % GROUP_N
    elif variant == 5:
        return (GROUP_N - ((base_scalar * pow(LAMBDA_GLV, 2, GROUP_N)) % GROUP_N)) % GROUP_N
    else:
        raise CombineError(f"Unsupported variant: {variant} (must be 0..5)")


def apply_point_variant(point: Tuple[int, int], variant: int) -> Tuple[int, int]:
    """Apply the worker's GLV/negation variant directly to a public point."""
    x, y = point
    if variant in (1, 4):
        x = x * BETA % FIELD_P
    elif variant in (2, 5):
        x = x * pow(BETA, 2, FIELD_P) % FIELD_P
    elif variant not in (0, 3):
        raise CombineError(f"Unsupported variant: {variant} (must be 0..5)")
    if variant >= 3:
        y = (-y) % FIELD_P
    return x, y


def verify_public_result(result_path: Path | str) -> str:
    """Verify a worker result using public data only; no seed private key required."""
    result = parse_result_file(result_path)
    seed_point = _parse_public_key(result["Seed Public Key"])
    share_point = scalar_multiply(int(result["Search Share"], 16))
    base_point = point_add(seed_point, share_point)
    if base_point is None:
        raise CombineError("Public result derives the point at infinity")
    final_point = apply_point_variant(base_point, result["Variant"])
    derived = evm_address(final_point)
    if derived.lower() != result["Address"].lower():
        raise CombineError(
            "Public result verification failed!\n"
            f"  Derived:  {derived}\n"
            f"  Expected: {result['Address']}"
        )
    return checksum_address(derived)


def _fsync_directory(path: Path) -> None:
    if os.name != "posix":
        return
    try:
        descriptor = os.open(path, os.O_RDONLY)
        try:
            os.fsync(descriptor)
        finally:
            os.close(descriptor)
    except OSError:
        pass


def _atomic_write_private(path: Path, content: str, force: bool) -> None:
    try:
        path.parent.mkdir(mode=0o700, parents=True, exist_ok=True)
    except OSError as exc:
        raise CombineError(f"Failed to create wallet directory '{path.parent}': {exc}") from exc
    try:
        existing = path.lstat()
    except FileNotFoundError:
        existing = None
    if existing is not None:
        if stat.S_ISLNK(existing.st_mode):
            raise CombineError(f"Refusing to write through symbolic link: '{path}'")
        if not force:
            raise CombineError(f"Output file already exists: '{path}'. Use --force to replace it.")
        if not stat.S_ISREG(existing.st_mode):
            raise CombineError(f"Refusing to replace non-regular file: '{path}'")

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
            os.link(temp_path, path)
            os.unlink(temp_path)
        _fsync_directory(path.parent)
    except FileExistsError as exc:
        raise CombineError(f"Output file already exists: '{path}'") from exc
    except OSError as exc:
        raise CombineError(f"Failed to write wallet file '{path}': {exc}") from exc
    finally:
        if descriptor is not None:
            os.close(descriptor)
        try:
            temp_path.unlink()
        except FileNotFoundError:
            pass


def combine(
    seed_path: Path | str,
    result_path: Path | str,
    output_path: Path | str,
    force: bool = False,
) -> Tuple[str, str]:
    seed_path = Path(seed_path)
    result_path = Path(result_path)
    output_path = Path(output_path)
    seed_priv, seed_pub = parse_seed_file(seed_path)
    res = parse_result_file(result_path)

    if res["Seed Public Key"] != seed_pub:
        raise CombineError(
            f"Seed public key mismatch!\n"
            f"  Seed file pubkey:   {seed_pub}\n"
            f"  Result file pubkey: {res['Seed Public Key']}"
        )

    share_scalar = int(res["Search Share"], 16)
    base_scalar = (seed_priv + share_scalar) % GROUP_N
    final_priv = apply_variant(base_scalar, res["Variant"])

    final_pub_point = scalar_multiply(final_priv)
    if final_pub_point is None:
        raise CombineError("Derived final public point at infinity")

    derived_address = evm_address(final_pub_point)
    expected_address = res["Address"]

    if derived_address.lower() != expected_address.lower():
        raise CombineError(
            f"Address derivation check failed!\n"
            f"  Derived:  {derived_address}\n"
            f"  Expected: {expected_address}"
        )

    checksummed = checksum_address(derived_address)
    content = (
        "EVM Split-Key Vanity Wallet v1\n"
        f"Address: {checksummed}\n"
        f"Private Key: {final_priv:064x}\n"
        f"Seed Public Key: {seed_pub}\n"
        f"Search Share: {res['Search Share']}\n"
        f"Variant: {res['Variant']}\n"
    )
    if "Pattern" in res:
        content += f"Pattern: {res['Pattern']}\n"
    _atomic_write_private(output_path, content, force=force)

    return checksummed, f"{final_priv:064x}"


def self_test() -> None:
    if scalar_multiply(1) != GENERATOR:
        raise CombineError("secp256k1 generator test failed")
    if scalar_multiply(GROUP_N) is not None:
        raise CombineError("secp256k1 group order test failed")
    known_addr = "0x7e5f4552091a69125d5dfcb7b8c2659029395bdf"
    if evm_address(GENERATOR) != known_addr:
        raise CombineError("EVM address test vector failed")
    checksum_vector = "0x52908400098527886E0F7030069857D2E4169EE7"
    if checksum_address(checksum_vector.lower()) != checksum_vector:
        raise CombineError("EIP-55 checksum test vector failed")
    for variant in range(6):
        scalar_point = scalar_multiply(apply_variant(1, variant))
        point_variant = apply_point_variant(GENERATOR, variant)
        if scalar_point != point_variant:
            raise CombineError(f"GLV public-point variant {variant} test failed")


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Cross-platform offline EVM Split-Key combiner (pure Python standard library).",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python combine.py seeds/split-seed.txt result/result-share.txt wallets/wallet.txt
  python combine.py --seed seeds/split-seed.txt --result result/result-share.txt --output wallets/wallet.txt
        """
    )
    parser.add_argument("--version", action="version", version=f"%(prog)s {__version__}")
    # Positional args
    parser.add_argument("pos_seed", nargs="?", type=Path, help="Path to offline split-seed.txt")
    parser.add_argument("pos_result", nargs="?", type=Path, help="Path to result-share.txt from miner")
    parser.add_argument("pos_output", nargs="?", type=Path, help="Output path for final wallet.txt")

    # Named args
    parser.add_argument("--seed", "-s", type=Path, help="Path to offline split-seed.txt")
    parser.add_argument("--result", "-r", type=Path, help="Path to result-share.txt from miner")
    parser.add_argument("--output", "-o", type=Path, help="Output path for final wallet.txt")
    parser.add_argument("--force", "-f", action="store_true", help="Overwrite output file if it exists")
    parser.add_argument("--self-test", action="store_true", help="Run cryptographic verification and exit")

    args = parser.parse_args()

    try:
        self_test()
        if args.self_test:
            print("Cryptographic Self-test: OK")
            return 0

        seed_file = args.seed or args.pos_seed
        result_file = args.result or args.pos_result
        output_file = args.output or args.pos_output

        if not seed_file or not result_file or not output_file:
            parser.print_help()
            print("\nERROR: Please specify seed, result, and output file paths.", file=sys.stderr)
            return 1

        address, _ = combine(
            seed_path=seed_file,
            result_path=result_file,
            output_path=output_file,
            force=args.force,
        )

        print("==================================================")
        print("Vaddro — Vanity Address Online")
        print("Seed Offline. Compute Online.")
        print("🎉 Split-Key Combined & Verified Successfully!")
        print("==================================================")
        print(f"💎 Final EVM Address:  {address}")
        print(f"📁 Wallet saved to:    {output_file}")
        if os.name == "posix":
            print("🔒 File permissions:   0600 (protected)")
        print("\n🔐 Note: The private key has been safely written only to the wallet file.")
        print("==================================================")
        return 0

    except CombineError as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
