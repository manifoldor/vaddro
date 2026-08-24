from __future__ import annotations

import os
import tempfile
import unittest
from pathlib import Path

import combine as combine_module
import seed as seed_module
import verify_result as verify_result_module
from combine import (
    CombineError,
    GROUP_N,
    apply_point_variant,
    apply_variant,
    checksum_address,
    combine,
    evm_address,
    parse_result_file,
    scalar_multiply,
    verify_public_result,
)
from seed import SeedGenError, write_seed_file


SEED_PRIVATE = 1
SEED_PUBLIC = (
    "79be667ef9dcbbac55a06295ce870b07029bfcdb2dce28d959f2815b16f81798"
    "483ada7726a3c4655da4fbfc0e1108a8fd17b448a68554199c47d08ffb10d4b8"
)
SHARE = 0x1234


def write_result(path: Path, variant: int = 0) -> str:
    base = (SEED_PRIVATE + SHARE) % GROUP_N
    final_private = apply_variant(base, variant)
    address = evm_address(scalar_multiply(final_private))
    path.write_text(
        "Split-Key Search Result v1\n"
        f"Address: {address}\n"
        f"Seed Public Key: {SEED_PUBLIC}\n"
        f"Search Share: {SHARE:064x}\n"
        f"Variant: {variant}\n",
        encoding="utf-8",
    )
    return address


class OfflineCryptoTests(unittest.TestCase):
    def test_versions_match_release_file(self) -> None:
        release_version = Path("VERSION").read_text(encoding="ascii").strip()
        self.assertEqual(seed_module.__version__, release_version)
        self.assertEqual(combine_module.__version__, release_version)
        self.assertEqual(verify_result_module.__version__, release_version)

    def test_eip55_checksum_vector(self) -> None:
        self.assertEqual(
            checksum_address("52908400098527886e0f7030069857d2e4169ee7"),
            "0x52908400098527886E0F7030069857D2E4169EE7",
        )

    def test_public_and_private_verification_all_variants(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            seed_path = root / "seed.txt"
            result_path = root / "result.txt"
            wallet_path = root / "wallet.txt"
            write_seed_file(seed_path, SEED_PRIVATE, SEED_PUBLIC)

            for variant in range(6):
                expected = write_result(result_path, variant)
                public_address = verify_public_result(result_path)
                private_address, private_key_hex = combine(
                    seed_path, result_path, wallet_path, force=True
                )
                expected_point = apply_point_variant(
                    scalar_multiply((SEED_PRIVATE + SHARE) % GROUP_N), variant
                )
                self.assertEqual(public_address.lower(), expected.lower())
                self.assertEqual(private_address.lower(), expected.lower())
                self.assertEqual(
                    scalar_multiply(int(private_key_hex, 16)),
                    scalar_multiply(apply_variant((SEED_PRIVATE + SHARE) % GROUP_N, variant)),
                )
                self.assertEqual(evm_address(expected_point).lower(), expected.lower())

    def test_duplicate_result_field_is_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            result_path = Path(temp_dir) / "result.txt"
            write_result(result_path)
            with result_path.open("a", encoding="utf-8") as handle:
                handle.write("Variant: 0\n")
            with self.assertRaisesRegex(CombineError, "Duplicate 'Variant' field"):
                parse_result_file(result_path)

    def test_oversized_result_is_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            result_path = Path(temp_dir) / "result.txt"
            result_path.write_bytes(b"x" * (64 * 1024 + 1))
            with self.assertRaisesRegex(CombineError, "oversized"):
                parse_result_file(result_path)

    def test_result_shard_metadata_is_validated(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            result_path = Path(temp_dir) / "result.txt"
            write_result(result_path)
            with result_path.open("a", encoding="utf-8") as handle:
                handle.write("Shard Index: 0\nShard Count: 4\n")
            parsed = parse_result_file(result_path)
            self.assertEqual(parsed["Shard Index"], 0)
            self.assertEqual(parsed["Shard Count"], 4)

            write_result(result_path)
            with result_path.open("a", encoding="utf-8") as handle:
                handle.write("Shard Index: 1\nShard Count: 4\n")
            with self.assertRaisesRegex(CombineError, "outside its declared shard"):
                parse_result_file(result_path)

            write_result(result_path)
            with result_path.open("a", encoding="utf-8") as handle:
                handle.write("Shard Index: 0\n")
            with self.assertRaisesRegex(CombineError, "must appear together"):
                parse_result_file(result_path)

    def test_seed_write_refuses_overwrite(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            seed_path = Path(temp_dir) / "seed.txt"
            write_seed_file(seed_path, SEED_PRIVATE, SEED_PUBLIC)
            with self.assertRaises(SeedGenError):
                write_seed_file(seed_path, SEED_PRIVATE, SEED_PUBLIC)

    @unittest.skipIf(os.name == "nt", "symlink behavior depends on Windows policy")
    def test_wallet_write_refuses_symlink_even_with_force(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            seed_path = root / "seed.txt"
            result_path = root / "result.txt"
            victim_path = root / "victim.txt"
            wallet_path = root / "wallet.txt"
            victim_path.write_text("do not overwrite", encoding="utf-8")
            wallet_path.symlink_to(victim_path)
            write_seed_file(seed_path, SEED_PRIVATE, SEED_PUBLIC)
            write_result(result_path)
            with self.assertRaisesRegex(CombineError, "symbolic link"):
                combine(seed_path, result_path, wallet_path, force=True)
            self.assertEqual(victim_path.read_text(encoding="utf-8"), "do not overwrite")


if __name__ == "__main__":
    unittest.main()
