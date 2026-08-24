#!/usr/bin/env python3
"""Public verifier for Vaddro worker results.

This tool needs only the public result file. It never reads or handles the
offline seed private key.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from combine import CombineError, self_test, verify_public_result

__version__ = "1.0.0"


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Verify a Vaddro Search Share using public data only."
    )
    parser.add_argument("--version", action="version", version=f"%(prog)s {__version__}")
    parser.add_argument("result", nargs="?", type=Path, help="Worker result-share.txt file")
    parser.add_argument("--self-test", action="store_true", help="Run public cryptographic tests")
    args = parser.parse_args()

    try:
        self_test()
        if args.self_test:
            print("Public verifier self-test: OK")
            return 0
        if args.result is None:
            parser.error("result file is required unless --self-test is used")
        address = verify_public_result(args.result)
        print("Vaddro — Vanity Address Online")
        print("Seed Offline. Compute Online.")
        print("Public result verification: OK")
        print(f"Address: {address}")
        return 0
    except CombineError as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
