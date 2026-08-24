#!/usr/bin/env python3
"""Fail if a publishable repository file appears to contain private material."""

from __future__ import annotations

import re
import subprocess
import sys
from pathlib import Path
from typing import List, Optional


PRIVATE_KEY_RE = re.compile(
    rb"(?:Seed Private Key|Private Key)\s*:\s*(?:0x)?[0-9a-fA-F]{64}(?![0-9a-fA-F])"
)
SENSITIVE_DIRS = {"seeds", "wallets", "result"}
CONFIDENTIAL_DIRS = {"docs"}
CONFIDENTIAL_FILES = {
    "DESIGN.md",
    "ARTICLE_CN.md",
    "README.txt",
    "RELEASE_CHECKLIST.md",
    "VALIDATION.md",
    "CONTRIBUTING.md",
    "CODE_OF_CONDUCT.md",
    "CITATION.cff",
    ".github/pull_request_template.md",
}
ALLOWED_NAMES = {".gitkeep", ".gitignore", ".DS_Store"}


def _git_files(*arguments: str) -> Optional[List[Path]]:
    completed = subprocess.run(
        ["git", "ls-files", "-z", *arguments],
        stdout=subprocess.PIPE,
        stderr=subprocess.DEVNULL,
    )
    if completed.returncode != 0:
        return None
    return [Path(item) for item in completed.stdout.decode().split("\0") if item]


def publishable_files() -> List[Path]:
    """Return tracked plus non-ignored untracked files, including a first commit."""
    tracked = _git_files()
    untracked = _git_files("--others", "--exclude-standard")
    if tracked is not None and untracked is not None:
        return list(dict.fromkeys([*tracked, *untracked]))

    excluded = {".git", "dist", "docs", "__pycache__", ".venv", "venv"}
    return [
        path
        for path in Path(".").rglob("*")
        if path.is_file() and not any(part in excluded for part in path.parts)
    ]


def main() -> int:
    findings: list[str] = []
    tracked = _git_files() or []
    for path in tracked:
        if path.parts and path.parts[0] in CONFIDENTIAL_DIRS:
            findings.append(f"confidential path is tracked by Git: {path}")
        if path.as_posix() in CONFIDENTIAL_FILES:
            findings.append(f"internal documentation is tracked by Git: {path}")

    for path in publishable_files():
        if not path.is_file():
            continue
        if path.parts and path.parts[0] in SENSITIVE_DIRS and path.name not in ALLOWED_NAMES:
            findings.append(f"repository-sensitive-path file: {path}")
        try:
            payload = path.read_bytes()
        except OSError as exc:
            findings.append(f"could not inspect {path}: {exc}")
            continue
        if PRIVATE_KEY_RE.search(payload):
            findings.append(f"private-key-like value: {path}")

    if findings:
        print("Sensitive material check failed:", file=sys.stderr)
        for finding in findings:
            print(f"- {finding}", file=sys.stderr)
        return 1
    print("Sensitive material check passed.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
