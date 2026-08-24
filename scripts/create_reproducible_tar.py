#!/usr/bin/env python3
"""Create a deterministic gzip-compressed tar archive from a directory."""

from __future__ import annotations

import argparse
import gzip
import os
import tarfile
from pathlib import Path


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("source", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument(
        "--epoch",
        type=int,
        default=int(os.environ.get("SOURCE_DATE_EPOCH", "0")),
        help="Timestamp applied to all archive entries",
    )
    args = parser.parse_args()

    source = args.source.resolve()
    paths = [source] + sorted(source.rglob("*"), key=lambda path: path.relative_to(source).as_posix())
    args.output.parent.mkdir(parents=True, exist_ok=True)

    with args.output.open("wb") as raw:
        with gzip.GzipFile(filename="", mode="wb", fileobj=raw, mtime=args.epoch) as compressed:
            with tarfile.open(fileobj=compressed, mode="w", format=tarfile.PAX_FORMAT) as archive:
                for path in paths:
                    arcname = "." if path == source else "./" + path.relative_to(source).as_posix()
                    info = archive.gettarinfo(str(path), arcname=arcname)
                    info.uid = 0
                    info.gid = 0
                    info.uname = "root"
                    info.gname = "root"
                    info.mtime = args.epoch
                    info.pax_headers = {}
                    if path.is_file():
                        with path.open("rb") as handle:
                            archive.addfile(info, handle)
                    else:
                        archive.addfile(info)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
