#!/usr/bin/env python3
"""Regenerate release checksums and a source/evidence ZIP, excluding build caches.

Administrative packaging only: run verify.py separately to check the proofs.
The archive is written next to the project, with one top-level project folder.
An existing archive is never overwritten unless --replace is explicitly given.
"""
from __future__ import annotations

import argparse
import hashlib
import os
from pathlib import Path
import stat
import zipfile


ROOT = Path(__file__).resolve().parent.parent
EXCLUDED_DIRS = {".lake", ".git", "__pycache__"}
EXCLUDED_FILES = {".DS_Store", "Thumbs.db"}


def delivered_files() -> list[Path]:
    paths: list[Path] = []
    for directory, folders, files in os.walk(ROOT, topdown=True, followlinks=False):
        folders[:] = sorted(name for name in folders if name not in EXCLUDED_DIRS)
        for name in folders + files:
            path = Path(directory) / name
            attributes = getattr(path.lstat(), "st_file_attributes", 0)
            if path.is_symlink() or attributes & getattr(stat, "FILE_ATTRIBUTE_REPARSE_POINT", 0):
                raise RuntimeError(f"Refusing to package a link/reparse point: {path.relative_to(ROOT)}")
        for name in sorted(files):
            if name in EXCLUDED_FILES or name.endswith(".pyc"):
                continue
            paths.append(Path(directory) / name)
    return sorted(paths)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--replace", action="store_true", help="replace an existing release ZIP")
    parser.add_argument("--name", default=ROOT.name + ".zip", metavar="BASENAME.zip",
                        help="archive basename, written next to the project")
    args = parser.parse_args()
    if (not args.name.endswith(".zip") or args.name in {".zip", "..zip"}
            or any(c in args.name for c in ("/", "\\", ":"))):
        parser.error("--name must be a ZIP basename, without a directory or drive")
    archive = ROOT.parent / args.name
    if archive.exists() and not args.replace:
        parser.error(f"Archive already exists: {archive.name}; pass --replace to regenerate it")
    files = [path for path in delivered_files() if path.name != "SHA256SUMS.txt"]
    checksum = ROOT / "SHA256SUMS.txt"
    checksum.write_text("".join(
        f"{hashlib.sha256(path.read_bytes()).hexdigest()}  {path.relative_to(ROOT).as_posix()}\n"
        for path in files), encoding="utf-8", newline="\n")
    files.append(checksum)
    mode = "w" if args.replace else "x"
    with zipfile.ZipFile(archive, mode, compression=zipfile.ZIP_DEFLATED, compresslevel=9) as package:
        for path in sorted(files):
            package.write(path, (Path(ROOT.name) / path.relative_to(ROOT)).as_posix())
    # Check the archive's CRCs and exact file bytes after writing.
    with zipfile.ZipFile(archive) as package:
        bad = package.testzip()
        if bad:
            raise RuntimeError(f"Archive CRC failure: {bad}")
        for path in files:
            entry = (Path(ROOT.name) / path.relative_to(ROOT)).as_posix()
            if package.read(entry) != path.read_bytes():
                raise RuntimeError(f"Archive differs from source: {entry}")
    print(f"Created {archive.name}: {len(files)} files, {archive.stat().st_size:,} bytes.")
    print(f"SHA-256: {hashlib.sha256(archive.read_bytes()).hexdigest()}")
    print("Archive CRCs and every delivered file's bytes match the package directory.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
