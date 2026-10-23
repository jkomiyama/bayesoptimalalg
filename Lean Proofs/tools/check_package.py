#!/usr/bin/env python3
"""Compare delivered files with SHA256SUMS.txt; this does not verify proofs."""
from __future__ import annotations

import argparse
import hashlib
from pathlib import Path, PurePosixPath
import sys


ROOT = Path(__file__).resolve().parent.parent
REQUIRED = {
    "README.txt", "verify.py", "lean-toolchain", "lakefile.toml", "lake-manifest.json",
    "JunpeiVersionLean.lean", "BundleAudit.lean", "declarations.json",
    "manuscript/junpei_version.pdf", "REPORT.txt", "verification_results.json",
    "docs/PROOF_MAP.txt", "docs/PROOF_MAP.html", "docs/proof_map.tsv", "docs/DECLARATIONS.html",
    "docs/declarations.tsv", "docs/MODULES.txt", "tools/check_package.py",
    "REVISION_NOTES.txt", "checks/GeneralOptimalityAudit.lean",
    "Junpei/GeneralOptimalityHistory.lean", "Junpei/GeneralOptimality.lean",
    "Junpei/GeneralOptimalityProper.lean", "Junpei/GeneralOptimalityFinite.lean",
    "Junpei/Footnote14Characterization.lean", "checks/Footnote14CharacterizationAudit.lean",
}


def main() -> int:
    argparse.ArgumentParser(description=__doc__).parse_args()
    manifest = ROOT / "SHA256SUMS.txt"
    if not manifest.is_file():
        print("SHA256SUMS.txt is missing. This is not a complete release archive.")
        return 1
    failures: list[str] = []
    checked = 0
    seen: set[str] = set()
    for number, line in enumerate(manifest.read_text(encoding="utf-8").splitlines(), 1):
        digest, separator, name = line.partition("  ")
        parts = PurePosixPath(name)
        if (not separator or len(digest) != 64
                or any(c not in "0123456789abcdef" for c in digest)
                or not name or "\\" in name or ":" in name
                or parts.is_absolute() or ".." in parts.parts or name in seen):
            failures.append(f"Invalid manifest entry at line {number}")
            continue
        seen.add(name)
        path = ROOT / parts
        if not path.resolve().is_relative_to(ROOT):
            failures.append(f"Path leaves the package: {name}")
            continue
        if not path.is_file():
            failures.append(f"Missing: {name}")
            continue
        checked += 1
        actual = hashlib.sha256(path.read_bytes()).hexdigest()
        if actual != digest:
            failures.append(f"Changed: {name}")
    for name in sorted(REQUIRED - seen):
        failures.append(f"Required entry omitted from manifest: {name}")
    for failure in failures:
        print(failure)
    if failures:
        print(f"FAIL: {len(failures)} integrity issue(s); {checked} files read.")
        print("Verification logs/reports change on a rerun. See README.txt.")
        return 1
    print(f"PASS: all {checked} delivered-file checksums match.")
    print("This establishes file identity only. Run verify.py for Lean verification.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
