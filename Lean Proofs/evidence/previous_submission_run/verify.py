"""Compile the submitted Lean proofs and record reproducible verification evidence.

Python performs administrative checks only. All mathematical verification is
performed by Lean. Requires Python 3.10+ and Lake (normally installed by elan).
Run `python verify.py --help` for options; help does not modify any files.
"""
from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import re
import shutil
import subprocess
import sys


ROOT = Path(__file__).resolve().parent
LOGS = ROOT / "logs"
PACKAGE = "junpei_version_lean"
EXPECTED_TOOLCHAIN = "leanprover/lean4:v4.24.0"
EXPECTED_MATHLIB_REV = "f897ebcf72cd16f89ab4577d0c826cd14afaafc7"
EXPECTED_MODULES = 151
EXPECTED_DECLARATIONS = 1470
EXPECTED_SUPPLEMENTARY = 10
ALLOWED_AXIOMS = {"propext", "Classical.choice", "Quot.sound"}
CONTEXTUAL_LIMITS = [
    "General non-Gaussian proper-prior Bellman assertion with finite first moment",
    "Remark 1 cited SR, change-of-measure, and Bayesian-rate extensions",
    "Arbitrary history-minimizer to posterior-state Markovization beyond Definition 3",
    "Compression of arbitrary auxiliary Gaussian blocks into one scalar randomizer",
]


def arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--fresh", action="store_true",
                        help="clean this package's generated outputs before rebuilding; "
                             "leave dependency caches intact")
    parser.add_argument("--lake", default="lake", metavar="EXECUTABLE",
                        help="Lake executable name or path (default: lake on PATH)")
    parser.add_argument("--jobs", type=int, default=2, metavar="N",
                        help="concurrent read-only Lean audits after the build (default: 2)")
    args = parser.parse_args()
    if args.jobs < 1:
        parser.error("--jobs must be a positive integer")
    return args


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def relative(path: Path) -> str:
    return path.relative_to(ROOT).as_posix()


def strip_comments(source: str) -> str:
    """Remove nested Lean comments for an administrative source scan."""
    result: list[str] = []
    index = depth = 0
    while index < len(source):
        pair = source[index:index + 2]
        if pair == "/-":
            depth += 1
            result.append("  ")
            index += 2
        elif depth and pair == "-/":
            depth -= 1
            result.append("  ")
            index += 2
        elif not depth and pair == "--":
            newline = source.find("\n", index)
            index = len(source) if newline < 0 else newline
        else:
            result.append(source[index] if not depth or source[index] == "\n" else " ")
            index += 1
    return "".join(result)


def bundle(names: list[str]) -> str:
    if len(names) == 1:
        return "(@" + names[0] + ")"
    split = len(names) // 2
    return "(PProd.mk " + bundle(names[:split]) + " " + bundle(names[split:]) + ")"


def proof_sources() -> list[Path]:
    return (sorted((ROOT / "Draft0").rglob("*.lean"))
            + sorted((ROOT / "Junpei").rglob("*.lean"))
            + [ROOT / "JunpeiVersionLean.lean", ROOT / "MainTheoremAudit.lean",
               ROOT / "ClaimsAudit.lean", ROOT / "AdditionalProofAudit.lean",
               ROOT / "EndToEndAudit.lean"]
            + sorted((ROOT / "checks").rglob("*.lean")))


def run(lake: str, label: str, lake_args: list[str]) -> dict:
    command = [lake, *lake_args]
    started = datetime.now(timezone.utc)
    with (LOGS / (label + ".log")).open("w", encoding="utf-8") as log:
        try:
            process = subprocess.Popen(
                command, cwd=ROOT, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                text=True, encoding="utf-8", errors="replace", shell=False)
            assert process.stdout is not None
            for line in process.stdout:
                log.write(line)
                log.flush()
            exit_code = process.wait()
        except OSError as error:
            log.write(f"Could not launch Lake: {error}\n")
            exit_code = 127
    elapsed = (datetime.now(timezone.utc) - started).total_seconds()
    print(f"{label}: exit code {exit_code} ({elapsed:.1f}s)", flush=True)
    return {"label": label, "command": ["lake", *lake_args],
            "lake_executable_name": Path(lake).name, "exit_code": exit_code,
            "log": f"logs/{label}.log", "duration_seconds": round(elapsed, 3)}


def log_text(label: str, results: list[dict]) -> str:
    # Do not accidentally report an older successful log after a failed run.
    if not any(result["label"] == label for result in results):
        return ""
    return (LOGS / (label + ".log")).read_text(encoding="utf-8")


def main() -> int:
    args = arguments()
    LOGS.mkdir(exist_ok=True)
    started = datetime.now(timezone.utc).isoformat()
    issues: list[str] = []
    declarations: list[str] = []
    module_counts: dict[str, int] = {}
    hashes: dict[str, str] = {}
    results: list[dict] = []
    sources = proof_sources()
    supplementary = sorted((ROOT / "checks").rglob("*.lean"))
    config_files = [ROOT / name for name in
                    ("lean-toolchain", "lakefile.toml", "lake-manifest.json", "verify.py")]
    paper = ROOT / "manuscript" / "junpei_version.pdf"
    for path in sources + config_files + [paper]:
        try:
            hashes[relative(path)] = sha256(path)
        except OSError as error:
            issues.append(f"Cannot read {relative(path)}: {error}")

    for path in sources:
        if relative(path) not in hashes:
            continue
        cleaned = strip_comments(path.read_text(encoding="utf-8"))
        name = relative(path)
        if re.search(r"\b(sorry|admit|sorryAx|native_decide)\b", cleaned):
            issues.append("Proof shortcut in " + name)
        if re.search(r"^\s*(?:(?:private|protected)\s+)?(?:axiom|opaque|constant)\s",
                     cleaned, re.M):
            issues.append("Unproved declaration in " + name)
        namespace = re.search(r"^namespace\s+([\w.]+)", cleaned, re.M)
        prefix = namespace.group(1) + "." if namespace else ""
        names = [prefix + theorem for theorem in re.findall(
            r"^\s*(?:@\[[^\]]*\]\s*)*(?:theorem|lemma)\s+([\w']+)", cleaned, re.M)]
        declarations.extend(names)
        module_counts[name] = len(names)
    declarations = sorted(set(declarations))
    module_count = sum(relative(path).startswith(("Draft0/", "Junpei/")) for path in sources)
    if module_count != EXPECTED_MODULES:
        issues.append(f"Expected {EXPECTED_MODULES} proof modules, found {module_count}")
    if len(declarations) != EXPECTED_DECLARATIONS:
        issues.append(f"Expected {EXPECTED_DECLARATIONS} public theorem declarations, "
                      f"found {len(declarations)}")
    if len(supplementary) != EXPECTED_SUPPLEMENTARY:
        issues.append(f"Expected {EXPECTED_SUPPLEMENTARY} supplementary audits, "
                      f"found {len(supplementary)}")

    pins: dict = {}
    try:
        toolchain = (ROOT / "lean-toolchain").read_text(encoding="utf-8").strip()
        manifest = json.loads((ROOT / "lake-manifest.json").read_text(encoding="utf-8"))
        lakefile = (ROOT / "lakefile.toml").read_text(encoding="utf-8")
        package_name = re.search(r'^name\s*=\s*"([^"]+)"', lakefile, re.M)
        mathlib = next(p for p in manifest["packages"] if p["name"] == "mathlib")
        pins = {"lean_toolchain": toolchain, "mathlib_revision": mathlib["rev"],
                "dependency_revisions": {p["name"]: p["rev"] for p in manifest["packages"]}}
        if toolchain != EXPECTED_TOOLCHAIN:
            issues.append(f"Unexpected toolchain pin: {toolchain}")
        if mathlib["rev"] != EXPECTED_MATHLIB_REV:
            issues.append(f"Unexpected mathlib revision: {mathlib['rev']}")
        if not package_name or package_name.group(1) != PACKAGE or manifest["name"] != PACKAGE:
            issues.append("Package name differs from the submitted package")
    except (OSError, ValueError, KeyError, StopIteration) as error:
        issues.append(f"Cannot inspect pinned project configuration: {error}")

    generated_hashes: dict[str, str] = {}
    if declarations:
        (ROOT / "declarations.json").write_text(json.dumps(declarations, indent=2), encoding="utf-8")
        (ROOT / "BundleAudit.lean").write_text(
            "import JunpeiVersionLean\n\nset_option maxRecDepth 4096\n\n"
            "noncomputable def allPublicTheorems :=\n  " + bundle(declarations)
            + "\n\n#print axioms allPublicTheorems\n", encoding="utf-8")
        generated_hashes = {name: sha256(ROOT / name)
                            for name in ("BundleAudit.lean", "declarations.json")}

    lake = shutil.which(args.lake)
    if lake is None:
        issues.append("Lake was not found. Install elan, put lake on PATH, or pass "
                      "--lake /path/to/lake. See README.txt for setup instructions.")
    if lake is not None and not issues:
        results.append(run(lake, "version", ["env", "lean", "--version"]))
        if not re.search(r"Lean \(version 4\.24\.0(?:[ ,)]|$)", log_text("version", results)):
            issues.append("The compiler did not report the pinned Lean version 4.24.0")
        if args.fresh and not issues and results[-1]["exit_code"] == 0:
            # Explicit package selection leaves all dependency build caches intact.
            # Resolve junctions/symlinks before letting Lake remove this build tree.
            build_directory = (ROOT / ".lake" / "build").resolve()
            if not build_directory.is_relative_to(ROOT) or build_directory == ROOT:
                issues.append(f"Refusing to clean outside this project's build tree: {build_directory}")
            else:
                results.append(run(lake, "clean", ["clean", PACKAGE]))
        if not issues and all(result["exit_code"] == 0 for result in results):
            results.append(run(lake, "build", ["--rehash", "build"]))
        if results[-1]["label"] == "build" and results[-1]["exit_code"] == 0:
            audits = [("claims", "ClaimsAudit.lean"),
                      ("main_theorem", "MainTheoremAudit.lean"),
                      ("additional_proofs", "AdditionalProofAudit.lean"),
                      ("end_to_end", "EndToEndAudit.lean"),
                      ("axioms", "BundleAudit.lean")]
            audits += [("supplementary_" + path.stem, relative(path)) for path in supplementary]
            with ThreadPoolExecutor(max_workers=args.jobs) as pool:
                results += list(pool.map(
                    lambda audit: run(lake, audit[0], ["env", "lean", audit[1]]), audits))
            for label, _ in audits:
                output = log_text(label, results)
                groups = re.findall(r"depends on axioms:\s*\[([^\]]*)\]", output)
                if label in {"main_theorem", "additional_proofs", "end_to_end", "axioms"} and not groups:
                    issues.append("No axiom output in " + label)
                for group in groups:
                    unexpected = {x.strip() for x in group.split(",") if x.strip()} - ALLOWED_AXIOMS
                    if unexpected:
                        issues.append(f"Unexpected axioms in {label}: {sorted(unexpected)}")
                if "sorryAx" in output:
                    issues.append("Axiom audit contains sorryAx: " + label)

    for name, digest in {**hashes, **generated_hashes}.items():
        try:
            if sha256(ROOT / name) != digest:
                issues.append("Input changed during verification: " + name)
        except OSError:
            issues.append("Input disappeared during verification: " + name)
    if set(proof_sources()) != set(sources):
        issues.append("Proof-source inventory changed during verification")

    expected_runs = 7 + len(supplementary) + int(args.fresh)
    ok = (len(results) == expected_runs
          and all(result["exit_code"] == 0 for result in results) and not issues)
    full_rebuild = args.fresh and ok
    summary = {
        "verification_started_utc": started,
        "verification_time_utc": datetime.now(timezone.utc).isoformat(),
        "verification_ok": ok,
        "fresh_rebuild_requested": args.fresh,
        "fresh_rebuild": full_rebuild,
        "pins": pins,
        "input_paper_sha256": hashes.get(relative(paper)),
        "source_sha256": {relative(path): hashes[relative(path)]
                          for path in sources if relative(path) in hashes},
        "configuration_and_verifier_sha256": {relative(path): hashes[relative(path)]
                                               for path in config_files if relative(path) in hashes},
        "generated_artifact_sha256": generated_hashes,
        "compiler_runs": results,
        "expected_command_count": expected_runs,
        "supplementary_audit_files": [relative(path) for path in supplementary],
        "proof_module_count": module_count,
        "inventoried_public_theorem_declarations": len(declarations),
        "accepted_public_theorem_declarations": len(declarations) if ok else 0,
        "module_declaration_counts": module_counts,
        "audit_issues": issues,
        "all_18_numbered_proved_results_verified": ok,
        "main_theorem_verified": ok,
        "additional_constructions_verified": ok,
        "actual_risk_end_to_end_verified": ok,
        "initialization_only_risk_end_to_end_verified": ok,
        "footnote14_necessary_scale_refuted": ok,
        "all_manuscript_assertions_certified_correct": False,
        "complete_literal_manuscript_formalization": False,
        "contextual_extensions_not_formalized": CONTEXTUAL_LIMITS,
        "manuscript_verdict": (
            "The formal counterparts of all 18 numbered proved results passed. "
            "Lean also accepted a counterexample to footnote 14's claimed necessary "
            "Gaussian gap scale for a positive polynomial tail lower bound. "
            "Some contextual assertions remain outside the formalization."
            if ok else "This run did not certify the proof bundle. See audit_issues and compiler logs."),
    }
    (ROOT / "verification_results.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    version_output = log_text("version", results).strip() or "Version command was not run."
    build_output = log_text("build", results)
    build_result = next((line for line in reversed(build_output.splitlines())
                         if "Build completed successfully" in line), "Build did not complete successfully.")
    axiom_output = log_text("axioms", results).strip() or "Aggregate axiom audit was not run."
    run_lines = "\n".join(f"  {r['label']}: exit code {r['exit_code']}" for r in results)
    if not run_lines:
        run_lines = "  No compiler commands ran; see audit issues below."
    report = f"""JUNPEI LEAN SUBMISSION: VERIFICATION REPORT

Result: {'PASS' if ok else 'FAIL'}.
Verification completed (UTC): {summary['verification_time_utc']}.
Fresh rebuild of all local proof modules completed: {str(full_rebuild).lower()}.
Accepted public theorem/lemma declarations: {summary['accepted_public_theorem_declarations']}.
Proof modules inventoried: {module_count}.
Entire manuscript certified correct: false.

{summary['manuscript_verdict']}

On PASS, this run verifies the formal counterparts of Theorems 1-2,
Lemmas 3-8 and 10-19, their constructions, and the end-to-end main result
from initialization-only actual-risk minimization for posterior-state policies.
The reduction to an almost-sure optimal repair, transfer to fixed-parameter
Gaussian laws, and link to the original frequentist regret are formal proofs.
Conjecture 9 remains open and is not an axiom or a claimed proved result.

Footnote 14: {'REFUTED by a Lean-checked counterexample.' if ok else 'NOT certified by this run.'}
The counterexample uses gap sqrt(log T) / sqrt(T^3). Its Gaussian tail has
a positive polynomial lower bound for integer T >= 5, although the gap is
not O(T^(-3/2)). A successful build accepts the counterexample, not the false
necessity claim. The supplied PDF has not been changed.

Selected raw Lean output:

{version_output}
{build_result}
{axiom_output}

Recorded compiler commands:

{run_lines}

Audit issues: {issues}.

The generated aggregate references every inventoried public theorem/lemma.
On PASS, its axiom dependencies are confined to propext, Classical.choice,
and Quot.sound. The source scan also rejects proof placeholders, custom
unproved declarations, and native_decide shortcuts. Python inventories files,
invokes Lean, and records results; Lean checks the mathematical proof terms.

Lean verifies explicit formal translations, not PDF sentences directly.
See docs/PROOF_MAP.txt, COVERAGE.txt, and ALL_ARGUMENTS_STATUS.txt for paper-to-code
correspondence, exact policy hypotheses, and all recorded scope limitations.
Contextual assertions not formalized include:
{chr(10).join('  - ' + item for item in CONTEXTUAL_LIMITS)}
Literature history, numerical illustrations, and explicitly open proposals
are not treated as proved theorems. The Lemma 19 fixed-tie negative certificate
refutes a stronger interpretation, not the manuscript's oriented comparison.

Reproduce from this directory:
  python verify.py --fresh

Setup instructions: README.txt. No sibling project is required.
verification_results.json records source, configuration, verifier, generated
artifact, and PDF hashes, command exit codes, counts, and scope flags.
logs/ contains raw outputs. Hashes and the source inventory are checked
again at the end of the run to detect changes made during verification.
"""
    (ROOT / "REPORT.txt").write_text(report, encoding="utf-8")
    print(json.dumps({key: summary[key] for key in
                      ("accepted_public_theorem_declarations", "audit_issues", "verification_ok")},
                     indent=2), flush=True)
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
