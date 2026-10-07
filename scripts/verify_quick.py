#!/usr/bin/env python3
"""Run the documented quick checks without changing archived reports.

Existing verification programs run in a temporary copy of the repository.
The checks cover the stored witnesses, not exhaustive search coverage.
"""
from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import json
import os
from pathlib import Path
import platform
import shutil
import subprocess
import sys
import tempfile
import time

ROOT = Path(__file__).resolve().parents[1]
CHECKS = (
    ("47-qubit comparison code", "verify_jacinto_d3.py", ()),
    ("48-qubit distances and logical witness", "q48_min_logical_basis.py", ()),
    ("physical-sign formula", "verify_tpattern_paper_form.py", ()),
    ("signed-weight conditions", "verify_native_ladder.py", ()),
    ("family witnesses and printed table", "verify_family_table.py", ("--check",)),
)
DEPENDENCIES = ("numpy", "galois", "numba", "llvmlite", "z3-solver", "typing_extensions")


def source_digest() -> str:
    """Hash the programs and data used by the checks, including this runner."""
    paths = []
    for directory in ("gate2code", "scripts", "data"):
        paths.extend(
            path for path in (ROOT / directory).rglob("*")
            if path.is_file() and path.suffix in (".py", ".npz", ".json")
        )
    paths.extend(ROOT.glob("*.npz"))
    paths.extend((ROOT / "pyproject.toml", ROOT / "requirements-quick.txt"))
    digest = hashlib.sha256()
    for path in sorted(set(paths)):
        digest.update(path.relative_to(ROOT).as_posix().encode() + b"\0")
        digest.update(path.read_bytes())
        digest.update(b"\0")
    return digest.hexdigest()


def require(ok: bool, message: str) -> None:
    if not ok:
        raise RuntimeError(message)


def validate_reports(workspace: Path) -> None:
    """Check reported quantities that the individual programs only print."""
    def read(name: str) -> dict:
        return json.loads((workspace / "reports" / name).read_text())

    comparison = read("jacinto_d3_47_verification.json")
    require((comparison["rank_G"], comparison["rank_S"]) == (9, 6), "comparison ranks differ")
    require((comparison["dX"], comparison["dZ"]) == (16, 3), "comparison distances differ")
    require(comparison["ch_conditions_pass"] is True, "comparison CH conditions fail")
    pattern = comparison["native_no_correction"]
    require(pattern["found"] is False and pattern["status"] == "gf2_unsat", "comparison sign exclusion differs")
    contradiction = comparison["mod2_contradiction"]
    require(contradiction["coefficient_xor_is_zero"] is True and
            contradiction["right_hand_side_parity"] == 1, "comparison contradiction fails")

    q48 = read("q48_min_logical_basis.json")
    require(q48["ranks"] == {"G": 9, "S": 6}, "48-qubit ranks differ")
    require(q48["distances"] == {"d_X": 16, "d_Z": 3}, "48-qubit distances differ")
    witness = q48["z_logical_witness"]
    require(witness["weight"] == 3 and not any(witness["stabilizer_syndrome"]) and
            any(witness["logical_syndrome"]), "weight-three logical witness fails")

    ladder = read("q48_native_ladder.json")
    require(ladder["all_passed"] is True and ladder["codewords_checked"] == 512,
            "signed-weight verification fails")
    require(len(ladder["checks"]) == 14 and all(row["passed"] is True for row in ladder["checks"]),
            "signed-weight check count or status differs")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--report", type=Path, help="save a fresh verification receipt outside archived reports")
    args = parser.parse_args()
    if sys.flags.optimize:
        parser.error("run without -O or PYTHONOPTIMIZE: existing checks use assertions")
    report_path = args.report.resolve() if args.report else None
    archived = (ROOT / "reports").resolve()
    if report_path and (report_path == archived or archived in report_path.parents):
        parser.error("choose a report path outside reports/ to preserve archived evidence")
    receipt = {
        "scope": "Stored-code distances, physical signs, signed weights, comparison code, and family witnesses",
        "source_sha256": source_digest(),
        "python": platform.python_version(),
        "platform": platform.platform(),
        "dependencies": {name: importlib.metadata.version(name) for name in DEPENDENCIES},
        "checks": [],
        "not_rerun": ["exhaustive length-48 classification", "large lower-bound sweeps", "SAT search"],
        "all_passed": False,
    }
    try:
        with tempfile.TemporaryDirectory(prefix="gate2code-quick-") as temporary:
            workspace = Path(temporary) / "repository"
            shutil.copytree(ROOT, workspace, ignore=shutil.ignore_patterns(
                ".git", ".venv", "__pycache__", "*.pyc", "*.egg-info", ".quick-check", ".DS_Store"))
            for name in ("jacinto_d3_47_verification.json", "q48_min_logical_basis.json",
                         "q48_tpattern_paper_form.json", "q48_native_ladder.json"):
                (workspace / "reports" / name).unlink(missing_ok=True)
            environment = os.environ.copy()
            environment["PYTHONPATH"] = str(workspace)
            environment["NUMBA_CACHE_DIR"] = str(Path(temporary) / "numba-cache")
            for label, script, arguments in CHECKS:
                print(f"Checking {label} ...", flush=True)
                started = time.monotonic()
                result = subprocess.run(
                    [sys.executable, "-B", str(workspace / "scripts" / script), *arguments],
                    cwd=workspace, env=environment, text=True, stdout=subprocess.PIPE,
                    stderr=subprocess.STDOUT, timeout=300,
                )
                print(result.stdout, end="", flush=True)
                receipt["checks"].append({"name": label, "program": script,
                    "exit_code": result.returncode, "seconds": round(time.monotonic() - started, 3)})
                require(result.returncode == 0, f"{label} failed with exit code {result.returncode}")
            validate_reports(workspace)
        receipt["all_passed"] = True
    except (RuntimeError, subprocess.TimeoutExpired, OSError, KeyError, ValueError) as error:
        receipt["error"] = str(error)
        print(f"FAIL: {error}", file=sys.stderr)
    if report_path:
        report_path.parent.mkdir(parents=True, exist_ok=True)
        report_path.write_text(json.dumps(receipt, indent=2) + "\n")
    if receipt["all_passed"]:
        print("All five quick verification programs and report checks passed. Archived reports unchanged.")
        return 0
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
