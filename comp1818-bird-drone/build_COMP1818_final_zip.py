"""Package a reproducible COMP1818 Bird-vs-Drone Python coursework project.

Put this script in the project root (beside dashboard.py) and run:
    python build_COMP1818_final_zip.py

Only existing files are included. The script never creates missing evidence or edits
original files. Read the generated PACKAGE_REVIEW.txt before uploading anything.
"""
from __future__ import annotations

from pathlib import Path
from collections import Counter
import datetime as dt
import json
import re
import zipfile

ROOT = Path(__file__).resolve().parent
DEST = ROOT.parent / "COMP1818_Final_Python_Code.zip"
PREFIX = "COMP1818_Bird_Drone_Final"

# Exclude datasets, enormous videos, machine-specific environments, cache and secrets.
EXCLUDED_DIRS = {
    ".venv", "venv", "env", "__pycache__", ".git", ".pytest_cache",
    ".mypy_cache", ".ruff_cache", ".idea", ".vscode", "node_modules",
    "datasets", "data", "dashboard_uploads", "CSV_Evidence", "labels_gt",
    "clips", "videos", "output", "outputs", "dist", "build",
}
EXCLUDED_FILENAMES = {"desktop.ini", "thumbs.db", ".ds_store", ".env"}
EXCLUDED_SUFFIXES = {
    ".mp4", ".mov", ".avi", ".mkv", ".zip", ".ptl", ".onnx",
    ".pyc", ".pyo", ".log", ".cache", ".tmp", ".db", ".sqlite",
}
SENSITIVE_PATTERNS = (".env", "token", "secret", "password", "private_key", "credential", "id_rsa")
SOURCE_SUFFIXES = {".py", ".yaml", ".yml", ".toml", ".ini", ".json", ".md", ".txt", ".cfg"}
EVIDENCE_SUFFIXES = {".csv", ".json", ".png", ".jpg", ".jpeg"}
ROOT_SOURCE_FILES = {"requirements.txt", "pyproject.toml", "setup.cfg", "README.md", "EXPERIMENTS.md", "data.yaml"}
MAX_FILE_MB = 35


def sensitive(name: str) -> bool:
    n = name.lower()
    return n == ".env" or any(x in n for x in SENSITIVE_PATTERNS)


def select_files():
    selected: dict[str, Path] = {}
    skipped = Counter()
    for file in ROOT.rglob("*"):
        if not file.is_file():
            continue
        rel = file.relative_to(ROOT)
        parts = rel.parts
        lowparts = [p.lower() for p in parts]
        if any(p in EXCLUDED_DIRS for p in lowparts[:-1]):
            skipped["excluded folder"] += 1
            continue
        if any(sensitive(p) for p in parts):
            skipped["potential secrets"] += 1
            continue
        if file.name.lower() in EXCLUDED_FILENAMES or file.suffix.lower() in EXCLUDED_SUFFIXES:
            skipped["excluded type"] += 1
            continue
        if file.stat().st_size > MAX_FILE_MB * 1024 * 1024:
            skipped["large file"] += 1
            continue
        if rel.parts[0] == "models":
            # Include only the two named checkpoint weights in the reproducibility package.
            if file.name in {"best_10epochs.pt", "best_50epochs.pt"}:
                selected[rel.as_posix()] = file
            else:
                skipped["other model"] += 1
            continue
        if file.suffix.lower() == ".pt":
            skipped["unselected checkpoint"] += 1
            continue
        if rel.parts[0] in {"results", "runs", "dashboard_runs", "tests"}:
            if file.suffix.lower() in EVIDENCE_SUFFIXES or (rel.parts[0] == "tests" and file.suffix == ".py"):
                selected[rel.as_posix()] = file
            else:
                skipped["extra generated artifact"] += 1
            continue
        if rel.parts[0].lower() in {"figures", "report", "reports"}:
            if file.suffix.lower() in {".tex", ".pdf", ".bib", ".png", ".jpg", ".jpeg", ".bst", ".cls"}:
                selected[rel.as_posix()] = file
            continue
        # Include source/configuration/documentation at root or in arbitrary source folders.
        if file.suffix.lower() in SOURCE_SUFFIXES or file.name in ROOT_SOURCE_FILES:
            selected[rel.as_posix()] = file
        else:
            skipped["non-source root file"] += 1
    return selected, skipped


def main():
    if not (ROOT / "dashboard.py").exists() and not (ROOT / "evaluate.py").exists():
        raise SystemExit("ERROR: Put this script in comp1818-bird-drone beside dashboard.py/evaluate.py and run it there.")
    selected, skipped = select_files()
    expected = ["dashboard.py", "evaluate.py", "eval_tracking.py", "analyze_motion.py", "compare_trackers.py", "requirements.txt", "data.yaml", "models/best_10epochs.pt", "models/best_50epochs.pt"]
    missing = [x for x in expected if x not in selected]
    report_present = any(x.lower().endswith(".pdf") for x in selected)
    user_note = """COMP1818 — FINAL PYTHON PROJECT PACKAGE

This ZIP contains the Python source/config files and existing supporting results in the project folder.
It deliberately excludes the AOD-4 dataset, virtual environment, raw videos and temporary outputs.
Original code and evidence were copied without alteration. Check this package before submission.

QUICK START (on Windows, from extracted project directory):
  python -m venv .venv
  .\\.venv\\Scripts\\Activate.ps1
  python -m pip install -r requirements.txt
  python dashboard.py

DATASET:
  The AOD-4 dataset is NOT bundled. Install it separately and adjust data.yaml
  to point to the correct local location, ensuring images/test and labels/test align.
  Do not use evaluation-only data.yaml to retrain the detector.

MODEL CHECKPOINTS:
  models/best_10epochs.pt and models/best_50epochs.pt are included only if
  both exist. Check the contents of PACKAGE_REVIEW.txt for missing files.

RESEARCH CAUTIONS:
  Track IDs are not verified physical object counts; no independent tracking
  ground truth is included. Check 10/50 epoch provenance and test-set independence.
  Figures in the report should be originals generated programmatically, not screenshots.

FINAL SUBMISSION:
  Check the official Moodle rules, file-size limit, code URL, and LO3 requirements.
  The final LaTeX PDF is not automatically included unless already stored in
  a report/ or figures/ folder inside your local project.
"""
    report = {
        "created_utc": dt.datetime.now(dt.timezone.utc).isoformat(timespec="seconds"),
        "source_folder": ROOT.name,
        "included_file_count": len(selected),
        "included_files": sorted(selected),
        "missing_recommended_files": missing,
        "report_pdf_included": report_present,
        "excluded_counts": dict(skipped),
    }
    with zipfile.ZipFile(DEST, "w", zipfile.ZIP_DEFLATED, compresslevel=6) as z:
        for name, path in sorted(selected.items()):
            z.write(path, arcname=f"{PREFIX}/{name}")
        z.writestr(f"{PREFIX}/PACKAGE_README.txt", user_note)
        z.writestr(f"{PREFIX}/PACKAGE_REVIEW.json", json.dumps(report, indent=2))
    print("Created:", DEST)
    print("Included files:", len(selected))
    print("Contains report PDF:", report_present)
    print("Missing recommended files:", ", ".join(missing) if missing else "none")
    print("Excluded:", dict(skipped))
    print("IMPORTANT: Review PACKAGE_REVIEW.json and confirm no sensitive files are included.")


if __name__ == "__main__":
    main()
