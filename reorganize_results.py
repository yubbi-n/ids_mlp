from __future__ import annotations
"""
One-off migration: move the flat files in results/ (e.g.
CICIoT2023_binary_rf_top5_summary.json) into a
<dataset>/<scope>/<method>/ subfolder layout (e.g.
results/CICIoT2023/binary/rf_top5/CICIoT2023_binary_rf_top5_summary.json).
Filenames are left unchanged -- only moved -- so every existing script
(summarize_results.py, this repo's own naming convention) keeps working
unmodified; summarize_results.py already searches recursively.

Safe by default: prints a dry-run preview (source -> destination) and
moves NOTHING unless --execute is passed. Run the dry-run first, read it,
then re-run with --execute.

Usage:
    python reorganize_results.py --results_dir results            # preview only
    python reorganize_results.py --results_dir results --execute  # actually move
"""

import argparse
import os
import re
import shutil

# Order matters: more specific patterns first.
RULES: list[tuple[re.Pattern, str]] = [
    # cross-dataset zero-shot runs
    (re.compile(r"^CICIoT2023_to_CICIDS2017_zeroshot"),
     "cross_dataset/CICIoT2023_to_CICIDS2017_zeroshot"),

    # CICIoT2023 binary
    (re.compile(r"^CICIoT2023_binary_rf_top(\d+)"), r"CICIoT2023/binary/rf_top\1"),
    (re.compile(r"^CICIoT2023_binary_shap_top(\d+)"), r"CICIoT2023/binary/shap_top\1"),
    (re.compile(r"^CICIoT2023_binary_baseline"), "CICIoT2023/binary/baseline"),
    (re.compile(r"^CICIoT2023_binary_(feature_ranking|feature_importance|run_settings|timing)"),
     "CICIoT2023/binary/feature_selection"),

    # CICIoT2023 6-class
    (re.compile(r"^CICIoT2023_rf_top(\d+)"), r"CICIoT2023/6class/rf_top\1"),
    (re.compile(r"^CICIoT2023_shap_top(\d+)"), r"CICIoT2023/6class/shap_top\1"),
    (re.compile(r"^CICIoT2023_lrsched"), "CICIoT2023/6class/lrsched"),
    (re.compile(r"^CICIoT2023_baseline"), "CICIoT2023/6class/baseline"),
    (re.compile(r"^CICIoT2023_(feature_ranking|feature_importance|run_settings|timing)"),
     "CICIoT2023/6class/feature_selection"),

    # CICIDS2017 binary
    (re.compile(r"^CICIDS2017_binary_rf_top(\d+)"), r"CICIDS2017/binary/rf_top\1"),
    (re.compile(r"^CICIDS2017_binary_shap_top(\d+)"), r"CICIDS2017/binary/shap_top\1"),
    (re.compile(r"^CICIDS2017_binary_baseline"), "CICIDS2017/binary/baseline"),
    (re.compile(r"^CICIDS2017_binary_(feature_ranking|feature_importance|run_settings|timing)"),
     "CICIDS2017/binary/feature_selection"),

    # UNSW-NB15 (3-class: Benign/DoS/Recon)
    (re.compile(r"^UNSW_NB15_rf_top(\d+)"), r"UNSW_NB15/3class/rf_top\1"),
    (re.compile(r"^UNSW_NB15_shap_top(\d+)"), r"UNSW_NB15/3class/shap_top\1"),
    (re.compile(r"^UNSW_NB15_baseline"), "UNSW_NB15/3class/baseline"),
    (re.compile(r"^UNSW_NB15_(feature_ranking|feature_importance|run_settings|timing)"),
     "UNSW_NB15/3class/feature_selection"),
]

# Aggregate comparison CSVs (summarize_results.py --out_csv output) stay at
# the results/ root -- they summarize ACROSS groups, so nesting them into
# any one group's folder would be misleading.
KEEP_AT_ROOT = re.compile(r"^(binary_comparison|unsw_comparison|comparison)\.csv$")


def classify(filename: str) -> str | None:
    """Return the destination subfolder (relative to results_dir) for a
    filename, or None if it should stay at the root / doesn't match."""
    if KEEP_AT_ROOT.match(filename):
        return None
    for pattern, dest_template in RULES:
        m = pattern.match(filename)
        if m:
            return m.expand(dest_template)
    return None


def plan_moves(results_dir: str) -> list[tuple[str, str]]:
    moves = []
    for entry in sorted(os.listdir(results_dir)):
        src = os.path.join(results_dir, entry)
        if not os.path.isfile(src):
            continue  # skip subfolders already created by a prior run
        dest_subdir = classify(entry)
        if dest_subdir is None:
            continue
        dest = os.path.join(results_dir, dest_subdir, entry)
        moves.append((src, dest))
    return moves


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--results_dir", default="results")
    parser.add_argument("--execute", action="store_true",
                         help="Actually move files. Without this flag, only previews.")
    args = parser.parse_args()

    moves = plan_moves(args.results_dir)
    if not moves:
        print(f"Nothing to move in {args.results_dir} (already organized, or no matching files).")
    else:
        print(f"{'MOVING' if args.execute else 'DRY RUN (pass --execute to actually move)'}:")
        for src, dest in moves:
            print(f"  {src}  ->  {dest}")
        unmatched = [
            e for e in sorted(os.listdir(args.results_dir))
            if os.path.isfile(os.path.join(args.results_dir, e)) and classify(e) is None
            and not KEEP_AT_ROOT.match(e)
        ]
        if unmatched:
            print("\nUnmatched files (left at root, review manually):")
            for e in unmatched:
                print(f"  {e}")

        if args.execute:
            for src, dest in moves:
                os.makedirs(os.path.dirname(dest), exist_ok=True)
                shutil.move(src, dest)
            print(f"\nMoved {len(moves)} file(s).")
