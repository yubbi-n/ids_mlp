from __future__ import annotations
"""
Aggregate repeated-seed train.py runs into one mean+-std summary per
setting, so a difference between two settings (e.g. baseline vs rf_top5)
can be checked against run-to-run training-randomness noise instead of
read off a single run each.

Expects the layout train.py's --seed + --output_dir naturally produce:
    <results_dir>/.../<setting>/seed1/<name>_summary.json
    <results_dir>/.../<setting>/seed2/<name>_summary.json
    ...
(one seedN/ subfolder per run, all under the same <setting> parent dir).

Usage:
    python aggregate_seeds.py --results_dir results/CICIoT2023/binary
    python aggregate_seeds.py --results_dir results/CICIoT2023/binary --out_csv results/CICIoT2023/binary/seed_aggregate.csv
"""

import argparse
import glob
import json
import os
import statistics

import pandas as pd

NUMERIC_FIELDS = [
    "accuracy", "precision", "recall", "f1", "auc",
    "best_epoch", "cpu_latency_us_per_sample", "train_time_s", "test_time_s",
]


def find_seed_groups(results_dir: str) -> dict[str, list[str]]:
    """Map each setting directory -> its seed*/*_summary.json paths."""
    groups: dict[str, list[str]] = {}
    pattern = os.path.join(results_dir, "**", "seed*", "*_summary.json")
    for path in sorted(glob.glob(pattern, recursive=True)):
        setting_dir = os.path.dirname(os.path.dirname(path))
        groups.setdefault(setting_dir, []).append(path)
    return groups


def aggregate(paths: list[str]) -> dict:
    runs = []
    for p in paths:
        with open(p) as f:
            runs.append(json.load(f))

    agg = {"n_seeds": len(runs), "seeds": [r.get("seed") for r in runs]}
    for field in NUMERIC_FIELDS:
        values = [
            r[field] for r in runs
            if isinstance(r.get(field), (int, float)) and r[field] == r[field]  # drop missing/NaN
        ]
        if values:
            agg[f"{field}_mean"] = statistics.mean(values)
            agg[f"{field}_std"] = statistics.stdev(values) if len(values) > 1 else 0.0
            agg[f"{field}_min"] = min(values)
            agg[f"{field}_max"] = max(values)
    return agg


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--results_dir", default="results")
    ap.add_argument("--out_csv", default=None, help="Optionally also write the table to a CSV")
    args = ap.parse_args()

    groups = find_seed_groups(args.results_dir)
    if not groups:
        raise FileNotFoundError(
            f"No seed*/..._summary.json runs found under {args.results_dir} "
            f"(expected <setting>/seed1/..5/*_summary.json)"
        )

    rows = []
    for setting_dir, paths in sorted(groups.items()):
        agg = aggregate(paths)
        agg["setting"] = os.path.relpath(setting_dir, args.results_dir)
        with open(os.path.join(setting_dir, "seed_aggregate_summary.json"), "w") as f:
            json.dump(agg, f, indent=2)
        rows.append(agg)

    df = pd.DataFrame(rows).set_index("setting")
    show_cols = ["n_seeds"] + sorted(c for c in df.columns if c.endswith(("_mean", "_std")))
    df = df.sort_values("accuracy_mean", ascending=False)

    pd.set_option("display.width", 200)
    pd.set_option("display.max_columns", None)
    print(df[show_cols].to_string(float_format=lambda v: f"{v:.5f}" if isinstance(v, float) else str(v)))

    if args.out_csv:
        df.to_csv(args.out_csv)
        print(f"\nWrote {args.out_csv}")


if __name__ == "__main__":
    main()
