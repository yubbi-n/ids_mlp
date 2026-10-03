from __future__ import annotations
"""
Aggregate every <name>_summary.json (and matching <name>_per_class_metrics.csv)
in a results directory into one comparison table, sorted by accuracy --
so a batch of train.py runs (e.g. baseline + rf_top{5,10,15,20} +
shap_top{5,10,15,20}) can be compared at a glance instead of opening each
JSON individually.

Usage:
    python summarize_results.py --results_dir results
    python summarize_results.py --results_dir results --classes DoS MITM Recon
    python summarize_results.py --results_dir results --out_csv results/comparison.csv
"""

import argparse
import glob
import json
import os

import pandas as pd


def summarize(results_dir: str, highlight_classes=None) -> pd.DataFrame:
    rows = []
    for path in sorted(glob.glob(os.path.join(results_dir, "*_summary.json"))):
        name = os.path.basename(path)[: -len("_summary.json")]
        with open(path) as f:
            s = json.load(f)

        row = {
            "name": name,
            "accuracy": s.get("accuracy"),
            "precision": s.get("precision"),
            "recall": s.get("recall"),
            "f1": s.get("f1"),
            "auc": s.get("auc"),
            "params": s.get("params"),
            "best_epoch": s.get("best_epoch"),
            "lr_scheduler": s.get("lr_scheduler"),
        }

        per_class_path = os.path.join(results_dir, f"{name}_per_class_metrics.csv")
        if highlight_classes and os.path.exists(per_class_path):
            pc = pd.read_csv(per_class_path).set_index("class")
            for cls in highlight_classes:
                if cls in pc.index:
                    row[f"{cls}_recall"] = pc.loc[cls, "recall"]

        rows.append(row)

    if not rows:
        raise FileNotFoundError(f"No *_summary.json files found in {results_dir}")

    return pd.DataFrame(rows).sort_values("accuracy", ascending=False).reset_index(drop=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--results_dir", default="results")
    parser.add_argument(
        "--classes", nargs="*", default=["DoS", "MITM", "Recon"],
        help="Per-class recall columns to include alongside the overall metrics "
             "(pass --classes with no values to omit per-class columns)",
    )
    parser.add_argument("--out_csv", default=None, help="Optionally also write the table to a CSV")
    args = parser.parse_args()

    df = summarize(args.results_dir, args.classes)

    pd.set_option("display.width", 200)
    pd.set_option("display.max_columns", None)
    print(df.to_string(
        index=False,
        float_format=lambda v: f"{v:.4f}" if isinstance(v, float) else str(v),
    ))

    if args.out_csv:
        df.to_csv(args.out_csv, index=False)
        print(f"\nWrote {args.out_csv}")
