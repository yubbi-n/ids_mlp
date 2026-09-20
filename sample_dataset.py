from __future__ import annotations
"""
Memory-friendly, per-class-capped sampling for very large IDS CSVs (e.g.
CICIoT2023's multi-million-row train.csv) that don't fit comfortably in RAM
on a typical laptop.

Reads the source CSV in chunks (never holding the full file in memory),
caps each label/class at --max_per_class rows, and writes a single combined
output CSV that train.py / feature_selection.py can consume directly. Small
classes (e.g. a 140-row attack type) are kept in full; only large classes
get downsampled, so all classes stay represented -- unlike restricting to a
single attack category (e.g. DDoS-only), which would lose the multi-class
structure the baseline model and Feature Selection stage require.

NOTE: sampling is a greedy per-chunk take (each chunk is shuffled first,
then up to the remaining per-class quota is kept). This is NOT a globally
uniform random sample across the whole file -- it's a practical
approximation for RAM-constrained prototyping, not a statistically rigorous
sample. Good enough for feature-importance exploration and baseline
comparisons; revisit if you need publication-grade sampling guarantees.

Usage:
    python sample_dataset.py --csv data/CICIOT23/train/train.csv \
        --label_col label --max_per_class 20000 \
        --out data/CICIoT2023_sampled.csv
"""

import argparse

import pandas as pd


def sample_dataset(
    csv_path: str,
    label_col: str,
    max_per_class: int,
    out_path: str,
    chunksize: int = 200_000,
    seed: int = 42,
):
    counts: dict = {}
    first_chunk = True

    for chunk in pd.read_csv(csv_path, chunksize=chunksize):
        chunk = chunk.sample(frac=1.0, random_state=seed)
        keep_parts = []
        for label, group in chunk.groupby(label_col):
            have = counts.get(label, 0)
            room = max_per_class - have
            if room <= 0:
                continue
            take = group.iloc[:room]
            keep_parts.append(take)
            counts[label] = have + len(take)

        if not keep_parts:
            continue
        kept = pd.concat(keep_parts)
        kept.to_csv(out_path, mode="w" if first_chunk else "a",
                    header=first_chunk, index=False)
        first_chunk = False

    print(f"Wrote {out_path}")
    for label, n in sorted(counts.items(), key=lambda kv: -kv[1]):
        print(f"  {label}: {n}")
    print(f"Total rows: {sum(counts.values())}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--csv", required=True, help="Path to the large source CSV")
    parser.add_argument("--label_col", default="label")
    parser.add_argument("--max_per_class", type=int, default=20000,
                         help="Cap on rows kept per class (smaller classes kept in full)")
    parser.add_argument("--out", required=True, help="Path to write the sampled CSV")
    parser.add_argument("--chunksize", type=int, default=200_000,
                         help="Rows read per chunk (lower this if still memory-tight)")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    sample_dataset(
        csv_path=args.csv,
        label_col=args.label_col,
        max_per_class=args.max_per_class,
        out_path=args.out,
        chunksize=args.chunksize,
        seed=args.seed,
    )
