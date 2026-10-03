from __future__ import annotations
"""
Filter CICIoT2023 down to Benign + DDoS rows only, with a binary label
(Benign=0, DDoS=1 once LabelEncoder sorts them alphabetically in
preprocess.py), matching the team research plan's required scheme ("기본은
정상/공격 이진 분류", attack type DDoS already selected) and the teammate's
Random Forest pipeline's own label convention (binary_label: DDoS=1,
Benign=0).

This is Benign vs DDoS ONLY -- rows belonging to any other attack category
(DoS, Mirai, MITM/Spoofing, Recon, Web, BruteForce) are dropped entirely,
not folded into a "non-DDoS" negative class.

Usage:
    python filter_ciciot2023_binary.py --in data/CICIOT23/train/train.csv \
        --out data/CICIOT23/train/train_binary_ddos.csv --label_col label
"""

import argparse
import pandas as pd

from remap_ciciot2023_labels import _DDOS

LABEL_MAP = {"BenignTraffic": "Benign", **{label: "DDoS" for label in _DDOS}}


def filter_binary(csv_in: str, csv_out: str, label_col: str = "label"):
    df = pd.read_csv(csv_in)

    mask = df[label_col].isin(LABEL_MAP.keys())
    n_dropped = int((~mask).sum())
    df = df[mask].copy()
    df[label_col] = df[label_col].map(LABEL_MAP)
    df.to_csv(csv_out, index=False)

    print(f"Dropped {n_dropped} rows that are neither Benign nor DDoS")
    print(f"Wrote {csv_out}  shape={df.shape}")
    print(df[label_col].value_counts())


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--in", dest="csv_in", required=True)
    parser.add_argument("--out", dest="csv_out", required=True)
    parser.add_argument("--label_col", default="label")
    args = parser.parse_args()
    filter_binary(args.csv_in, args.csv_out, args.label_col)
