from __future__ import annotations
"""
One-off prep script: turn the teammate's PCAP-extracted CICIoT2023 dataset
(yejinhan11/edge_ai_iot_ids_optimization's data/processed/dataset_v1_parts/,
38 official features, uniform 10-packet windows -- no window-size/label
leak unlike the official Kaggle CSV) into a train/test CSV pair that
train.py's new --test_csv option can consume directly.

Split methodology matches the teammate's own train_rf.py exactly (per
source_pcap file, first --train_frac of rows in file order -> train, rest
-> test), so results are directly comparable to her RF numbers and avoid
randomly mixing adjacent/highly-correlated windows across train and test
(a real leakage risk for windowed features like these, independent of the
window-size-leak issue found in the Kaggle CSV).

Usage:
    python prepare_pcap_dataset.py \
        --parquet_path /path/to/dataset_v1_parts \
        --out_prefix data/CICIoT2023_pcap_binary \
        --max_train_rows 1000000
"""

import argparse

import numpy as np
import pandas as pd

# Matches yejinhan11/edge_ai_iot_ids_optimization's src/build_dataset.py
# FEATURES exactly (official 39 CICIoT2023 features minus LLC, which is
# dropped there as always-1 due to an extractor bug).
FEATURES = [
    "Header_Length", "Protocol Type", "Time_To_Live", "Rate", "fin_flag_number",
    "syn_flag_number", "rst_flag_number", "psh_flag_number", "ack_flag_number",
    "ece_flag_number", "cwr_flag_number", "ack_count", "syn_count", "fin_count",
    "rst_count", "HTTP", "HTTPS", "DNS", "Telnet", "SMTP", "SSH", "IRC", "TCP",
    "UDP", "DHCP", "ARP", "ICMP", "IGMP", "IPv", "Tot sum", "Min", "Max", "AVG",
    "Std", "Tot size", "IAT", "Number", "Variance",
]


def time_split_mask(df: pd.DataFrame, train_frac: float) -> np.ndarray:
    """is_train = row within the first `train_frac` of its source_pcap file
    (by row_in_file order) -- identical to the teammate's train_rf.py."""
    n_in_file = df.groupby("source_pcap", observed=True)["row_in_file"].transform("max") + 1
    return (df["row_in_file"] < n_in_file * train_frac).to_numpy()


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--parquet_path", required=True,
                     help="Path to dataset_v1.parquet or the dataset_v1_parts/ directory")
    ap.add_argument("--out_prefix", required=True,
                     help="Writes <out_prefix>_train.csv and <out_prefix>_test.csv")
    ap.add_argument("--train_frac", type=float, default=0.7,
                     help="Per-file fraction of rows (in row_in_file order) used for train")
    ap.add_argument("--max_train_rows", type=int, default=1_000_000,
                     help="Random cap on training rows after the time split (matches the "
                          "teammate's train_rf.py default). 0 disables capping.")
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    print(f"loading {args.parquet_path} ...")
    df = pd.read_parquet(args.parquet_path, columns=FEATURES + ["source_pcap", "row_in_file", "binary_label"])
    print(f"rows={len(df):,} cols={len(df.columns)}")

    is_train = time_split_mask(df, args.train_frac)
    label = df["binary_label"].map({0: "Benign", 1: "DDoS"})

    train_df = df.loc[is_train, FEATURES].copy()
    train_df["label"] = label[is_train].values
    test_df = df.loc[~is_train, FEATURES].copy()
    test_df["label"] = label[~is_train].values

    rng = np.random.default_rng(args.seed)
    if args.max_train_rows and len(train_df) > args.max_train_rows:
        idx = rng.choice(len(train_df), args.max_train_rows, replace=False)
        train_df = train_df.iloc[idx].reset_index(drop=True)

    print(f"train rows={len(train_df):,} ({train_df['label'].value_counts().to_dict()})")
    print(f"test  rows={len(test_df):,} ({test_df['label'].value_counts().to_dict()})")

    train_path = f"{args.out_prefix}_train.csv"
    test_path = f"{args.out_prefix}_test.csv"
    train_df.to_csv(train_path, index=False)
    test_df.to_csv(test_path, index=False)
    print(f"\nwrote {train_path}\nwrote {test_path}")


if __name__ == "__main__":
    main()
