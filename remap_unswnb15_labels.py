from __future__ import annotations
"""
Remap UNSW-NB15's 10-class `attack_cat` labels down to the 3 categories it
shares with CICIoT2023's 6-class scheme (Benign/DoS/Recon), for the
cross-dataset generalization check in the research plan (2.1절).

UNSW-NB15 has no DDoS/MITM/Mirai equivalent -- its `attack_cat` values are
Normal, Generic, Exploits, Fuzzers, DoS, Reconnaissance, Analysis, Backdoor,
Shellcode, Worms (confirmed from the actual parquet files). Only Normal,
DoS, and Reconnaissance have a clean match to CICIoT2023's scheme; the rest
(Generic/Exploits/Fuzzers/Analysis/Backdoor/Shellcode/Worms) are dropped
entirely, the same way CICIoT2023's 6-class remap drops Web/BruteForce.

IMPORTANT -- leakage: the dataset's own binary `label` column (0=Normal,
1=attack) is 100% redundant with `attack_cat` and MUST be dropped before
training, or it trivially determines the target. This script always drops
it, regardless of --label_col.

Usage:
    python remap_unswnb15_labels.py \
        --in data/UNSW-NB15/UNSW_NB15_training-set.parquet \
        --out data/UNSW-NB15/train_3class.csv
    python remap_unswnb15_labels.py \
        --in data/UNSW-NB15/UNSW_NB15_testing-set.parquet \
        --out data/UNSW-NB15/test_3class.csv
"""

import argparse
import pandas as pd

ATTACK_CAT_TO_3CLASS = {
    "Normal": "Benign",
    "DoS": "DoS",
    "Reconnaissance": "Recon",
}
# No clean CICIoT2023-scheme equivalent -- dropped, not merged.
EXCLUDED = {
    "Generic", "Exploits", "Fuzzers", "Analysis",
    "Backdoor", "Shellcode", "Worms",
}

LEAKY_BINARY_LABEL_COL = "label"


def remap(parquet_in: str, csv_out: str, attack_cat_col: str = "attack_cat"):
    df = pd.read_parquet(parquet_in)

    unmapped = set(df[attack_cat_col].unique()) - set(ATTACK_CAT_TO_3CLASS.keys()) - EXCLUDED
    if unmapped:
        raise ValueError(
            f"Unmapped attack_cat value(s) found: {sorted(unmapped)}. "
            f"Add them to ATTACK_CAT_TO_3CLASS or EXCLUDED before proceeding."
        )

    n_before = len(df)
    df = df[~df[attack_cat_col].isin(EXCLUDED)].copy()
    n_dropped = n_before - len(df)
    if n_dropped:
        print(f"Dropped {n_dropped} rows not in the 3-class scheme "
              f"(labels: {sorted(EXCLUDED)})")

    df[attack_cat_col] = df[attack_cat_col].map(ATTACK_CAT_TO_3CLASS)

    if LEAKY_BINARY_LABEL_COL in df.columns:
        df = df.drop(columns=[LEAKY_BINARY_LABEL_COL])
        print(f"Dropped leaky binary '{LEAKY_BINARY_LABEL_COL}' column "
              f"(100% redundant with attack_cat)")

    df.to_csv(csv_out, index=False)

    print(f"Wrote {csv_out}  shape={df.shape}")
    print(df[attack_cat_col].value_counts())


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--in", dest="parquet_in", required=True,
                         help="Path to UNSW_NB15_{training,testing}-set.parquet")
    parser.add_argument("--out", dest="csv_out", required=True)
    parser.add_argument("--attack_cat_col", default="attack_cat")
    args = parser.parse_args()
    remap(args.parquet_in, args.csv_out, args.attack_cat_col)
