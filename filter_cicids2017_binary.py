from __future__ import annotations
"""
Prepare CICIDS2017's Friday-Afternoon-DDoS capture as a Benign vs DDoS
binary dataset, for the cross-dataset generalization check the professor
asked for directly ("모든 종류의 DDoS 데이터셋에 대해서 0.99 이상의 정확도를
보이는지"): does the lightweight MLP's near-perfect CICIoT2023 binary
result hold on DDoS traffic from a COMPLETELY different collection
environment/tooling (CICFlowMeter on a university testbed, 2017), not just
a different DDoS subtype within the same dataset?

This file (Friday-WorkingHours-Afternoon-DDos.pcap_ISCX.csv) already
contains ONLY Benign/DDoS rows -- no other attack types to drop, unlike
CICIoT2023's filter_ciciot2023_binary.py. What this script does instead:

  1. Strips the leading/trailing whitespace CICIDS2017's column names are
     notorious for (e.g. " Label", " Destination Port").
  2. De-duplicates the 'Fwd Header Length' column, which appears TWICE in
     the raw CSV (a known quirk of the original dataset) -- pandas would
     silently rename the second occurrence to 'Fwd Header Length.1' on
     read, which would otherwise double-count a feature under two names.
  3. Leaves missing-value / non-finite-value cleanup to preprocess.py's
     existing pipeline (same as every other dataset here), so this script
     does no extra cleaning of its own.

Note: this is a SEPARATE model trained on CICIDS2017's own 78
CICFlowMeter features (not weight-transfer from the CICIoT2023 model --
the two datasets use different flow-feature extractors with no shared
schema). The comparison is "does the same lightweight-MLP methodology
reach similar accuracy on this dataset", not "does this exact trained
model generalize zero-shot".

`--extra_benign_files` pulls in BENIGN rows from CICIDS2017's other six
capture files (Monday/Tuesday/Wednesday/Thursday x2/Friday-Morning,
Friday-Afternoon-PortScan), matching the teammate's own RF experiment's
scope (a full-week BENIGN pool, ~1.86M rows, instead of just the ~97.7K
BENIGN rows that happen to fall in the Friday-Afternoon-DDoS capture).
Only BENIGN rows are taken from these files -- their other attack types
(PortScan, WebAttacks, Infiltration, ...) are dropped, since this is
still a Benign-vs-DDoS binary task, not a multi-attack one.

Usage:
    # Original (Friday-only BENIGN, ~97.7K rows):
    python filter_cicids2017_binary.py \
        --in "data/CICIDS2017/MachineLearningCVE/Friday-WorkingHours-Afternoon-DDos.pcap_ISCX.csv" \
        --out data/CICIDS2017/cicids2017_binary_ddos.csv

    # Full-week BENIGN pool (~1.86M rows), matching the teammate's RF scope:
    python filter_cicids2017_binary.py \
        --in "data/CICIDS2017/MachineLearningCVE/Friday-WorkingHours-Afternoon-DDos.pcap_ISCX.csv" \
        --extra_benign_files \
            "data/CICIDS2017/MachineLearningCVE/Monday-WorkingHours.pcap_ISCX.csv" \
            "data/CICIDS2017/MachineLearningCVE/Tuesday-WorkingHours.pcap_ISCX.csv" \
            "data/CICIDS2017/MachineLearningCVE/Wednesday-workingHours.pcap_ISCX.csv" \
            "data/CICIDS2017/MachineLearningCVE/Thursday-WorkingHours-Morning-WebAttacks.pcap_ISCX.csv" \
            "data/CICIDS2017/MachineLearningCVE/Thursday-WorkingHours-Afternoon-Infilteration.pcap_ISCX.csv" \
            "data/CICIDS2017/MachineLearningCVE/Friday-WorkingHours-Morning.pcap_ISCX.csv" \
            "data/CICIDS2017/MachineLearningCVE/Friday-WorkingHours-Afternoon-PortScan.pcap_ISCX.csv" \
        --out data/CICIDS2017/cicids2017_binary_ddos_fullweek.csv
"""

import argparse
import pandas as pd


def _load_clean(csv_path: str) -> pd.DataFrame:
    df = pd.read_csv(csv_path)
    df.columns = df.columns.str.strip()
    dup_cols = df.columns[df.columns.duplicated()].tolist()
    if dup_cols:
        print(f"  Dropping duplicate column(s) (keeping first occurrence): {dup_cols}")
        df = df.loc[:, ~df.columns.duplicated()]
    return df


def _dedup_and_resolve_conflicts(df: pd.DataFrame, label_col: str) -> pd.DataFrame:
    """
    Matches the teammate's prepare_cicids2017_flows.py cleaning: rows whose
    numeric features are byte-for-byte identical are collapsed to one
    (global, across the whole combined pool -- so duplicates introduced by
    merging multiple days' files are caught too), and rows whose features
    are identical but whose labels disagree (an ambiguous CICFlowMeter
    artifact) are dropped entirely rather than arbitrarily picking one.
    """
    feature_cols = [c for c in df.columns if c != label_col]
    numeric = df[feature_cols].apply(pd.to_numeric, errors="coerce")
    fingerprint = pd.util.hash_pandas_object(numeric, index=False)

    label_counts = pd.DataFrame({"hash": fingerprint, "y": df[label_col]}).groupby("hash")["y"].nunique()
    conflict_hashes = set(label_counts.index[label_counts > 1])
    is_conflict = fingerprint.isin(conflict_hashes)
    is_duplicate = fingerprint.duplicated(keep="first") & ~is_conflict

    print(f"Dedup: removed {int(is_conflict.sum()):,} feature-identical-but-conflicting-label rows, "
          f"{int(is_duplicate.sum()):,} feature-identical duplicate rows (kept first occurrence)")
    return df.loc[~is_conflict & ~is_duplicate].reset_index(drop=True)


def filter_binary(csv_in: str, csv_out: str, label_col: str = "Label",
                   extra_benign_files: list[str] | None = None, dedup: bool = True):
    df = _load_clean(csv_in)

    mask = df[label_col].isin(["BENIGN", "DDoS"])
    n_dropped = int((~mask).sum())
    df = df[mask].copy()
    print(f"{csv_in}: kept {len(df):,} BENIGN/DDoS rows, dropped {n_dropped:,} other rows")

    extra_frames = []
    for path in extra_benign_files or []:
        edf = _load_clean(path)
        edf = edf[edf[label_col] == "BENIGN"].copy()
        print(f"{path}: +{len(edf):,} BENIGN rows")
        extra_frames.append(edf)

    if extra_frames:
        df = pd.concat([df] + extra_frames, ignore_index=True)

    if dedup:
        df = _dedup_and_resolve_conflicts(df, label_col)

    df.to_csv(csv_out, index=False)

    print(f"\nWrote {csv_out}  shape={df.shape}")
    print(df[label_col].value_counts())


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--in", dest="csv_in", required=True)
    parser.add_argument("--out", dest="csv_out", required=True)
    parser.add_argument("--label_col", default="Label")
    parser.add_argument(
        "--extra_benign_files", nargs="*", default=[],
        help="Additional CICIDS2017 capture CSVs to pull BENIGN-only rows from "
             "(their other attack types are dropped).",
    )
    parser.add_argument(
        "--no_dedup", action="store_true",
        help="Skip the feature-identical duplicate/conflicting-label cleanup "
             "(default: applied, matching the teammate's own CICIDS2017 "
             "cleaning in prepare_cicids2017_flows.py).",
    )
    args = parser.parse_args()
    filter_binary(args.csv_in, args.csv_out, args.label_col, args.extra_benign_files,
                   dedup=not args.no_dedup)
