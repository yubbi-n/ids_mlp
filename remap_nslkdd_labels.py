"""
Remap NSL-KDD raw attack-type labels to the standard 5-class scheme:
Normal / DoS / Probe / R2L / U2R.

This mapping follows the widely-used categorization from Tavallaee et al.
(2009) and is what most NSL-KDD papers (including, most likely, the paper
being reproduced) mean by "5 classes". It covers both KDDTrain+ attack
types and the extra novel attack types that only appear in KDDTest+.

Usage:
    python remap_nslkdd_labels.py --in data/NSL-KDD_train.csv \
        --out data/NSL-KDD_train_5class.csv --label_col label
"""

import argparse
import pandas as pd

ATTACK_TO_CATEGORY = {
    "normal": "Normal",

    # DoS
    "back": "DoS", "land": "DoS", "neptune": "DoS", "pod": "DoS",
    "smurf": "DoS", "teardrop": "DoS", "mailbomb": "DoS",
    "processtable": "DoS", "udpstorm": "DoS", "apache2": "DoS",
    "worm": "DoS",

    # Probe
    "satan": "Probe", "ipsweep": "Probe", "nmap": "Probe",
    "portsweep": "Probe", "mscan": "Probe", "saint": "Probe",

    # R2L
    "guess_passwd": "R2L", "ftp_write": "R2L", "imap": "R2L",
    "phf": "R2L", "multihop": "R2L", "warezmaster": "R2L",
    "warezclient": "R2L", "spy": "R2L", "xlock": "R2L", "xsnoop": "R2L",
    "snmpgetattack": "R2L", "snmpguess": "R2L", "httptunnel": "R2L",
    "sendmail": "R2L", "named": "R2L",

    # U2R
    "buffer_overflow": "U2R", "loadmodule": "U2R", "perl": "U2R",
    "rootkit": "U2R", "ps": "U2R", "sqlattack": "U2R", "xterm": "U2R",
}


def remap(csv_in: str, csv_out: str, label_col: str = "label"):
    df = pd.read_csv(csv_in)

    unmapped = set(df[label_col].unique()) - set(ATTACK_TO_CATEGORY.keys())
    if unmapped:
        raise ValueError(
            f"Unmapped label(s) found: {sorted(unmapped)}. "
            f"Add them to ATTACK_TO_CATEGORY before proceeding."
        )

    df[label_col] = df[label_col].map(ATTACK_TO_CATEGORY)
    df.to_csv(csv_out, index=False)

    print(f"Wrote {csv_out}  shape={df.shape}")
    print(df[label_col].value_counts())


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--in", dest="csv_in", required=True)
    parser.add_argument("--out", dest="csv_out", required=True)
    parser.add_argument("--label_col", default="label")
    args = parser.parse_args()
    remap(args.csv_in, args.csv_out, args.label_col)
