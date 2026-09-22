"""
Remap CICIoT2023's 34 fine-grained labels (BenignTraffic + 33 attack
subtypes) to the dataset's official 8-class scheme: Benign + 7 attack
categories (DDoS, DoS, Mirai, Recon, Spoofing, Web, BruteForce).

Why: the reference paper does not state how many classes it used for
CICIoT2023. Training on all 34 fine-grained labels forces the model to
tell near-identical attack subtypes apart (e.g. DDoS-SYN_Flood vs.
DDoS-RSTFINFlood vs. DDoS-ACK_Fragmentation) -- a much harder problem than
telling attack categories apart from Benign traffic -- and combined with
severe per-subtype class imbalance (some subtypes have only ~100-1000
rows), this measurably drags down weighted accuracy despite the model's
underlying discriminative power staying high (see the near-1.0 AUC in
train.py's runs). The categorization below matches the official grouping
from the CICIoT2023 dataset paper (Neto et al., 2023) and is the standard
"8-class" setting used by most CICIoT2023 papers.

Usage:
    python remap_ciciot2023_labels.py --in data/CICIOT23/train/train.csv \
        --out data/CICIOT23/train/train_8class.csv --label_col label
"""

import argparse
import pandas as pd

ATTACK_TO_CATEGORY = {
    "BenignTraffic": "Benign",

    # DDoS
    "DDoS-ICMP_Flood": "DDoS", "DDoS-UDP_Flood": "DDoS", "DDoS-TCP_Flood": "DDoS",
    "DDoS-PSHACK_Flood": "DDoS", "DDoS-SYN_Flood": "DDoS", "DDoS-RSTFINFlood": "DDoS",
    "DDoS-SynonymousIP_Flood": "DDoS", "DDoS-ICMP_Fragmentation": "DDoS",
    "DDoS-UDP_Fragmentation": "DDoS", "DDoS-ACK_Fragmentation": "DDoS",
    "DDoS-HTTP_Flood": "DDoS", "DDoS-SlowLoris": "DDoS",

    # DoS
    "DoS-UDP_Flood": "DoS", "DoS-TCP_Flood": "DoS", "DoS-SYN_Flood": "DoS",
    "DoS-HTTP_Flood": "DoS",

    # Mirai
    "Mirai-greeth_flood": "Mirai", "Mirai-udpplain": "Mirai", "Mirai-greip_flood": "Mirai",

    # Recon
    "Recon-HostDiscovery": "Recon", "Recon-OSScan": "Recon", "Recon-PortScan": "Recon",
    "Recon-PingSweep": "Recon", "VulnerabilityScan": "Recon",

    # Spoofing
    "MITM-ArpSpoofing": "Spoofing", "DNS_Spoofing": "Spoofing",

    # Web
    "BrowserHijacking": "Web", "CommandInjection": "Web", "SqlInjection": "Web",
    "XSS": "Web", "Backdoor_Malware": "Web", "Uploading_Attack": "Web",

    # BruteForce
    "DictionaryBruteForce": "BruteForce",
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
