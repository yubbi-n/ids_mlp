"""
Remap CICIoT2023's 34 fine-grained labels (BenignTraffic + 33 attack
subtypes) to a coarser class scheme.

Two schemes are supported:

  --scheme 6class (default): the exact scheme the reference paper (Chandroth
  & Ali, Electronics 2026, 15(8), 1604) states it used for CICIoT2023
  (Section 4.1.3 / Table 6): "Benign, DDoS, DoS, MITM, Mirai, and Recon".
  Web-attack subtypes (BrowserHijacking, CommandInjection, SqlInjection,
  XSS, Backdoor_Malware, Uploading_Attack) and DictionaryBruteForce are NOT
  part of this scheme -- the paper's Table 6 never lists a Web or
  BruteForce class -- so rows with those labels are DROPPED rather than
  merged into another class. "MITM" merges both spoofing subtypes
  (MITM-ArpSpoofing, DNS_Spoofing); the paper doesn't state whether it used
  one or both, so this is a best-effort guess worth confirming with the
  authors.

  --scheme 8class: the dataset's own official 8-class categorization from
  Neto et al. (2023) -- Benign + DDoS/DoS/Mirai/Recon/Spoofing/Web/
  BruteForce -- used before we found the paper's explicit 6-class
  statement. Kept for comparison.

Why remap at all: training on all 34 fine-grained labels forces the model
to tell near-identical attack subtypes apart (e.g. DDoS-SYN_Flood vs.
DDoS-RSTFINFlood vs. DDoS-ACK_Fragmentation) -- a much harder problem than
telling attack categories apart from Benign traffic -- and combined with
severe per-subtype class imbalance (some subtypes have only ~100-1000
rows), this measurably drags down weighted accuracy despite the model's
underlying discriminative power staying high (see the near-1.0 AUC in
train.py's runs).

Usage:
    python remap_ciciot2023_labels.py --in data/CICIOT23/train/train.csv \
        --out data/CICIOT23/train/train_6class.csv --label_col label

    python remap_ciciot2023_labels.py --in data/CICIOT23/train/train.csv \
        --out data/CICIOT23/train/train_8class.csv --label_col label --scheme 8class
"""

import argparse
import pandas as pd

_DDOS = {
    "DDoS-ICMP_Flood", "DDoS-UDP_Flood", "DDoS-TCP_Flood",
    "DDoS-PSHACK_Flood", "DDoS-SYN_Flood", "DDoS-RSTFINFlood",
    "DDoS-SynonymousIP_Flood", "DDoS-ICMP_Fragmentation",
    "DDoS-UDP_Fragmentation", "DDoS-ACK_Fragmentation",
    "DDoS-HTTP_Flood", "DDoS-SlowLoris",
}
_DOS = {
    "DoS-UDP_Flood", "DoS-TCP_Flood", "DoS-SYN_Flood", "DoS-HTTP_Flood",
}
_MIRAI = {
    "Mirai-greeth_flood", "Mirai-udpplain", "Mirai-greip_flood",
}
_RECON = {
    "Recon-HostDiscovery", "Recon-OSScan", "Recon-PortScan",
    "Recon-PingSweep", "VulnerabilityScan",
}
_SPOOFING = {"MITM-ArpSpoofing", "DNS_Spoofing"}
_WEB = {
    "BrowserHijacking", "CommandInjection", "SqlInjection",
    "XSS", "Backdoor_Malware", "Uploading_Attack",
}
_BRUTEFORCE = {"DictionaryBruteForce"}

ATTACK_TO_CATEGORY_6CLASS = {
    "BenignTraffic": "Benign",
    **{label: "DDoS" for label in _DDOS},
    **{label: "DoS" for label in _DOS},
    **{label: "Mirai" for label in _MIRAI},
    **{label: "Recon" for label in _RECON},
    **{label: "MITM" for label in _SPOOFING},
}
# Not part of the paper's stated 6-class scheme -- dropped, not merged.
EXCLUDED_6CLASS = _WEB | _BRUTEFORCE

ATTACK_TO_CATEGORY_8CLASS = {
    "BenignTraffic": "Benign",
    **{label: "DDoS" for label in _DDOS},
    **{label: "DoS" for label in _DOS},
    **{label: "Mirai" for label in _MIRAI},
    **{label: "Recon" for label in _RECON},
    **{label: "Spoofing" for label in _SPOOFING},
    **{label: "Web" for label in _WEB},
    **{label: "BruteForce" for label in _BRUTEFORCE},
}


def remap(csv_in: str, csv_out: str, label_col: str = "label", scheme: str = "6class"):
    df = pd.read_csv(csv_in)

    if scheme == "6class":
        mapping, excluded = ATTACK_TO_CATEGORY_6CLASS, EXCLUDED_6CLASS
    elif scheme == "8class":
        mapping, excluded = ATTACK_TO_CATEGORY_8CLASS, set()
    else:
        raise ValueError('scheme must be "6class" or "8class"')

    unmapped = set(df[label_col].unique()) - set(mapping.keys()) - excluded
    if unmapped:
        raise ValueError(
            f"Unmapped label(s) found: {sorted(unmapped)}. "
            f"Add them to the mapping before proceeding."
        )

    if excluded:
        n_before = len(df)
        df = df[~df[label_col].isin(excluded)].copy()
        n_dropped = n_before - len(df)
        if n_dropped:
            print(f"Dropped {n_dropped} rows not in the {scheme} scheme "
                  f"(labels: {sorted(excluded)})")

    df[label_col] = df[label_col].map(mapping)
    df.to_csv(csv_out, index=False)

    print(f"Wrote {csv_out}  shape={df.shape}  scheme={scheme}")
    print(df[label_col].value_counts())


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--in", dest="csv_in", required=True)
    parser.add_argument("--out", dest="csv_out", required=True)
    parser.add_argument("--label_col", default="label")
    parser.add_argument(
        "--scheme", choices=["6class", "8class"], default="6class",
        help="6class (default): the paper's stated scheme (Section 4.1.3) -- "
             "drops Web/BruteForce rows, merges spoofing into MITM. "
             "8class: the dataset's official Neto et al. scheme.",
    )
    args = parser.parse_args()
    remap(args.csv_in, args.csv_out, args.label_col, args.scheme)
