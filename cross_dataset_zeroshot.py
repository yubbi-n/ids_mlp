from __future__ import annotations
"""
True zero-shot cross-dataset transfer test for Benign vs DDoS binary
detection: train ONE model on CICIoT2023, then evaluate that SAME trained
model -- no retraining, no re-fit scaler -- directly on CICIDS2017.

This is a different (and stronger) claim than the earlier cross-dataset
result (CICIoT2023_binary_baseline vs CICIDS2017_binary_baseline), which
trained a SEPARATE model on each dataset's own native features and showed
only that the *methodology* transfers. This script tests whether the
*exact trained weights* generalize to data the model has never seen in
any form.

CICIoT2023 (46 native features) and CICIDS2017 (78 native features) come
from different flow extractors with no shared column schema, so this
requires a manual semantic mapping (COMMON_FEATURES below) down to a
smaller common input space. Several entries are best-effort judgment
calls, not exact definition matches:
  - header_length / tot_bytes / pkt_count: CICIoT2023 reports one
    aggregate number where CICIDS2017 splits Fwd/Bwd separately --
    summed here.
  - cwr_flag: CICIoT2023's CWR vs CICIDS2017's CWE flag count (closest
    available match; not confirmed to be the same bit).
  - iat: CICIoT2023's single IAT vs CICIDS2017's Flow/Fwd/Bwd IAT split
    -- Flow IAT Mean used as the closest match.
  - The *_flag_number columns may be presence indicators (0/1) in
    CICIoT2023 but are actual counts in CICIDS2017's *_Flag_Count columns
    -- a real scale/semantics mismatch the MinMaxScaler (fit on
    CICIoT2023 only) will not correct for. Worth flagging as a caveat
    when interpreting the zero-shot result, not something this script
    papers over.
  - CICIoT2023's protocol one-hot flags (HTTP/HTTPS/DNS/TCP/UDP/etc.) and
    derived stats (Magnitude/Radius/Covariance/Weight), and CICIDS2017's
    port/bulk/subflow/active-idle timing features have NO match on the
    other side and are dropped entirely. Net result: 21 common features
    out of CICIoT2023's 46.

Usage:
    python cross_dataset_zeroshot.py \
        --train_csv data/CICIOT23/train/train_binary_ddos.csv --train_label_col label \
        --test_csv data/CICIDS2017/cicids2017_binary_ddos.csv --test_label_col Label \
        --name CICIoT2023_to_CICIDS2017_zeroshot --output_dir results
"""

import argparse
import json
import os
import time

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, MinMaxScaler
from sklearn.metrics import accuracy_score, precision_recall_fscore_support

from model import LightweightMLP_IDS

# common_name -> (CICIoT2023 source column(s), CICIDS2017 source column(s))
# a multi-column list is SUMMED to build that one common feature.
COMMON_FEATURES = {
    "flow_duration":    (["flow_duration"], ["Flow Duration"]),
    "header_length":    (["Header_Length"], ["Fwd Header Length", "Bwd Header Length"]),
    "rate":             (["Rate"], ["Flow Packets/s"]),
    "srate":            (["Srate"], ["Fwd Packets/s"]),
    "drate":            (["Drate"], ["Bwd Packets/s"]),
    "fin_flag":         (["fin_flag_number"], ["FIN Flag Count"]),
    "syn_flag":         (["syn_flag_number"], ["SYN Flag Count"]),
    "rst_flag":         (["rst_flag_number"], ["RST Flag Count"]),
    "psh_flag":         (["psh_flag_number"], ["PSH Flag Count"]),
    "ack_flag":         (["ack_flag_number"], ["ACK Flag Count"]),
    "ece_flag":         (["ece_flag_number"], ["ECE Flag Count"]),
    "cwr_flag":         (["cwr_flag_number"], ["CWE Flag Count"]),
    "urg_count":        (["urg_count"], ["URG Flag Count"]),
    "tot_bytes":        (["Tot sum"], ["Total Length of Fwd Packets", "Total Length of Bwd Packets"]),
    "pkt_len_min":      (["Min"], ["Min Packet Length"]),
    "pkt_len_max":      (["Max"], ["Max Packet Length"]),
    "pkt_len_mean":     (["AVG"], ["Packet Length Mean"]),
    "pkt_len_std":      (["Std"], ["Packet Length Std"]),
    "pkt_len_variance": (["Variance"], ["Packet Length Variance"]),
    "iat":              (["IAT"], ["Flow IAT Mean"]),
    "pkt_count":        (["Number"], ["Total Fwd Packets", "Total Backward Packets"]),
}


def build_common_df(csv_path: str, label_col: str, source_idx: int) -> tuple[pd.DataFrame, pd.Series]:
    """source_idx: 0 selects the CICIoT2023 source columns, 1 selects CICIDS2017's."""
    df = pd.read_csv(csv_path)
    df.columns = df.columns.str.strip()
    df = df.loc[:, ~df.columns.duplicated()]
    df = df.dropna()

    out = {}
    for common_name, sources in COMMON_FEATURES.items():
        cols = sources[source_idx]
        missing = [c for c in cols if c not in df.columns]
        if missing:
            raise ValueError(f"Column(s) {missing} not found in {csv_path} (common feature '{common_name}')")
        out[common_name] = df[cols].sum(axis=1)
    X = pd.DataFrame(out)

    finite_mask = np.isfinite(X).all(axis=1)
    X = X.loc[finite_mask].reset_index(drop=True)
    y_raw = df.loc[finite_mask, label_col].reset_index(drop=True)
    return X, y_raw


def evaluate(model, device, le, X_np: np.ndarray, y_np: np.ndarray):
    model.eval()
    with torch.no_grad():
        X_t = torch.tensor(X_np, dtype=torch.float32, device=device)
        preds = model(X_t).argmax(dim=1).cpu().numpy()
    acc = accuracy_score(y_np, preds)
    prec, rec, f1, _ = precision_recall_fscore_support(y_np, preds, average="weighted", zero_division=0)
    class_prec, class_rec, class_f1, class_support = precision_recall_fscore_support(
        y_np, preds, average=None, labels=range(len(le.classes_)), zero_division=0
    )
    per_class = pd.DataFrame({
        "class": le.classes_, "precision": class_prec, "recall": class_rec,
        "f1": class_f1, "support": class_support,
    })
    return {"accuracy": acc, "precision": prec, "recall": rec, "f1": f1}, per_class


def run(train_csv, train_label_col, test_csv, test_label_col, name, output_dir,
        epochs=100, lr=3e-3, weight_decay=1e-4, batch_size=128, seed=42,
        lr_scheduler=True):
    os.makedirs(output_dir, exist_ok=True)
    torch.manual_seed(seed)

    X_train_full, y_train_raw = build_common_df(train_csv, train_label_col, 0)
    X_foreign, y_foreign_raw = build_common_df(test_csv, test_label_col, 1)

    # Unify label text: CICIoT2023 uses "Benign"/"DDoS", CICIDS2017 "BENIGN"/"DDoS".
    y_train_raw = y_train_raw.str.upper()
    y_foreign_raw = y_foreign_raw.str.upper()

    le = LabelEncoder()
    le.fit(sorted(set(y_train_raw) | set(y_foreign_raw)))
    y_train_full = le.transform(y_train_raw)
    y_foreign = le.transform(y_foreign_raw)

    X_tr_raw, X_te_raw, y_tr, y_te = train_test_split(
        X_train_full.values.astype(np.float32), y_train_full,
        test_size=0.2, random_state=seed, stratify=y_train_full,
    )

    scaler = MinMaxScaler()
    X_tr = scaler.fit_transform(X_tr_raw)
    X_te = scaler.transform(X_te_raw)
    X_foreign_scaled = scaler.transform(X_foreign.values.astype(np.float32))  # SAME scaler, never refit

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = LightweightMLP_IDS(input_dim=X_tr.shape[1], num_classes=len(le.classes_)).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    criterion = nn.CrossEntropyLoss()
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs) if lr_scheduler else None

    X_tr_t = torch.tensor(X_tr, dtype=torch.float32, device=device)
    y_tr_t = torch.tensor(y_tr, dtype=torch.long, device=device)
    loader = DataLoader(TensorDataset(X_tr_t, y_tr_t), batch_size=batch_size, shuffle=True)

    start_train = time.time()
    model.train()
    for epoch in range(epochs):
        for xb, yb in loader:
            optimizer.zero_grad()
            loss = criterion(model(xb), yb)
            loss.backward()
            optimizer.step()
        if scheduler is not None:
            scheduler.step()
    train_time = time.time() - start_train

    held_out_metrics, held_out_per_class = evaluate(model, device, le, X_te, y_te)
    zeroshot_metrics, zeroshot_per_class = evaluate(model, device, le, X_foreign_scaled, y_foreign)

    print(f"\n=== {name}: held-out CICIoT2023 test (same dataset, sanity check, {X_tr.shape[1]} common features) ===")
    print(held_out_metrics)
    print(held_out_per_class.to_string(index=False, float_format=lambda v: f"{v:.4f}"))

    print(f"\n=== {name}: ZERO-SHOT on {test_csv} (no retraining, no scaler refit) ===")
    print(zeroshot_metrics)
    print(zeroshot_per_class.to_string(index=False, float_format=lambda v: f"{v:.4f}"))

    held_out_per_class.to_csv(f"{output_dir}/{name}_held_out_per_class_metrics.csv", index=False)
    zeroshot_per_class.to_csv(f"{output_dir}/{name}_zeroshot_per_class_metrics.csv", index=False)

    summary = {
        "common_features": list(COMMON_FEATURES.keys()),
        "n_common_features": len(COMMON_FEATURES),
        "held_out_cicIot2023": held_out_metrics,
        "zeroshot_foreign": zeroshot_metrics,
        "train_time_s": train_time,
        "train_csv": train_csv, "test_csv": test_csv, "name": name,
        "lr_scheduler": lr_scheduler, "epochs": epochs, "seed": seed,
    }
    with open(f"{output_dir}/{name}_summary.json", "w") as f:
        json.dump(summary, f, indent=2)

    return summary


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--train_csv", required=True, help="Dataset the model is trained on")
    parser.add_argument("--train_label_col", required=True)
    parser.add_argument("--test_csv", required=True, help="Foreign dataset for zero-shot evaluation only")
    parser.add_argument("--test_label_col", required=True)
    parser.add_argument("--name", default="cross_dataset_zeroshot")
    parser.add_argument("--output_dir", default="results")
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--lr", type=float, default=3e-3)
    parser.add_argument("--weight_decay", type=float, default=1e-4)
    parser.add_argument("--batch_size", type=int, default=128)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--no_lr_scheduler", action="store_true")
    args = parser.parse_args()

    run(
        train_csv=args.train_csv, train_label_col=args.train_label_col,
        test_csv=args.test_csv, test_label_col=args.test_label_col,
        name=args.name, output_dir=args.output_dir,
        epochs=args.epochs, lr=args.lr, weight_decay=args.weight_decay,
        batch_size=args.batch_size, seed=args.seed,
        lr_scheduler=not args.no_lr_scheduler,
    )
