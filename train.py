from __future__ import annotations
"""
Training / evaluation script matching Section 4 of the paper:
  - 80/20 train/test split (Section 4.1)
  - AdamW optimizer, lr=0.003, weight_decay=1e-4 (Section 4.3, Table 3)
  - 100 epochs, batch size 128 (Section 4.3, Table 3)
  - CrossEntropyLoss (Eq. 6)
  - Metrics: accuracy, precision, recall, F1, AUC-ROC (Section 4.4)
  - Computational complexity: param count, FLOPs/MACs (approx), model size,
    train/test time (Section 4.6, Table 7)

Usage:
    python train.py --csv path/to/dataset.csv --label_col Label \
        --categorical_cols protocol_type service flag \
        --name NSL-KDD

NOTE (ambiguity to confirm with authors -- see README.md for the full list):
  - Exact FLOPs/MACs counting convention used to get their reported numbers
    (e.g. 45,200 FLOPs / 22,600 MACs for CICIDS2017) is not stated. This
    script provides a simple analytic estimate; matching their exact figures
    may require knowing the profiling tool (thop / ptflops / fvcore / manual)
    they used.
  - Whether the 80/20 split was stratified and what random seed was used.
"""

import argparse
import os
import time

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import (
    accuracy_score,
    precision_recall_fscore_support,
    confusion_matrix,
    roc_auc_score,
)
import matplotlib.pyplot as plt
import seaborn as sns

from model import LightweightMLP_IDS
from preprocess import preprocess_full_pipeline


def count_params(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def estimate_flops(input_dim: int, hidden1: int, embedding_dim: int, num_classes: int):
    """
    Simple analytic MACs/FLOPs estimate (weights only, ignores biases and the
    final softmax): FLOPs = 2 * MACs. This is an approximation -- see the
    module docstring NOTE about confirming the authors' exact convention.
    """
    macs = input_dim * hidden1 + hidden1 * embedding_dim + embedding_dim * num_classes
    flops = 2 * macs
    return flops, macs


def train_and_evaluate(
    csv_path: str,
    label_col: str,
    categorical_cols=None,
    categorical_mode: str = "onehot",
    embedding_activation: bool = True,
    epochs: int = 100,
    batch_size: int = 128,
    lr: float = 3e-3,
    weight_decay: float = 1e-4,
    test_size: float = 0.2,
    seed: int = 42,
    dataset_name: str = "dataset",
    output_dir: str = ".",
):
    os.makedirs(output_dir, exist_ok=True)

    df = pd.read_csv(csv_path)
    X_df, y, le = preprocess_full_pipeline(
        df, label_col, categorical_cols, categorical_mode=categorical_mode
    )

    X_train, X_test, y_train, y_test = train_test_split(
        X_df.values.astype(np.float32), y,
        test_size=test_size, random_state=seed, stratify=y,
    )

    scaler = MinMaxScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    X_train_t = torch.tensor(X_train, dtype=torch.float32)
    y_train_t = torch.tensor(y_train, dtype=torch.long)
    X_test_t = torch.tensor(X_test, dtype=torch.float32).to(device)
    y_test_t = torch.tensor(y_test, dtype=torch.long).to(device)

    train_loader = DataLoader(
        TensorDataset(X_train_t, y_train_t), batch_size=batch_size, shuffle=True
    )

    num_classes = len(le.classes_)
    input_dim = X_df.shape[1]

    model = LightweightMLP_IDS(
        input_dim=input_dim, num_classes=num_classes,
        embedding_activation=embedding_activation,
    ).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    criterion = nn.CrossEntropyLoss()

    train_acc_hist, test_acc_hist = [], []
    train_loss_hist, test_loss_hist = [], []

    start_train = time.time()
    for epoch in range(epochs):
        model.train()
        correct, total, running_loss = 0, 0, 0.0
        for xb, yb in train_loader:
            xb, yb = xb.to(device), yb.to(device)
            optimizer.zero_grad()
            logits = model(xb)
            loss = criterion(logits, yb)
            loss.backward()
            optimizer.step()

            running_loss += loss.item() * xb.size(0)
            correct += (logits.argmax(dim=1) == yb).sum().item()
            total += xb.size(0)

        train_loss = running_loss / total
        train_acc = correct / total

        model.eval()
        with torch.no_grad():
            test_logits = model(X_test_t)
            test_loss = criterion(test_logits, y_test_t).item()
            test_acc = (test_logits.argmax(dim=1) == y_test_t).float().mean().item()

        train_acc_hist.append(train_acc)
        test_acc_hist.append(test_acc)
        train_loss_hist.append(train_loss)
        test_loss_hist.append(test_loss)

        if (epoch + 1) % 10 == 0 or epoch == 0:
            print(
                f"[{dataset_name}] epoch {epoch + 1}/{epochs} "
                f"train_loss={train_loss:.4f} train_acc={train_acc:.4f} "
                f"test_loss={test_loss:.4f} test_acc={test_acc:.4f}"
            )

    train_time = time.time() - start_train

    # ---- final evaluation ----
    model.eval()
    start_test = time.time()
    with torch.no_grad():
        logits = model(X_test_t)
        probs = torch.softmax(logits, dim=1).cpu().numpy()
        preds = logits.argmax(dim=1).cpu().numpy()
    test_time = time.time() - start_test

    y_true = y_test_t.cpu().numpy()
    acc = accuracy_score(y_true, preds)
    prec, rec, f1, _ = precision_recall_fscore_support(
        y_true, preds, average="weighted", zero_division=0
    )
    try:
        auc = roc_auc_score(y_true, probs, multi_class="ovr", average="weighted")
    except ValueError:
        auc = float("nan")

    n_params = count_params(model)
    flops, macs = estimate_flops(input_dim, 128, 64, num_classes)
    model_size_kb = n_params * 4 / 1024  # float32 assumption

    print(f"\n=== {dataset_name} summary ===")
    print(
        f"Accuracy={acc:.4f} Precision={prec:.4f} Recall={rec:.4f} "
        f"F1={f1:.4f} AUC={auc:.4f}"
    )
    print(
        f"Params={n_params} FLOPs(approx)={flops} MACs(approx)={macs} "
        f"ModelSize={model_size_kb:.1f} KB TrainTime={train_time:.2f}s "
        f"TestTime={test_time:.3f}s"
    )

    # ---- plots (accuracy/loss curves, confusion matrix -- Figures 2-4) ----
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    axes[0].plot(train_acc_hist, label="train")
    axes[0].plot(test_acc_hist, label="test")
    axes[0].set_title(f"{dataset_name} - Accuracy")
    axes[0].set_xlabel("epoch")
    axes[0].legend()

    axes[1].plot(train_loss_hist, label="train")
    axes[1].plot(test_loss_hist, label="test")
    axes[1].set_title(f"{dataset_name} - Loss")
    axes[1].set_xlabel("epoch")
    axes[1].legend()
    plt.tight_layout()
    plt.savefig(f"{output_dir}/{dataset_name}_curves.png", dpi=150)
    plt.close(fig)

    cm = confusion_matrix(y_true, preds, normalize="true")
    plt.figure(figsize=(6, 5))
    sns.heatmap(
        cm, annot=True, fmt=".2f", cmap="Blues",
        xticklabels=le.classes_, yticklabels=le.classes_,
    )
    plt.title(f"{dataset_name} - Normalized Confusion Matrix")
    plt.ylabel("True label")
    plt.xlabel("Predicted label")
    plt.tight_layout()
    plt.savefig(f"{output_dir}/{dataset_name}_confusion_matrix.png", dpi=150)
    plt.close()

    return {
        "accuracy": acc, "precision": prec, "recall": rec, "f1": f1, "auc": auc,
        "params": n_params, "flops": flops, "macs": macs,
        "model_size_kb": model_size_kb,
        "train_time_s": train_time, "test_time_s": test_time,
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--csv", required=True, help="Path to dataset CSV")
    parser.add_argument("--label_col", required=True, help="Name of the label/target column")
    parser.add_argument(
        "--categorical_cols", nargs="*", default=[],
        help="Names of categorical feature columns to encode (or drop)",
    )
    parser.add_argument(
        "--categorical_mode", choices=["onehot", "drop"], default="onehot",
        help="How to handle categorical_cols: one-hot encode, or drop entirely",
    )
    parser.add_argument(
        "--no_embedding_activation", action="store_true",
        help="Disable ReLU on the embedding layer (Eq. 3 literal reading)",
    )
    parser.add_argument("--name", default="dataset", help="Dataset name (for logging/plots)")
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--batch_size", type=int, default=128)
    parser.add_argument("--lr", type=float, default=3e-3)
    parser.add_argument("--weight_decay", type=float, default=1e-4)
    parser.add_argument("--output_dir", default=".")
    args = parser.parse_args()

    train_and_evaluate(
        csv_path=args.csv,
        label_col=args.label_col,
        categorical_cols=args.categorical_cols,
        categorical_mode=args.categorical_mode,
        embedding_activation=not args.no_embedding_activation,
        epochs=args.epochs,
        batch_size=args.batch_size,
        lr=args.lr,
        weight_decay=args.weight_decay,
        dataset_name=args.name,
        output_dir=args.output_dir,
    )
