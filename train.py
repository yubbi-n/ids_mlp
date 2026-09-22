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
import copy
import json
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
    val_size: float = 0.1,
    seed: int = 42,
    dataset_name: str = "dataset",
    output_dir: str = ".",
):
    os.makedirs(output_dir, exist_ok=True)

    df = pd.read_csv(csv_path)
    X_df, y, le = preprocess_full_pipeline(
        df, label_col, categorical_cols, categorical_mode=categorical_mode
    )

    # 3-way split: test_size held out for the FINAL report only, never seen
    # during training or for checkpoint selection. val_size (also a fraction
    # of the full dataset) is used only to pick the best-epoch checkpoint --
    # picking that checkpoint by test accuracy would leak the test set into
    # model selection and bias the reported numbers optimistically.
    # NOTE: the paper (Section 4.1) describes a plain 80/20 train/test split;
    # this 3-way split is a deliberate deviation for a methodologically
    # sound "best epoch" (see train_and_evaluate's best-checkpoint logic
    # below). Pass val_size=0 to fall back to a plain 80/20 split.
    X_temp, X_test, y_temp, y_test = train_test_split(
        X_df.values.astype(np.float32), y,
        test_size=test_size, random_state=seed, stratify=y,
    )
    if val_size > 0:
        val_frac_of_temp = val_size / (1 - test_size)
        X_train, X_val, y_train, y_val = train_test_split(
            X_temp, y_temp, test_size=val_frac_of_temp, random_state=seed, stratify=y_temp,
        )
    else:
        X_train, y_train = X_temp, y_temp
        X_val, y_val = X_test, y_test  # no held-out val: fall back to test-based selection

    scaler = MinMaxScaler()
    X_train = scaler.fit_transform(X_train)
    X_val = scaler.transform(X_val)
    X_test = scaler.transform(X_test)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    X_train_t = torch.tensor(X_train, dtype=torch.float32)
    y_train_t = torch.tensor(y_train, dtype=torch.long)
    X_val_t = torch.tensor(X_val, dtype=torch.float32).to(device)
    y_val_t = torch.tensor(y_val, dtype=torch.long).to(device)
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

    train_acc_hist, val_acc_hist = [], []
    train_loss_hist, val_loss_hist = [], []
    best_val_acc = -1.0
    best_epoch = -1
    best_state = None

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
            val_logits = model(X_val_t)
            val_loss = criterion(val_logits, y_val_t).item()
            val_acc = (val_logits.argmax(dim=1) == y_val_t).float().mean().item()

        train_acc_hist.append(train_acc)
        val_acc_hist.append(val_acc)
        train_loss_hist.append(train_loss)
        val_loss_hist.append(val_loss)

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            best_epoch = epoch + 1
            best_state = copy.deepcopy(model.state_dict())

        if (epoch + 1) % 10 == 0 or epoch == 0:
            print(
                f"[{dataset_name}] epoch {epoch + 1}/{epochs} "
                f"train_loss={train_loss:.4f} train_acc={train_acc:.4f} "
                f"val_loss={val_loss:.4f} val_acc={val_acc:.4f}"
            )

    train_time = time.time() - start_train

    # Report on the best-validation-accuracy epoch's weights rather than
    # whatever epoch training happened to stop on: without an LR schedule
    # the loss curve oscillates a lot (see the accuracy/loss plots), so the
    # final epoch's number is a noisy, arbitrary landing point -- not a
    # reliable basis for comparing runs (e.g. Baseline vs a Top-k feature
    # subset). Selection uses the VALIDATION set, never the test set, so
    # the final test-set metrics below stay an unbiased, held-out estimate.
    if best_state is not None:
        model.load_state_dict(best_state)
        print(f"[{dataset_name}] using best epoch {best_epoch}/{epochs} "
              f"(val_acc={best_val_acc:.4f}) for final evaluation")

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
    axes[0].plot(val_acc_hist, label="val")
    if best_epoch > 0:
        axes[0].axvline(best_epoch - 1, color="grey", linestyle="--", linewidth=1,
                         label=f"best epoch ({best_epoch})")
    axes[0].set_title(f"{dataset_name} - Accuracy")
    axes[0].set_xlabel("epoch")
    axes[0].legend()

    axes[1].plot(train_loss_hist, label="train")
    axes[1].plot(val_loss_hist, label="val")
    axes[1].set_title(f"{dataset_name} - Loss")
    axes[1].set_xlabel("epoch")
    axes[1].legend()
    plt.tight_layout()
    plt.savefig(f"{output_dir}/{dataset_name}_curves.png", dpi=150)
    plt.close(fig)

    # Scale the figure and tick labels with the number of classes -- with a
    # few dozen IDS attack classes (e.g. CICIoT2023's 34), a fixed small
    # figure with annotated cell values becomes unreadable (overlapping
    # numbers/labels), so annotation is dropped and the font shrinks once
    # there are too many classes to label individually.
    cm = confusion_matrix(y_true, preds, normalize="true")
    n_classes = len(le.classes_)
    fig_side = max(6, n_classes * 0.45)
    tick_fontsize = 9 if n_classes <= 15 else max(5, 9 - (n_classes - 15) // 5)

    plt.figure(figsize=(fig_side, fig_side))
    ax = sns.heatmap(
        cm, annot=n_classes <= 15, fmt=".2f", cmap="Blues",
        xticklabels=le.classes_, yticklabels=le.classes_,
        square=True,
    )
    ax.tick_params(axis="x", labelsize=tick_fontsize, rotation=90)
    ax.tick_params(axis="y", labelsize=tick_fontsize, rotation=0)
    plt.title(f"{dataset_name} - Normalized Confusion Matrix")
    plt.ylabel("True label")
    plt.xlabel("Predicted label")
    plt.tight_layout()
    plt.savefig(f"{output_dir}/{dataset_name}_confusion_matrix.png", dpi=150)
    plt.close()

    summary = {
        "accuracy": acc, "precision": prec, "recall": rec, "f1": f1, "auc": auc,
        "params": n_params, "flops": flops, "macs": macs,
        "model_size_kb": model_size_kb,
        "train_time_s": train_time, "test_time_s": test_time,
        "best_epoch": best_epoch, "epochs": epochs,
        "csv_path": csv_path, "dataset_name": dataset_name,
    }
    with open(f"{output_dir}/{dataset_name}_summary.json", "w") as f:
        json.dump(summary, f, indent=2)

    return summary


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
    parser.add_argument(
        "--val_size", type=float, default=0.1,
        help="Fraction of the full dataset held out for best-epoch checkpoint "
             "selection (kept separate from the test set). Pass 0 for a plain "
             "80/20 train/test split with no validation-based checkpointing "
             "(closer to the paper's literal Section 4.1 description).",
    )
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
        val_size=args.val_size,
        dataset_name=args.name,
        output_dir=args.output_dir,
    )
