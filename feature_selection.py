from __future__ import annotations
"""
Feature importance / feature selection, matching Section 2.3 of the research
plan ("특징 선택 및 경량화 기법 기반 Edge AI IoT 침입 탐지 및 자원 효율 분석").

Two importance views are computed on the TRAIN split only, and the SAME
train/test split as train.py (same seed / test_size / stratify), to avoid
leaking test-set information into feature selection (research plan
Section 2.3-(4)):

  1. Random Forest feature importance
     RandomForestClassifier is fit on X_train and used as the baseline
     importance-ranking model (impurity-based `feature_importances_`).

  2. SHAP feature importance
     By default computed on the same Random Forest via shap.TreeExplainer
     (exact, fast). Pass --shap_model mlp to instead train the paper's
     LightweightMLP_IDS on the same split and explain IT with
     shap.GradientExplainer -- the research plan explicitly allows either
     ("MLP 또는 baseline 모델의 예측에 대한 feature contribution을 SHAP을
     이용하여 분석").

The two rankings are compared (Spearman rank correlation + top-k overlap)
and written to <output_dir>/<name>_feature_ranking.csv. For each --topk
value, this script also writes a ready-to-use reduced CSV per method
(<name>_rf_top<k>.csv / <name>_shap_top<k>.csv) containing only the
selected feature columns plus the original label column, so it can be fed
straight back into train.py to compare Baseline vs FS-k detection
performance (research plan Section 2.3-(3), Table "FS-1/FS-2/FS-3").

Usage:
    python feature_selection.py --csv path/to/dataset.csv --label_col Label \
        --categorical_cols protocol_type service flag \
        --name NSL-KDD --topk 10 20 30

    # SHAP computed on the MLP instead of the Random Forest:
    python feature_selection.py --csv path/to/dataset.csv --label_col Label \
        --name CICIDS2017 --shap_model mlp --mlp_epochs 30
"""

import argparse
import os

import numpy as np
import pandas as pd
import shap
import matplotlib.pyplot as plt
from scipy.stats import spearmanr
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from tqdm import tqdm

from preprocess import preprocess_full_pipeline


def rf_importance(X_train: np.ndarray, y_train: np.ndarray, seed: int,
                   n_estimators: int = 300, verbose: int = 1, n_jobs: int = 2,
                   max_depth: int | None = None):
    """
    Method 1: Random Forest feature importance (mean decrease in impurity).

    `verbose` is passed straight to RandomForestClassifier: with n_jobs != 1,
    sklearn/joblib print a "building tree K of N" line as each tree
    finishes, which is the only progress signal available for a long RF fit
    on a large dataset (there's no percentage-complete API to poll).

    `n_jobs` defaults to 2 to be considerate on a shared machine (e.g. a
    lab server) -- pass -1 explicitly to use every available core.

    `max_depth` defaults to None (sklearn's default: nodes expand until
    leaves are pure), which on a multi-million-row dataset can produce very
    large, deep trees -- driving up both RF fit memory and, especially,
    downstream SHAP TreeExplainer time (SHAP's exact tree algorithm scales
    with tree depth). Capping it (e.g. 20) keeps both tractable with only a
    minor effect on the resulting importance ranking.
    """
    rf = RandomForestClassifier(
        n_estimators=n_estimators, random_state=seed, n_jobs=n_jobs, verbose=verbose,
        max_depth=max_depth,
    )
    rf.fit(X_train, y_train)
    return rf, rf.feature_importances_


def _mean_abs_shap(shap_values) -> np.ndarray:
    """
    Normalize SHAP's various return shapes across versions/explainers into a
    single (n_features,) mean(|shap value|) importance vector, averaged over
    samples and (for multi-class) over classes.
    """
    if isinstance(shap_values, list):
        # older API: list of (n_samples, n_features) arrays, one per class
        stacked = np.stack([np.abs(sv) for sv in shap_values], axis=-1)
        return stacked.mean(axis=(0, 2))

    values = np.asarray(shap_values)
    if values.ndim == 3:
        # (n_samples, n_features, n_classes)
        return np.abs(values).mean(axis=(0, 2))
    return np.abs(values).mean(axis=0)


def _concat_shap_chunks(chunks):
    """Concatenate per-chunk shap_values() outputs back into one array/list,
    handling both the list-of-per-class-arrays and ndarray return shapes."""
    if isinstance(chunks[0], list):
        n_classes = len(chunks[0])
        return [np.concatenate([c[k] for c in chunks], axis=0) for k in range(n_classes)]
    return np.concatenate(chunks, axis=0)


def shap_importance_from_rf(rf: RandomForestClassifier, X_train: np.ndarray,
                             sample_size: int, seed: int, chunk_size: int = 50):
    """
    Method 2 (default target): SHAP on the Random Forest via TreeExplainer.

    The sample is explained in chunks (default 50 rows) under a tqdm
    progress bar instead of one single explainer.shap_values() call --
    SHAP gives no progress feedback of its own, and on a large/deep forest
    a single call can run for a long time with zero visible output.
    """
    rng = np.random.RandomState(seed)
    n = min(sample_size, X_train.shape[0])
    idx = rng.choice(X_train.shape[0], size=n, replace=False)
    X_sample = X_train[idx]

    explainer = shap.TreeExplainer(rf)
    chunks = [X_sample[i:i + chunk_size] for i in range(0, len(X_sample), chunk_size)]
    values_per_chunk = [
        explainer.shap_values(chunk)
        for chunk in tqdm(chunks, desc="SHAP (RF)", unit="chunk")
    ]
    return _mean_abs_shap(_concat_shap_chunks(values_per_chunk))


def shap_importance_from_mlp(X_train: np.ndarray, y_train: np.ndarray,
                              input_dim: int, num_classes: int,
                              sample_size: int, background_size: int,
                              epochs: int, seed: int, device):
    """Method 2 (alternative target): train the paper's MLP, then explain it."""
    import torch
    from model import LightweightMLP_IDS

    torch.manual_seed(seed)
    model = LightweightMLP_IDS(input_dim=input_dim, num_classes=num_classes).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=3e-3, weight_decay=1e-4)
    criterion = torch.nn.CrossEntropyLoss()

    X_train_t = torch.tensor(X_train, dtype=torch.float32, device=device)
    y_train_t = torch.tensor(y_train, dtype=torch.long, device=device)
    loader = torch.utils.data.DataLoader(
        torch.utils.data.TensorDataset(X_train_t, y_train_t),
        batch_size=128, shuffle=True,
    )

    model.train()
    for epoch in range(epochs):
        for xb, yb in loader:
            optimizer.zero_grad()
            loss = criterion(model(xb), yb)
            loss.backward()
            optimizer.step()

    model.eval()
    rng = np.random.RandomState(seed)
    bg_idx = rng.choice(X_train.shape[0], size=min(background_size, X_train.shape[0]), replace=False)
    sample_idx = rng.choice(X_train.shape[0], size=min(sample_size, X_train.shape[0]), replace=False)

    background = X_train_t[bg_idx]
    X_sample = X_train_t[sample_idx]

    explainer = shap.GradientExplainer(model, background)
    shap_values = explainer.shap_values(X_sample)
    return _mean_abs_shap(shap_values)


def rank_table(feature_names, rf_imp: np.ndarray, shap_imp: np.ndarray) -> pd.DataFrame:
    df = pd.DataFrame({
        "feature": feature_names,
        "rf_importance": rf_imp,
        "shap_importance": shap_imp,
    })
    df["rf_rank"] = df["rf_importance"].rank(ascending=False, method="first").astype(int)
    df["shap_rank"] = df["shap_importance"].rank(ascending=False, method="first").astype(int)
    return df.sort_values("rf_rank").reset_index(drop=True)


def compare_rankings(df: pd.DataFrame, topk_values: list[int]) -> dict:
    corr, pvalue = spearmanr(df["rf_importance"], df["shap_importance"])
    overlaps = {}
    rf_order = df.sort_values("rf_rank")["feature"].tolist()
    shap_order = df.sort_values("shap_rank")["feature"].tolist()
    for k in topk_values:
        k = min(k, len(df))
        rf_topk = set(rf_order[:k])
        shap_topk = set(shap_order[:k])
        overlaps[k] = len(rf_topk & shap_topk) / k
    return {"spearman_corr": corr, "spearman_pvalue": pvalue, "topk_overlap": overlaps}


def plot_top_features(df: pd.DataFrame, name: str, output_dir: str, top_n: int = 20):
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    for ax, col, title in (
        (axes[0], "rf_importance", "Random Forest"),
        (axes[1], "shap_importance", "SHAP"),
    ):
        top = df.sort_values(col, ascending=False).head(top_n)
        ax.barh(top["feature"][::-1], top[col][::-1])
        ax.set_title(f"{name} - {title} feature importance (top {top_n})")
        ax.set_xlabel("importance")
    plt.tight_layout()
    plt.savefig(f"{output_dir}/{name}_feature_importance.png", dpi=150)
    plt.close(fig)


def write_selected_csv(X_df: pd.DataFrame, y: np.ndarray, le, selected_cols: list[str],
                        label_col: str, out_path: str):
    """
    Write a reduced CSV (selected feature columns + original label text) that
    can be fed straight back into train.py to reproduce Baseline vs FS-k
    comparisons without re-deriving the categorical encoding.
    """
    df_out = X_df[selected_cols].copy()
    df_out[label_col] = le.inverse_transform(y)
    df_out.to_csv(out_path, index=False)


def run_feature_selection(
    csv_path: str,
    label_col: str,
    categorical_cols=None,
    categorical_mode: str = "onehot",
    topk: list[int] | None = None,
    shap_model: str = "rf",
    rf_n_estimators: int = 300,
    rf_verbose: int = 1,
    rf_n_jobs: int = 2,
    rf_max_depth: int | None = None,
    shap_sample_size: int = 500,
    shap_background_size: int = 100,
    mlp_epochs: int = 30,
    test_size: float = 0.2,
    seed: int = 42,
    dataset_name: str = "dataset",
    output_dir: str = ".",
):
    topk = topk or [10, 20, 30]

    os.makedirs(output_dir, exist_ok=True)

    df = pd.read_csv(csv_path)
    X_df, y, le = preprocess_full_pipeline(
        df, label_col, categorical_cols, categorical_mode=categorical_mode
    )
    feature_names = X_df.columns.tolist()

    # Same split as train.py so importance is computed on TRAIN only
    # (research plan Section 2.3-(4): no leakage into val/test).
    X_train_raw, X_test_raw, y_train, y_test = train_test_split(
        X_df.values.astype(np.float32), y,
        test_size=test_size, random_state=seed, stratify=y,
    )
    scaler = MinMaxScaler()
    X_train = scaler.fit_transform(X_train_raw)

    rf, rf_imp = rf_importance(X_train, y_train, seed=seed, n_estimators=rf_n_estimators,
                                verbose=rf_verbose, n_jobs=rf_n_jobs, max_depth=rf_max_depth)

    if shap_model == "rf":
        shap_imp = shap_importance_from_rf(rf, X_train, sample_size=shap_sample_size, seed=seed)
    elif shap_model == "mlp":
        import torch
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        shap_imp = shap_importance_from_mlp(
            X_train, y_train, input_dim=X_train.shape[1], num_classes=len(le.classes_),
            sample_size=shap_sample_size, background_size=shap_background_size,
            epochs=mlp_epochs, seed=seed, device=device,
        )
    else:
        raise ValueError('shap_model must be "rf" or "mlp"')

    df_rank = rank_table(feature_names, rf_imp, shap_imp)
    df_rank.to_csv(f"{output_dir}/{dataset_name}_feature_ranking.csv", index=False)
    plot_top_features(df_rank, dataset_name, output_dir)

    comparison = compare_rankings(df_rank, topk)
    print(f"\n=== {dataset_name} feature importance comparison (RF vs SHAP[{shap_model}]) ===")
    print(f"Spearman correlation = {comparison['spearman_corr']:.4f} "
          f"(p={comparison['spearman_pvalue']:.2e})")
    for k, overlap in comparison["topk_overlap"].items():
        print(f"Top-{k} overlap = {overlap:.2%}")

    rf_order = df_rank.sort_values("rf_rank")["feature"].tolist()
    shap_order = df_rank.sort_values("shap_rank")["feature"].tolist()
    for k in topk:
        k = min(k, len(feature_names))
        write_selected_csv(X_df, y, le, rf_order[:k], label_col,
                            f"{output_dir}/{dataset_name}_rf_top{k}.csv")
        write_selected_csv(X_df, y, le, shap_order[:k], label_col,
                            f"{output_dir}/{dataset_name}_shap_top{k}.csv")
        print(f"Wrote {dataset_name}_rf_top{k}.csv / {dataset_name}_shap_top{k}.csv")

    return df_rank, comparison


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
    )
    parser.add_argument(
        "--topk", nargs="*", type=int, default=[10, 20, 30],
        help="Feature-subset sizes to write out (Top-k per research plan 2.3-(3))",
    )
    parser.add_argument(
        "--shap_model", choices=["rf", "mlp"], default="rf",
        help="Model SHAP explains: the Random Forest (default, fast/exact) "
             "or the paper's LightweightMLP_IDS (research plan allows either)",
    )
    parser.add_argument("--rf_n_estimators", type=int, default=300)
    parser.add_argument("--rf_verbose", type=int, default=1,
                         help="RandomForestClassifier verbosity (0=silent, "
                              "1=per-tree progress via joblib, 2=more detail)")
    parser.add_argument("--rf_n_jobs", type=int, default=2,
                         help="CPU cores for the Random Forest fit. Defaults to 2 to "
                              "be considerate on a shared machine; pass -1 for all cores.")
    parser.add_argument("--rf_max_depth", type=int, default=None,
                         help="Cap tree depth (default: unlimited, sklearn's default). "
                              "Recommended on large datasets -- unbounded trees blow up "
                              "RF memory and make SHAP TreeExplainer very slow.")
    parser.add_argument("--shap_sample_size", type=int, default=500)
    parser.add_argument("--shap_background_size", type=int, default=100)
    parser.add_argument("--mlp_epochs", type=int, default=30,
                         help="Epochs to train the MLP before SHAP, if --shap_model mlp")
    parser.add_argument("--test_size", type=float, default=0.2)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--name", default="dataset")
    parser.add_argument("--output_dir", default=".")
    args = parser.parse_args()

    run_feature_selection(
        csv_path=args.csv,
        label_col=args.label_col,
        categorical_cols=args.categorical_cols,
        categorical_mode=args.categorical_mode,
        topk=args.topk,
        shap_model=args.shap_model,
        rf_n_estimators=args.rf_n_estimators,
        rf_verbose=args.rf_verbose,
        rf_n_jobs=args.rf_n_jobs,
        rf_max_depth=args.rf_max_depth,
        shap_sample_size=args.shap_sample_size,
        shap_background_size=args.shap_background_size,
        mlp_epochs=args.mlp_epochs,
        test_size=args.test_size,
        seed=args.seed,
        dataset_name=args.name,
        output_dir=args.output_dir,
    )
