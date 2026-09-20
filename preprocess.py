from __future__ import annotations
"""
Preprocessing pipeline matching Section 3.1 of the paper:
  1. Remove rows with missing values
  2. Remove rows with non-finite (inf/-inf) values
  3. Label-encode the target column (LabelEncoder -> {0, ..., c-1})
  4. One-hot encode categorical feature columns
  5. Min-Max normalize all numeric features to [0, 1]   (Eq. 1)

NOTE (ambiguity to confirm with authors):
  - The paper's Algorithm 1 shows preprocessing happening once on the full
    (X, Y) BEFORE the 80/20 train/test split is mentioned in Section 4.1.
    If Min-Max scaling is literally fit on the full dataset (train+test)
    before splitting, that is a (very common, but technically leaky) choice.
    This implementation instead fits the scaler on the TRAIN split only and
    applies it to test, which is best practice. Worth asking which the
    authors actually did, since it can affect the reported accuracy slightly.
  - Exact list of categorical columns per dataset (e.g., NSL-KDD's
    protocol_type / service / flag columns) and the resulting final feature
    dimension d are not stated in the paper -- fill in `categorical_cols`
    per dataset accordingly.
"""

import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder, MinMaxScaler


def clean_dataframe(df: pd.DataFrame) -> pd.DataFrame:
    """Steps 1-2: drop missing values and non-finite rows."""
    df = df.dropna()
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    finite_mask = np.isfinite(df[numeric_cols]).all(axis=1)
    return df.loc[finite_mask].reset_index(drop=True)


def encode_labels(y_raw: np.ndarray) -> tuple[np.ndarray, LabelEncoder]:
    """Step 3: LabelEncoder on the target column."""
    le = LabelEncoder()
    y = le.fit_transform(y_raw)
    return y, le


def one_hot_encode(X_df: pd.DataFrame, categorical_cols: list[str]) -> pd.DataFrame:
    """Step 4: one-hot encode specified categorical columns."""
    if not categorical_cols:
        return X_df
    X_df = pd.get_dummies(X_df, columns=categorical_cols)
    # pandas >= 2.0 returns bool dtype for dummy columns; cast to numeric
    bool_cols = X_df.select_dtypes(include=["bool"]).columns
    X_df[bool_cols] = X_df[bool_cols].astype(np.float32)
    return X_df


def drop_categorical(X_df: pd.DataFrame, categorical_cols: list[str]) -> pd.DataFrame:
    """
    Alternative to one_hot_encode: simply drop the categorical columns
    instead of encoding them. Useful for testing whether a paper's much
    lower reported feature dimension implies the categorical text columns
    (e.g. NSL-KDD's protocol_type/service/flag) were excluded entirely
    rather than one-hot encoded.
    """
    cols_present = [c for c in categorical_cols if c in X_df.columns]
    return X_df.drop(columns=cols_present)


def fit_minmax(X_train: np.ndarray):
    """Step 5 (fit on train only -- see NOTE above)."""
    scaler = MinMaxScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    return X_train_scaled, scaler


def preprocess_full_pipeline(
    df: pd.DataFrame,
    label_col: str,
    categorical_cols: list[str] | None = None,
    categorical_mode: str = "onehot",
):
    """
    Convenience wrapper running steps 1-4 (cleaning, label encoding,
    categorical handling). Min-Max scaling (step 5) is deliberately left to
    be applied AFTER the train/test split -- see train.py.

    Args:
        categorical_mode: "onehot" (default, Section 3.1 as literally
            described) or "drop" (excludes categorical_cols entirely --
            useful for matching a paper's much smaller reported feature
            dimension; see NOTE at top of this file).

    Returns:
        X_df   : feature dataframe (numeric, not yet scaled)
        y      : integer-encoded label array
        le     : fitted LabelEncoder
    """
    categorical_cols = categorical_cols or []
    if categorical_mode not in ("onehot", "drop"):
        raise ValueError('categorical_mode must be "onehot" or "drop"')

    df = clean_dataframe(df)
    y_raw = df[label_col].values
    X_df = df.drop(columns=[label_col])

    y, le = encode_labels(y_raw)
    if categorical_mode == "onehot":
        X_df = one_hot_encode(X_df, categorical_cols)
    else:
        X_df = drop_categorical(X_df, categorical_cols)

    # ensure everything is numeric at this point
    non_numeric = X_df.select_dtypes(exclude=[np.number]).columns.tolist()
    if non_numeric:
        raise ValueError(
            f"Non-numeric columns remain after encoding: {non_numeric}. "
            f"Add them to `categorical_cols` or drop them explicitly."
        )

    return X_df, y, le
