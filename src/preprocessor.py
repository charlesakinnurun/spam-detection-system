"""
Module: preprocessor.py
Responsibility: Clean the DataFrame, encode labels, define features & target.
"""

import pandas as pd
from sklearn.preprocessing import LabelEncoder


def preprocess(df: pd.DataFrame) -> tuple[pd.DataFrame, LabelEncoder]:
    """
    Encode labels and drop duplicate rows.

    Args:
        df: Raw DataFrame with columns ['label', 'message'].

    Returns:
        A tuple of:
            - Preprocessed DataFrame (adds 'label_encoded' column)
            - Fitted LabelEncoder instance
    """
    # Drop duplicate rows
    before = len(df)
    df = df.drop_duplicates().reset_index(drop=True)
    after = len(df)
    print(f"[preprocessor] Removed {before - after} duplicate rows. Remaining: {after}")

    # Check for missing values
    missing = df.isnull().sum()
    if missing.any():
        print(f"[preprocessor] Missing values found:\n{missing[missing > 0]}")
        df = df.dropna(subset=["message", "label"]).reset_index(drop=True)

    # Encode labels (ham → 0, spam → 1)
    le = LabelEncoder()
    df["label_encoded"] = le.fit_transform(df["label"])
    print(f"[preprocessor] Label classes: {list(le.classes_)}")

    return df, le


def get_features_and_target(df: pd.DataFrame) -> tuple:
    """
    Split the DataFrame into feature series and target series.

    Args:
        df: Preprocessed DataFrame.

    Returns:
        Tuple of (X, y) where X is the message series and y is the encoded label.
    """
    X = df["message"]
    y = df["label_encoded"]
    print(f"[preprocessor] Features shape: {X.shape}, Target shape: {y.shape}")
    return X, y
