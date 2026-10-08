"""
Module: vectoriser.py
Responsibility: Split data into train/test sets and apply TF-IDF vectorisation.
"""

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from scipy.sparse import spmatrix


def split_data(
    X: pd.Series,
    y: pd.Series,
    test_size: float = 0.2,
    random_state: int = 42,
) -> tuple:
    """
    Split feature and target arrays into train/test sets.

    Args:
        X: Feature series (raw messages).
        y: Target series (encoded labels).
        test_size: Fraction of data reserved for testing.
        random_state: Seed for reproducibility.

    Returns:
        Tuple of (X_train, X_test, y_train, y_test).
    """
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state, stratify=y
    )
    print(
        f"[vectoriser] Train size: {len(X_train)}, Test size: {len(X_test)}"
    )
    return X_train, X_test, y_train, y_test


def vectorise(
    X_train: pd.Series,
    X_test: pd.Series,
    max_features: int = 5000,
) -> tuple[spmatrix, spmatrix, TfidfVectorizer]:
    """
    Fit a TF-IDF vectorizer on training data, then transform both splits.

    Args:
        X_train: Raw training messages.
        X_test: Raw test messages.
        max_features: Maximum vocabulary size for TF-IDF.

    Returns:
        Tuple of (X_train_vec, X_test_vec, fitted TfidfVectorizer).
    """
    tfidf = TfidfVectorizer(stop_words="english", max_features=max_features)
    X_train_vec = tfidf.fit_transform(X_train)
    X_test_vec = tfidf.transform(X_test)
    print(
        f"[vectoriser] TF-IDF vocabulary size: {len(tfidf.vocabulary_)}"
    )
    return X_train_vec, X_test_vec, tfidf
