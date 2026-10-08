"""
Module: trainer.py
Responsibility: Train multiple classifiers, evaluate them, and return the best one.
"""

from sklearn.naive_bayes import MultinomialNB
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report
from scipy.sparse import spmatrix
import pandas as pd


# Registry of all candidate models
MODEL_REGISTRY: dict = {
    "Multinomial Naive Bayes": MultinomialNB(),
    "Logistic Regression": LogisticRegression(
        max_iter=1000, solver="liblinear", random_state=42
    ),
    "Support Vector Machine (SVC)": SVC(random_state=42),
    "Random Forest": RandomForestClassifier(n_estimators=200, random_state=42),
}


def train_and_evaluate(
    X_train_vec: spmatrix,
    X_test_vec: spmatrix,
    y_train: pd.Series,
    y_test: pd.Series,
) -> tuple:
    """
    Train each model in MODEL_REGISTRY, evaluate on test set, and select the best.

    Args:
        X_train_vec: Vectorised training features.
        X_test_vec: Vectorised test features.
        y_train: Training labels.
        y_test: Test labels.

    Returns:
        Tuple of:
            - results (dict[str, float]): model name → accuracy
            - best_model: Fitted model with highest accuracy
            - best_model_name (str): Name of the best model
    """
    results: dict[str, float] = {}
    best_model = None
    best_accuracy = 0.0
    best_model_name = ""

    print("\n[trainer] ── Model Training and Evaluation ──")

    for name, model in MODEL_REGISTRY.items():
        print(f"\n[trainer] Training: {name} ...")
        model.fit(X_train_vec, y_train)

        y_pred = model.predict(X_test_vec)
        accuracy = accuracy_score(y_test, y_pred)
        results[name] = accuracy

        print(f"[trainer] {name} Accuracy: {accuracy:.4f}")
        print(classification_report(y_test, y_pred, target_names=["ham", "spam"]))

        if accuracy > best_accuracy:
            best_accuracy = accuracy
            best_model = model
            best_model_name = name

    print(
        f"\n[trainer] ✓ Best model: {best_model_name} "
        f"(Accuracy: {best_accuracy:.4f})"
    )
    return results, best_model, best_model_name
