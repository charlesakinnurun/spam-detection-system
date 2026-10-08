"""
main.py — Spam Detection Pipeline
==================================
Connects all modules in order:

  data_loader  →  preprocessor  →  visualiser
                      ↓
                  vectoriser
                      ↓
                   trainer   →  visualiser (model comparison)
                      ↓
                  predictor  (interactive CLI)
"""

from data_loader import load_data
from preprocessor import preprocess, get_features_and_target
from visualiser import plot_class_distribution, plot_model_comparison
from vectoriser import split_data, vectorise
from trainer import train_and_evaluate
from predictor import run_interactive_predictor


def run_pipeline(data_path: str = "src/spam.csv") -> None:
    """
    Execute the full spam-detection pipeline end-to-end.

    Args:
        data_path: Path to the raw CSV dataset.
    """

    # ── Step 1: Load raw data ──────────────────────────────────────────────
    df = load_data(filepath=data_path)

    # ── Step 2: Preprocess (deduplicate, encode labels) ────────────────────
    df, label_encoder = preprocess(df)

    # ── Step 3: Visualise class distribution ───────────────────────────────
    plot_class_distribution(df)

    # ── Step 4: Extract features and target ────────────────────────────────
    X, y = get_features_and_target(df)

    # ── Step 5: Train/test split ───────────────────────────────────────────
    X_train, X_test, y_train, y_test = split_data(X, y, test_size=0.2, random_state=42)

    # ── Step 6: TF-IDF vectorisation ───────────────────────────────────────
    X_train_vec, X_test_vec, tfidf_vectorizer = vectorise(X_train, X_test, max_features=5000)

    # ── Step 7: Train all models and select the best ───────────────────────
    results, best_model, best_model_name = train_and_evaluate(
        X_train_vec, X_test_vec, y_train, y_test
    )

    # ── Step 8: Visualise model comparison ─────────────────────────────────
    plot_model_comparison(results)

    # ── Step 9: Interactive prediction CLI ─────────────────────────────────
    run_interactive_predictor(
        model=best_model,
        vectorizer=tfidf_vectorizer,
        label_encoder=label_encoder,
        model_name=best_model_name,
    )


if __name__ == "__main__":
    run_pipeline(data_path="src/spam.csv")
