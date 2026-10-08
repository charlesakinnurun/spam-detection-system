"""
Module: visualiser.py
Responsibility: All plotting functions — class distribution & model comparison.
"""

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns


def plot_class_distribution(df: pd.DataFrame) -> None:
    """
    Plot a count plot showing the Ham vs Spam distribution.

    Args:
        df: DataFrame with a 'label' column.
    """
    plt.figure(figsize=(7, 5))
    sns.countplot(x="label", data=df)
    plt.title("Distribution of Ham vs Spam Messages")
    plt.xlabel("Message Type")
    plt.ylabel("Count")
    plt.tight_layout()
    plt.show()
    print("[visualiser] Class distribution plot displayed.")


def plot_model_comparison(results: dict[str, float]) -> None:
    """
    Plot a horizontal bar chart comparing model accuracies.

    Args:
        results: Dict mapping model name → accuracy score.
    """
    results_df = pd.DataFrame(results.items(), columns=["Model", "Accuracy"])
    results_df = results_df.sort_values(by="Accuracy", ascending=False)

    plt.figure(figsize=(10, 6))
    sns.barplot(x="Accuracy", y="Model", data=results_df)
    plt.title("Comparison of Classification Model Accuracy for Spam Detection")
    plt.xlim(0.9, 1.0)
    plt.xlabel("Accuracy Score")
    plt.ylabel("Model")
    plt.tight_layout()
    plt.show()
    print("[visualiser] Model comparison plot displayed.")
