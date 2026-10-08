"""
Module: predictor.py
Responsibility: Provide a reusable predict function and an interactive CLI loop.
"""

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import LabelEncoder


def predict_message(
    message: str,
    model,
    vectorizer: TfidfVectorizer,
    label_encoder: LabelEncoder,
) -> str:
    """
    Predict whether a single text message is 'ham' or 'spam'.

    Args:
        message: Raw text message to classify.
        model: A fitted scikit-learn classifier.
        vectorizer: The fitted TF-IDF vectorizer.
        label_encoder: The fitted LabelEncoder used during preprocessing.

    Returns:
        Prediction label as a string ('ham' or 'spam').
    """
    message_vec = vectorizer.transform([message])
    prediction_encoded = model.predict(message_vec)
    prediction_label = label_encoder.inverse_transform(prediction_encoded)[0]
    return prediction_label


def run_interactive_predictor(
    model,
    vectorizer: TfidfVectorizer,
    label_encoder: LabelEncoder,
    model_name: str,
) -> None:
    """
    Launch an interactive CLI loop that classifies user-supplied messages.

    Args:
        model: A fitted scikit-learn classifier.
        vectorizer: The fitted TF-IDF vectorizer.
        label_encoder: The fitted LabelEncoder.
        model_name: Human-readable name of the model in use.
    """
    if model is None:
        print("[predictor] Error: No trained model available.")
        return

    print(f"\n[predictor] Interactive Prediction using: {model_name}")
    print("Enter a message to check if it's 'ham' or 'spam'. Type 'exit' to quit.\n")

    while True:
        user_input = input("Your message: ").strip()

        if user_input.lower() == "exit":
            print("[predictor] Exiting prediction tool. Goodbye!")
            break

        if not user_input:
            print("[predictor] Please enter a non-empty message.\n")
            continue

        predicted_label = predict_message(user_input, model, vectorizer, label_encoder)

        if predicted_label == "spam":
            print(
                f"→ [!!! SPAM !!!] — {model_name} flagged this as spam.\n"
            )
        else:
            print(
                f"→ [HAM] — {model_name} determined this is a legitimate message.\n"
            )
