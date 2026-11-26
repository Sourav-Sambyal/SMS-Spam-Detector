
from pathlib import Path
import joblib
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    classification_report,
    confusion_matrix
)

# Project paths
BASE_DIR = Path(__file__).resolve().parent.parent
DATA_PATH = BASE_DIR / "data" / "SMSSpamCollection"
MODEL_DIR = BASE_DIR / "models"
REPORT_DIR = BASE_DIR / "reports"

MODEL_DIR.mkdir(exist_ok=True)
REPORT_DIR.mkdir(exist_ok=True)


def load_data():
    """Load and clean the SMS Spam Collection dataset."""

    df = pd.read_csv(
        DATA_PATH,
        sep="\t",
        header=None,
        names=["label", "message"],
        encoding="utf-8"
    )

    df = df.dropna()
    df = df.drop_duplicates()
    df = df[df["label"].isin(["ham", "spam"])]

    return df


def main():
    df = load_data()

    X = df["message"]
    y = df["label"].map({"ham": 0, "spam": 1})

    # Split data into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.20,
        random_state=42,
        stratify=y
    )

    # Convert text into TF-IDF features
    vectorizer = TfidfVectorizer(
        lowercase=True,
        stop_words="english",
        ngram_range=(1, 2)
    )

    X_train_tfidf = vectorizer.fit_transform(X_train)
    X_test_tfidf = vectorizer.transform(X_test)

    # Compare two machine learning models
    models = {
        "Naive Bayes": MultinomialNB(),
        "Logistic Regression": LogisticRegression(
            max_iter=1000,
            class_weight="balanced",
            random_state=42
        )
    }

    results = []
    best_f1 = -1
    best_model_name = None
    best_model = None

    for name, model in models.items():

        model.fit(X_train_tfidf, y_train)
        predictions = model.predict(X_test_tfidf)

        accuracy = accuracy_score(y_test, predictions)
        precision = precision_score(
            y_test, predictions, zero_division=0
        )
        recall = recall_score(
            y_test, predictions, zero_division=0
        )
        f1 = f1_score(
            y_test, predictions, zero_division=0
        )

        results.append({
            "Model": name,
            "Accuracy": accuracy,
            "Precision": precision,
            "Recall": recall,
            "F1 Score": f1
        })

        print(f"\n{'=' * 45}")
        print(name)
        print(f"{'=' * 45}")
        print(f"Accuracy:  {accuracy:.4f}")
        print(f"Precision: {precision:.4f}")
        print(f"Recall:    {recall:.4f}")
        print(f"F1 Score:  {f1:.4f}")

        print("\nClassification Report:")
        print(classification_report(
            y_test,
            predictions,
            target_names=["Ham", "Spam"],
            zero_division=0
        ))

        cm = confusion_matrix(y_test, predictions)

        print("Confusion Matrix:")
        print(cm)

        # Generate confusion matrix for Logistic Regression
        if name == "Logistic Regression":

            plt.figure(figsize=(6, 4))

            sns.heatmap(
                cm,
                annot=True,
                fmt="d",
                cmap="Blues",
                xticklabels=["Ham", "Spam"],
                yticklabels=["Ham", "Spam"]
            )

            plt.title("SMS Spam Detector - Confusion Matrix")
            plt.xlabel("Predicted Label")
            plt.ylabel("Actual Label")
            plt.tight_layout()

            plt.savefig(
                REPORT_DIR / "confusion_matrix.png",
                dpi=300
            )

            plt.close()

            print("Confusion matrix saved successfully!")

        # Select the model with the highest Spam F1-score
        if f1 > best_f1:
            best_f1 = f1
            best_model_name = name
            best_model = model

    # Save comparison results
    results_df = pd.DataFrame(results)

    results_df.to_csv(
        REPORT_DIR / "model_comparison.csv",
        index=False
    )

    # Save the best model and its vectorizer
    joblib.dump(
        best_model,
        MODEL_DIR / "spam_model.pkl"
    )

    joblib.dump(
        vectorizer,
        MODEL_DIR / "tfidf_vectorizer.pkl"
    )

    print("\nBest Model:", best_model_name)
    print("Best F1 Score:", round(best_f1, 4))
    print("\nModel and vectorizer saved successfully!")


if __name__ == "__main__":
    main()
