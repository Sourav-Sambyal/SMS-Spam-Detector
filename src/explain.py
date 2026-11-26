
import numpy as np


def explain_prediction(message, model, vectorizer, top_n=5):
    """
    Explain a prediction using actual TF-IDF feature contributions.
    """

    # Convert the message into TF-IDF features
    message_tfidf = vectorizer.transform([message])

    # Get the Logistic Regression coefficients
    coefficients = model.coef_[0]

    # Get feature names from the vectorizer
    feature_names = vectorizer.get_feature_names_out()

    # Calculate each feature's contribution
    contributions = message_tfidf.toarray()[0] * coefficients

    # Find features that push towards Spam
    spam_indices = np.argsort(contributions)[::-1]

    spam_features = [
        (feature_names[i], round(float(contributions[i]), 4))
        for i in spam_indices
        if contributions[i] > 0
    ][:top_n]

    # Find features that push towards Ham
    ham_indices = np.argsort(contributions)

    ham_features = [
        (feature_names[i], round(float(contributions[i]), 4))
        for i in ham_indices
        if contributions[i] < 0
    ][:top_n]

    return {
        "spam_indicators": spam_features,
        "ham_indicators": ham_features
    }
