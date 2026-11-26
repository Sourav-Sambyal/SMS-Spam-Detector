
from pathlib import Path
import joblib
import streamlit as st

from src.explain import explain_prediction


# Project paths
BASE_DIR = Path(__file__).resolve().parent.parent
MODEL_PATH = BASE_DIR / "models" / "spam_model_sigmoid.pkl"

THRESHOLD = 0.5
def detect_suspicious_indicators(message):
    text = message.lower()

    warnings = []

    patterns = {
        "Account security": [
            "account suspended",
            "account blocked",
            "verify your account",
            "unusual activity",
            "account will be suspended"
        ],
        "Sensitive information request": [
            "share otp",
            "enter otp",
            "share your pin",
            "enter your pin",
            "password",
            "bank details",
            "credit card details"
        ],
        "Prize or reward language": [
            "you have won",
            "claim your prize",
            "claim your reward",
            "free prize",
            "lottery winner"
        ]
    }

    for category, phrases in patterns.items():
        if any(phrase in text for phrase in phrases):
            warnings.append(category)

    if "http://" in text or "https://" in text or "www." in text:
        warnings.append("Contains a web link")

    return warnings

# Page configuration
st.set_page_config(
    page_title="SpamShield",
    page_icon="🛡️",
    layout="centered"
)


# Load calibrated model
@st.cache_resource
def load_model():
    return joblib.load(MODEL_PATH)


model = load_model()


# Get matching vectorizer and classifier from calibrated pipeline
explanation_pipeline = model.calibrated_classifiers_[0].estimator

explanation_vectorizer = explanation_pipeline.named_steps["tfidf"]
explanation_model = explanation_pipeline.named_steps["classifier"]


# Application interface
st.title("🛡️ SpamShield")

st.write(
    "Analyze SMS messages for potential spam using "
    "Machine Learning and explainable predictions."
)

st.divider()

message = st.text_area(
    "Enter your SMS message:",
    placeholder="Example: Congratulations! You have won a free prize...",
    height=150
)


if st.button("🔍 Analyze Message", use_container_width=True):

    if not message.strip():
        st.warning("Please enter a message first.")

    else:
        # Pass raw text to the complete calibrated pipeline
        probabilities = model.predict_proba([message])[0]

        spam_class_index = list(model.classes_).index(1)
        ham_class_index = list(model.classes_).index(0)

        spam_probability = probabilities[spam_class_index]
        ham_probability = probabilities[ham_class_index]

        # Apply selected threshold
        prediction = int(spam_probability >= THRESHOLD)

        st.subheader("Prediction Result")

        if prediction == 1:
            st.error("🚨 This message is classified as SPAM.")
            st.write(
                f"Spam probability: **{spam_probability * 100:.2f}%**"
            )

        else:
            st.success("✅ This message is classified as NOT SPAM.")
            st.write(
                f"Legitimate message probability: "
                f"**{ham_probability * 100:.2f}%**"
            )
        # Independent rule-based safety indicators
        warnings = detect_suspicious_indicators(message)

        if warnings:
            st.warning("Potentially suspicious content detected.")

            for warning in warnings:
                st.write(f"- {warning}")

            st.caption(
                "These are rule-based indicators, not verified threats. "
                "A message may contain these patterns for legitimate reasons."
            )
        else:
            st.info("No predefined suspicious patterns detected. "
                    "This does not guarantee that the message is safe.")
        # Explain using the matching pipeline components
        explanation = explain_prediction(
            message,
            explanation_model,
            explanation_vectorizer
        )

        st.divider()
        st.subheader("Why did the model make this prediction?")

        spam_indicators = explanation["spam_indicators"]
        ham_indicators = explanation["ham_indicators"]

        col1, col2 = st.columns(2)

        with col1:
            st.markdown("**🚨 Spam indicators**")

            if spam_indicators:
                for feature, contribution in spam_indicators:
                    st.write(f"`{feature}` — +{contribution:.4f}")
            else:
                st.caption("No positive Spam contributions found.")

        with col2:
            st.markdown("**✅ Legitimate indicators**")

            if ham_indicators:
                for feature, contribution in ham_indicators:
                    st.write(f"`{feature}` — {contribution:.4f}")
            else:
                st.caption("No negative Ham contributions found.")

        st.caption(
            "Feature contributions explain the first fitted "
            "Logistic Regression estimator, not the entire "
            "calibrated ensemble."
        )

        st.caption(
            "Probabilities are calibrated model estimates, not guarantees."
        )


st.divider()

st.caption(
    "SpamShield | Built with Python, Scikit-learn and Streamlit."
)
