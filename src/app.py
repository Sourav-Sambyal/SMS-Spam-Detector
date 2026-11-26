
from pathlib import Path
import joblib
import streamlit as st

# Project paths
BASE_DIR = Path(__file__).resolve().parent.parent
MODEL_PATH = BASE_DIR / "models" / "spam_model.pkl"
VECTORIZER_PATH = BASE_DIR / "models" / "tfidf_vectorizer.pkl"

# Page configuration
st.set_page_config(
    page_title="SMS Spam Detector",
    page_icon="📩",
    layout="centered"
)

# Load saved model and vectorizer
@st.cache_resource
def load_model():
    model = joblib.load(MODEL_PATH)
    vectorizer = joblib.load(VECTORIZER_PATH)
    return model, vectorizer


model, vectorizer = load_model()

# Application interface
st.title("📩 SMS Spam Detector")
st.write(
    "Check whether an SMS message is legitimate "
    "or potentially spam using Machine Learning."
)

st.divider()

message = st.text_area(
    "Enter your SMS message:",
    placeholder="Example: Congratulations! You have won a free prize...",
    height=150
)

if st.button("🔍 Check Message", use_container_width=True):

    if not message.strip():
        st.warning("Please enter a message first.")

    else:
        message_tfidf = vectorizer.transform([message])

        prediction = model.predict(message_tfidf)[0]
        probabilities = model.predict_proba(message_tfidf)[0]

        spam_probability = probabilities[1]
        ham_probability = probabilities[0]

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

        st.caption(
            "Probabilities are model estimates, not guarantees "
            "that a message is safe or fraudulent."
        )

st.divider()

st.caption("Built with Python, Scikit-learn and Streamlit.")
