# SpamShield — SMS Spam Detection System

## 1. Project Overview

SpamShield is a machine learning-based SMS classification application that identifies potentially unwanted or spam messages.

It uses Natural Language Processing (NLP), TF-IDF feature extraction, and a calibrated Logistic Regression model to classify messages as **Spam** or **Not Spam (Ham)**.

The application also provides:
- Estimated spam and legitimate-message probabilities.
- Explainable predictions using TF-IDF feature contributions.
- Rule-based indicators for potentially suspicious message content.
- An interactive web interface built with Streamlit.

**Important:** SpamShield is an educational SMS spam detection project. It is not a comprehensive phishing, smishing, or cybersecurity protection system.

## 2. Project Objectives

- Preprocess and analyze SMS text data.
- Convert text into numerical features using TF-IDF.
- Train and compare machine learning algorithms.
- Improve probability estimates through model calibration.
- Explain predictions using feature contributions.
- Identify predefined suspicious message patterns.
- Build an interactive Streamlit application.

## 3. Technologies Used

- Python
- Pandas
- NumPy
- Scikit-learn
- Matplotlib
- Seaborn
- Streamlit
- Joblib

## 4. Dataset

The original classification experiments use the UCI SMS Spam Collection dataset.

The original dataset contains 5,574 SMS messages.

After preprocessing and duplicate removal, 5,169 records were retained:

| Category | Records |
|---|---:|
| Ham | 4,516 |
| Spam | 653 |
| **Total** | **5,169** |

Label encoding:

- Ham = 0
- Spam = 1

Dataset source: [UCI SMS Spam Collection](https://archive.ics.uci.edu/dataset/228/sms+spam+collection)

## 5. Data Preprocessing

The preprocessing workflow includes:

- Loading the SMS dataset.
- Removing missing records.
- Removing duplicate records.
- Retaining valid Ham and Spam labels.
- Converting labels into numerical values.

## 6. Feature Extraction

TF-IDF (Term Frequency-Inverse Document Frequency) converts SMS messages into numerical features.

The feature extraction configuration includes:

- Lowercase conversion.
- English stop-word removal.
- Unigrams and bigrams.

## 7. Machine Learning Models

The original experiments compared two classification algorithms:

1. Multinomial Naive Bayes
2. Logistic Regression

Logistic Regression was selected based on its higher Spam F1-score in the original experiment.

### Original baseline comparison

| Metric | Naive Bayes | Logistic Regression |
|---|---:|---:|
| Accuracy | 95.36% | 97.49% |
| Precision | 100.00% | 91.34% |
| Recall | 63.36% | 88.55% |
| F1-score | 77.57% | 89.92% |

These are results from the original baseline experiment, not the final calibrated model evaluation.

## 8. Probability Calibration

The upgraded application uses a calibrated Logistic Regression pipeline.

Calibration was evaluated using a stratified train-validation-test split:

- Training: 60%
- Validation: 20%
- Final testing: 20%

The calibration process compared:

- Original Logistic Regression
- Sigmoid calibration
- Isotonic calibration

Sigmoid calibration was selected for the application.

### Final holdout test results

| Metric | Original model | Sigmoid calibrated |
|---|---:|---:|
| Spam precision | 85.92% | 92.19% |
| Spam recall | 93.13% | 90.08% |
| Spam F1-score | 89.38% | 91.12% |
| Brier score | 0.0470 | 0.0179 |
| Log loss | 0.2207 | 0.0744 |

The calibrated model achieved improved probability-quality metrics on this test split.

These results are dataset-specific and do not guarantee equivalent performance on real-world messages.

## 9. Explainable Predictions

SpamShield provides feature-level explanations for its predictions.

It displays words or terms that contribute positively or negatively to the spam classification.

Examples include:

- Spam indicators: terms that push the prediction toward Spam.
- Legitimate indicators: terms that push the prediction toward Ham.

The displayed contributions are calculated using the first fitted Logistic Regression estimator inside the calibrated model.

They are explanatory indicators, not proof that a particular message is fraudulent or legitimate.

## 10. Suspicious Content Indicators

The application includes a supplementary rule-based warning system.

It checks for predefined patterns in four categories:

- Account security
- Sensitive information requests
- Prize or reward language
- Web links

These warnings operate independently of the machine learning prediction.

A warning does not confirm that a message is malicious. Similarly, the absence of a warning does not guarantee safety.

## 11. Web Application

The Streamlit interface allows users to:

1. Enter an SMS message.
2. Analyze the message.
3. View the Spam or Not Spam prediction.
4. View the estimated classification probability.
5. Review suspicious content warnings.
6. Examine the features contributing to the prediction.

The application uses a classification threshold of 0.5.

Model probabilities are estimates, not guarantees.

## 12. Project Structure

```text
SMS-Spam-Detector/
│
├── data/
│   └── SMSSpamCollection
│
├── models/
│   ├── spam_model.pkl
│   ├── spam_model_sigmoid.pkl
│   └── tfidf_vectorizer.pkl
│
├── reports/
│   ├── confusion_matrix.png
│   └── model_comparison.csv
│
├── screenshots/
│
├── src/
│   ├── app.py
│   ├── explain.py
│   ├── preprocess.py
│   └── train.py
│
├── .python-version
├── .gitignore
├── requirements.txt
└── README.md
```

## 13. Installation and Execution

### Step 1: Clone the repository

```bash
git clone https://github.com/Sourav-Sambyal/SMS-Spam-Detector.git
cd SMS-Spam-Detector
```

### Step 2: Create a virtual environment

```bash
python -m venv .venv
```

Activate it on Windows:

```powershell
.venv\Scripts\Activate.ps1
```

### Step 3: Install dependencies

```bash
pip install -r requirements.txt
```

### Step 4: Launch the application

```bash
python -m streamlit run src/app.py
```

### Step 5: Open the application

Visit:

```text
http://localhost:8501
```

The application loads the saved calibrated model from the `models` directory.

## 14. Limitations

- The model is trained and evaluated on a specific SMS dataset.
- Real-world messages may contain language and patterns absent from the training data.
- The rule-based warning system uses predefined phrases and can produce false positives or miss suspicious content.
- Feature contributions do not establish the actual intent of a message.
- SpamShield is not a comprehensive phishing or smishing detector.
- A Not Spam prediction should never be treated as proof that a message is safe.

## 15. Conclusion

SpamShield demonstrates the application of Natural Language Processing and machine learning to SMS spam classification.

The project combines TF-IDF feature extraction, Logistic Regression, probability calibration, explainable predictions, and supplementary suspicious-content indicators in an interactive Streamlit application.

It provides an educational demonstration of how machine learning can support SMS classification while highlighting the limitations of automated predictions.