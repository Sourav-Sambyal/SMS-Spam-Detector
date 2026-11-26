
# SMS Spam Detector

## 1. Project Overview

SMS Spam Detector is a machine learning-based application that classifies SMS messages as Spam or Not Spam (Ham).

The project uses Natural Language Processing (NLP), TF-IDF vectorization, and supervised machine learning algorithms to identify potentially fraudulent or unwanted messages.

## 2. Objectives

- Analyze and preprocess SMS text data.
- Convert text into numerical features using TF-IDF.
- Train and compare two machine learning algorithms.
- Evaluate model performance using standard classification metrics.
- Build an interactive web application using Streamlit.

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

The project uses the UCI SMS Spam Collection dataset.

The original dataset contains 5,574 SMS messages:
- Ham (legitimate messages): 4,827
- Spam messages: 747

After cleaning and removing duplicate records, 5,169 records were retained:
- Ham: 4,516
- Spam: 653

## 5. Data Preprocessing

The preprocessing process includes:
- Loading the SMS dataset.
- Removing missing records.
- Removing duplicate records.
- Keeping valid Ham and Spam labels.
- Converting labels into numerical values.

Label encoding:
- Ham = 0
- Spam = 1

## 6. Feature Extraction

TF-IDF (Term Frequency-Inverse Document Frequency) is used to convert SMS text into numerical features.

Configuration:
- Lowercase conversion enabled.
- English stop words removed.
- Unigrams and bigrams used.

## 7. Machine Learning Algorithms

Two classification algorithms were trained and compared:

1. Multinomial Naive Bayes
2. Logistic Regression

The dataset was divided into:
- Training set: 80%
- Testing set: 20%

Stratified splitting and random_state=42 were used.

## 8. Model Evaluation

The models were evaluated using accuracy, precision, recall, F1-score, classification report, and confusion matrix.

### Model Comparison

| Metric | Naive Bayes | Logistic Regression |
|---|---:|---:|
| Accuracy | 95.36% | 97.49% |
| Precision | 100.00% | 91.34% |
| Recall | 63.36% | 88.55% |
| F1-score | 77.57% | 89.92% |

Logistic Regression achieved the higher Spam F1-score and was selected as the final model.

## 9. Confusion Matrix

The Logistic Regression confusion matrix:

| Actual / Predicted | Ham | Spam |
|---|---:|---:|
| Ham | 892 | 11 |
| Spam | 15 | 116 |

The model correctly classified 892 legitimate messages and 116 spam messages in the test set.

## 10. Web Application

A Streamlit web application allows users to:
- Enter an SMS message.
- Classify it as Spam or Not Spam.
- View the model's estimated prediction probability.

The probability is a model estimate and does not guarantee that a message is safe or fraudulent.

## 11. Project Structure

SMS-Spam-Detector/
|
|-- data/
|   |-- SMSSpamCollection
|
|-- models/
|   |-- spam_model.pkl
|   |-- tfidf_vectorizer.pkl
|
|-- reports/
|   |-- confusion_matrix.png
|   |-- model_comparison.csv
|
|-- src/
|   |-- app.py
|   |-- preprocess.py
|   |-- train.py
|
|-- requirements.txt
|-- README.md

## 12. How to Run the Project

### Step 1: Install dependencies

pip install -r requirements.txt

### Step 2: Train the models

python src/train.py

### Step 3: Launch the web application

python -m streamlit run src/app.py

### Step 4: Open the local URL

http://localhost:8501

## 13. Conclusion

The SMS Spam Detector demonstrates how Natural Language Processing and machine learning can be used to classify SMS messages.

Among the two evaluated algorithms, Logistic Regression achieved the higher Spam F1-score of 89.92% and was selected as the final model.

The project combines data preprocessing, feature extraction, model training, evaluation, and deployment through an interactive Streamlit application.
