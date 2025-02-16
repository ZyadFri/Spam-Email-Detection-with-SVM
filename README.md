# Spam Email Detection with SVM

A comprehensive end-to-end project demonstrating how to build, evaluate, and deploy a Machine Learning model for spam detection using Flask. This repository includes data preprocessing, multiple model comparisons, and a final web application for real-time email classification.

---

## Table of Contents
1. [Project Overview](#project-overview)  
2. [Dataset](#dataset)  
3. [Data Preprocessing & Exploration](#data-preprocessing--exploration)  
4. [Model Building & Comparison](#model-building--comparison)  
5. [Chosen Model: SVM](#chosen-model-svm)  
6. [Project Structure](#project-structure)  
7. [Usage Instructions](#usage-instructions)  
8. [Detailed Steps](#detailed-steps)  
9. [Future Improvements](#future-improvements)  
10. [License](#license)

---

## Project Overview
This project classifies emails as **spam** or **non-spam (ham)**. It begins with loading and exploring the dataset, followed by feature engineering, training multiple machine learning models, and finally deploying the best model (SVM) as a Flask web application. Users can paste email text into a web form and receive instant predictions.

**Key Highlights**:
- Multiple models tested: **Naive Bayes**, **SVM**, **K-Nearest Neighbors (KNN)**, **Decision Tree (DT)**, and **Random Forest**.
- **SVM** achieved the highest accuracy (~97.61%) and was chosen for deployment.
- A **Flask** application serves predictions via a simple web interface (`index.html`).

---

## Dataset
The dataset (e.g., `spam_data.pkl` or CSV files) contains a collection of emails labeled as `spam` (1) or `ham` (0). It’s typically imbalanced, with more non-spam emails than spam.

**Key Points**:
- **Number of Spam Emails**: ~2,115  
- **Number of Non-Spam Emails**: ~9,185  
- **Imbalance**: ~4:1 ratio (ham:spam)

> *Why This Matters:* An imbalanced dataset can bias models toward predicting the majority class (non-spam). Techniques such as resampling or class weighting may be applied if necessary.

---

## Data Preprocessing & Exploration
1. **Loading Data**: The raw data is read from a CSV file or from `spam_data.pkl`.  
2. **Cleaning & Tokenization**: Common steps include lowercasing text, removing punctuation, and optionally removing stopwords.  
3. **Exploratory Data Analysis (EDA)**: Check class distribution and visualize common words.  
4. **Feature Extraction**: A **CountVectorizer** (saved as `countvectorizer.pkl`) transforms email text into numeric vectors.

---

## Model Building & Comparison
Multiple machine learning models were trained and evaluated on a training set, then tested on a validation/test set. The following models were considered:

1. **Naive Bayes**  
2. **SVM**  
3. **KNN**  
4. **Decision Tree**  
5. **Random Forest**

### Performance Metrics
- **Accuracy** for quick comparison.  
- Additional metrics like **precision**, **recall**, and **F1-score** can provide deeper insight, especially for imbalanced data.

**Approximate Accuracy Results**:
- **Naive Bayes**: 94.66%  
- **SVM**: 97.61%  
- **KNN**: 90.97%  
- **Decision Tree**: 96.02%  
- **Random Forest**: 96.17%

---

## Chosen Model: SVM
Because SVM yielded the **highest accuracy** (around 97.61%), it was chosen for final deployment.

1. **Vectorization**: Input email text is transformed using the same `CountVectorizer` from training.  
2. **Classification**: The SVM model (with RBF kernel, for instance) predicts spam or ham.

---

## Project Structure

ML_DEPLOYMENT/ │ 
   ├── models/ │
               ├── clf_NaiveBaised.pkl # Example alternative model (Naive Bayes) │ 
               ├── countvectorizer.pkl # Vectorizer for text features │ 
               └── spam_data.pkl # Serialized dataset or final training data │ 
   ├── static/ │ └── shield-logo.png # Example static asset (image) │ 
   ├── templates/ │ └── index.html # Frontend HTML for user input and display │ 
   ├── app.py # Main Flask application ├── requirements.txt # Dependencies
   ├── utils.py # Helper functions (load model, predict, etc.)
   └── README.md # Project documentation

   
- **`models/`**: Stores serialized models (`.pkl` files) and possibly the dataset if saved as a pickle.  
- **`static/`**: Holds images, CSS, or JavaScript files for the web interface.  
- **`templates/`**: Flask’s default folder for HTML templates.  
- **`app.py`**: Entry point for the Flask app.  
- **`utils.py`**: Helper module for loading the model, preprocessing text, or making predictions.  
- **`requirements.txt`**: Lists Python dependencies.

---

## Usage Instructions

1. **Clone or Download** this repository.  
2. **Create a Virtual Environment** (recommended):
   ```bash
   conda create -n spam_env python=3.9
   conda activate spam_env

**Install Dependencies**:
```bash
pip install -r requirements.txt
