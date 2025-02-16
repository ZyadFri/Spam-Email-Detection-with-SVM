# Spam Email Detection with SVM

A comprehensive end-to-end project demonstrating how to build, evaluate, and deploy a Machine Learning model for spam detection using Flask. This repository includes data preprocessing, multiple model evaluations, and deployment.

## Table of Contents
- [Project Overview](#project-overview)
- [Dataset](#dataset)
- [Data Preprocessing & Exploration](#data-preprocessing--exploration)
- [Model Building & Comparison](#model-building--comparison)
- [Chosen Model: SVM](#chosen-model-svm)
- [Project Structure](#project-structure)
- [Usage Instructions](#usage-instructions)
- [Detailed Steps](#detailed-steps)
- [Future Improvements](#future-improvements)
- [License](#license)

## Project Overview
This project classifies emails as spam or non-spam (ham). It begins with loading and exploring the dataset, followed by feature engineering, training multiple machine learning models, and finally deploying the chosen model.

### Key Highlights:
- Multiple models tested: Naive Bayes, SVM, K-Nearest Neighbors (KNN), Decision Tree (DT), and Random Forest.
- SVM achieved the highest accuracy (~97.61%) and was chosen for deployment.
- A Flask application serves predictions via a simple web interface (index.html).

## Dataset
The dataset (e.g., `spam_data.pkl` or CSV files) contains a collection of emails labeled as spam (1) or ham (0). It’s typically imbalanced, with more non-spam emails than spam.

### Key Points:
- **Number of Spam Emails**: ~2,115
- **Number of Non-Spam Emails**: ~9,185
- **Imbalance**: ~4:1 ratio (ham:spam)
- **Why This Matters**: An imbalanced dataset can bias models toward predicting the majority class (non-spam). Techniques such as resampling or class weighting may be applied if necessary.

## Data Preprocessing & Exploration
- **Loading Data**: The raw data is read from a CSV file or from `spam_data.pkl`.
- **Cleaning & Tokenization**: Common steps include lowercasing text, removing punctuation, and optionally removing stopwords.
- **Exploratory Data Analysis (EDA)**: Check class distribution and visualize common words.
- **Feature Extraction**: A `CountVectorizer` (saved as `countvectorizer.pkl`) transforms email text into numeric vectors.

## Model Building & Comparison
Multiple machine learning models were trained and evaluated on a training set, then tested on a validation/test set. The following models were considered:
- **Naive Bayes**
- **SVM**
- **KNN**
- **Decision Tree**
- **Random Forest**

### Performance Metrics
- **Accuracy** for quick comparison.
- Additional metrics like precision, recall, and F1-score can provide deeper insight, especially for imbalanced data.

#### Approximate Accuracy Results:
- **Naive Bayes**: 94.66%
- **SVM**: 97.61%
- **KNN**: 90.97%
- **Decision Tree**: 96.02%
- **Random Forest**: 96.17%

## Chosen Model: SVM
Because **SVM** yielded the highest accuracy (around 97.61%), it was chosen for final deployment.

- **Vectorization**: Input email text is transformed using the same `CountVectorizer` from training.
- **Classification**: The SVM model (with RBF kernel, for instance) predicts spam or ham.

## Project Structure
```
ML_DEPLOYMENT/
├── models/
│   ├── clf_NaiveBaised.pkl     # Example alternative model (Naive Bayes)
│   ├── countvectorizer.pkl     # Vectorizer for text features
│   └── spam_data.pkl           # Serialized dataset or final training data
├── static/
│   └── shield-logo.png         # Example static asset (image)
├── templates/
│   └── index.html              # Frontend HTML for user input and display
├── app.py                      # Main Flask application
├── requirements.txt            # Dependencies
├── utils.py                    # Helper functions (load model, predict, etc.)
└── README.md                   # Project documentation
```

### Folder Descriptions:
- **models/**: Stores serialized models (.pkl files) and possibly the dataset if saved as a pickle.
- **static/**: Holds images, CSS, or JavaScript files for the web interface.
- **templates/**: Flask’s default folder for HTML templates.
- **app.py**: Entry point for the Flask app.
- **utils.py**: Helper module for loading the model, preprocessing text, or making predictions.
- **requirements.txt**: Lists Python dependencies.

## Usage Instructions

### 1. Clone or Download this repository.
```bash
git clone https://github.com/your_username/spam_email_detection.git
cd spam_email_detection
```

### 2. Create a Virtual Environment (recommended):
```bash
conda create -n spam_env python=3.9
conda activate spam_env
```

### 3. Install Dependencies:
```bash
pip install -r requirements.txt
```

### 4. Run the Flask App:
```bash
python app.py
```

**If port 5000 is blocked**, open `app.py` and change:
```python
app.run(host="0.0.0.0", port=5001, debug=True)
```

### 5. Open Your Browser:
```plaintext
http://127.0.0.1:5000
```

**(or whichever port you used)**

### 6. Usage:
- Enter Email Text and click **“Check for Spam.”**
- The model will classify it as **Spam** or **Legitimate Email**.

## Future Improvements
- Experiment with different vectorization techniques like TF-IDF.
- Use more sophisticated ensemble models.
- Enhance the web interface with better UI/UX.
- Deploy the model to a cloud platform.

## License
This project is licensed under the MIT License - see the LICENSE file for details.

