{
  "title": "Spam Email Detection with SVM",
  "description": "A comprehensive end-to-end project demonstrating how to build, evaluate, and deploy a Machine Learning model for spam detection using Flask. This repository includes data preprocessing, multiple model evaluations, and deployment.",
  "table_of_contents": [
    "Screenshots",
    "Project Overview",
    "Dataset",
    "Data Preprocessing & Exploration",
    "Model Building & Comparison",
    "Chosen Model: SVM",
    "Project Structure",
    "Usage Instructions",
    "Detailed Steps",
    "Future Improvements",
    "License"
  ],
  "screenshots": [
    "https://github.com/user-attachments/assets/fa5eddb4-727c-461f-b6fb-7882376a515a",
    "https://github.com/user-attachments/assets/60d0e8a1-e281-40b7-a310-30570cec835b",
    "https://github.com/user-attachments/assets/ed493f66-b159-463a-9932-19da8422545b"
  ],
  "project_overview": {
    "description": "This project classifies emails as spam or non-spam (ham). It begins with loading and exploring the dataset, followed by feature engineering, training multiple machine learning models, and finally deploying the chosen model.",
    "highlights": [
      "Multiple models tested: Naive Bayes, SVM, K-Nearest Neighbors (KNN), Decision Tree (DT), and Random Forest.",
      "SVM achieved the highest accuracy (~97.61%) and was chosen for deployment.",
      "A Flask application serves predictions via a simple web interface (index.html)."
    ]
  },
  "dataset": {
    "description": "The dataset contains a collection of emails labeled as spam (1) or ham (0). It’s typically imbalanced, with more non-spam emails than spam.",
    "details": {
      "spam_emails": 2115,
      "ham_emails": 9185,
      "imbalance_ratio": "4:1 (ham:spam)"
    }
  },
  "data_preprocessing_exploration": {
    "steps": [
      "Loading Data: The raw data is read from a CSV file or from spam_data.pkl.",
      "Cleaning & Tokenization: Include lowercasing text, removing punctuation, and optionally removing stopwords.",
      "EDA: Check class distribution and visualize common words.",
      "Feature Extraction: A CountVectorizer (saved as countvectorizer.pkl) transforms email text into numeric vectors."
    ]
  },
  "model_building_comparison": {
    "models": ["Naive Bayes", "SVM", "KNN", "Decision Tree", "Random Forest"],
    "metrics": "Accuracy, precision, recall, and F1-score",
    "results": {
      "Naive Bayes": "94.66%",
      "SVM": "97.61%",
      "KNN": "90.97%",
      "Decision Tree": "96.02%",
      "Random Forest": "96.17%"
    }
  },
  "chosen_model": {
    "model": "SVM",
    "reason": "Highest accuracy of ~97.61%",
    "details": {
      "vectorization": "Using CountVectorizer from training",
      "classification": "RBF kernel used for prediction"
    }
  },
  "project_structure": {
    "folder_structure": {
      "ML_DEPLOYMENT": {
        "models": ["clf_NaiveBaised.pkl", "countvectorizer.pkl", "spam_data.pkl"],
        "static": ["shield-logo.png"],
        "templates": ["index.html"],
        "root_files": ["app.py", "requirements.txt", "utils.py", "README.md"]
      }
    },
    "descriptions": {
      "models": "Stores serialized models (.pkl files)",
      "static": "Holds images, CSS, etc.",
      "templates": "Flask’s default folder for HTML templates",
      "app.py": "Entry point for the Flask app",
      "utils.py": "Helper module for loading the model, preprocessing text, or making predictions",
      "requirements.txt": "Lists Python dependencies"
    }
  },
  "usage_instructions": {
    "steps": [
      "Clone or Download the repository",
      "Create a Virtual Environment",
      "Install Dependencies",
      "Run the Flask App",
      "Open Browser to access app",
      "Use the app to classify emails"
    ],
    "commands": {
      "clone": "git clone https://github.com/ZyadFri/spam_email_detection.git",
      "navigate": "cd spam_email_detection",
      "create_env_conda": "conda create -n spam_env python=3.9",
      "activate_env_conda": "conda activate spam_env",
      "install_dependencies_pip": "pip install -r requirements.txt",
      "install_dependencies_conda": "conda env create -f environment.yml",
      "run_app": "python app.py",
      "browser_url": "http://127.0.0.1:5000"
    },
    "port_change": {
      "file": "app.py",
      "code_snippet": "app.run(host=\"0.0.0.0\", port=5001, debug=True)"
    }
  },
  "api_usage": {
    "tool": "Postman",
    "method": "POST",
    "url": "http://127.0.0.1:5001/api/predict",
    "headers": {
      "Content-Type": "application/json"
    },
    "body": {
      "email": "Congratulations!!! You've won a FREE iPhone. Click here to claim your prize!"
    },
    "response": {
      "email": "Congratulations!!! You've won a FREE iPhone...",
      "prediction": "Spam"
    }
  },
  "future_improvements": [
    "Experiment with different vectorization techniques like TF-IDF",
    "Use more sophisticated ensemble models",
    "Enhance the web interface with better UI/UX",
    "Deploy the model to a cloud platform"
  ]
}

