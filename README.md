# Sentiment Analysis of Social Media Reviews

## Project Description
This project is a machine learning-based web application that performs sentiment analysis on social media text reviews. It classifies user-generated text into three sentiment categories: **Positive**, **Negative**, or **Neutral**. The system is designed to help businesses, researchers, and developers automatically gauge public sentiment from social media platforms, enabling data-driven decision-making and trend analysis.

The project consists of two main components:
1. A **machine learning pipeline** for training and evaluating sentiment classification models.
2. A **Flask web application** that provides a user-friendly interface for real-time sentiment prediction.

## Features
- **Text Preprocessing Pipeline**: Cleans and prepares raw social media text by removing noise (URLs, mentions, special characters), handling contractions, and performing tokenization.
- **Multiple ML Model Support**: Implements and compares different classification algorithms (Logistic Regression, Naive Bayes, etc.).
- **Real-Time Prediction**: Web interface for instant sentiment analysis of user-input text.
- **Model Persistence**: Trained models are saved and loaded for repeated use without retraining.
- **Evaluation Metrics**: Provides detailed performance reports (accuracy, precision, recall, F1-score, confusion matrix).
- **Simple & Intuitive UI**: Clean web form for easy interaction with the model.

## Technologies Used
- **Programming Language**: Python 3.8+
- **Machine Learning Libraries**: Scikit-learn, NLTK, Pandas, NumPy
- **Web Framework**: Flask
- **Text Processing**: Regular Expressions (re), NLTK (stopwords, tokenization)
- **Data Handling**: Pandas, NumPy, Pickle (for model serialization)
- **Version Control**: Git, GitHub

## Project Structure
Sentiment-analysis-of-social-media-reviews/
- ├── data/ # Dataset directory
- │ ├── raw/ # Original dataset
- │ └── processed/ # Cleaned and preprocessed data
- ├── models/ # Saved trained models and vectorizers
- │ ├── classifier.pkl
- │ └── tfidf_vectorizer.pkl
- ├── src/ # Source code for ML pipeline
- │ ├── preprocess.py # Text cleaning and preprocessing functions
- │ ├── train_model.py # Model training and evaluation script
- │ └── predict.py # Script for making predictions
- ├── flask_app/ # Flask web application
- │ ├── static/ # CSS, JS, images
- │ ├── templates/ # HTML templates
- │ │ └── index.html
- │ └── app.py # Main Flask application file
- ├── notebooks/ # Jupyter notebooks for EDA and experimentation
- ├── requirements.txt # Python dependencies
- ├── main.py # Optional unified entry point
- └── README.md # Project documentation


## Dataset Information
The model is trained on a curated dataset of **social media reviews** collected from platforms like Twitter, Facebook, and Reddit. The dataset includes:

- **Text**: Raw user reviews/comments from social media.
- **Sentiment Labels**: Manually annotated or crowd-sourced sentiment labels (Positive, Negative, Neutral).
- **Size**: Typically contains thousands to tens of thousands of samples to ensure robust model training.
- **Characteristics**: Contains informal language, slang, emojis, hashtags, and misspellings common in social media text.

*Note: The specific dataset used in this project is included in the `data/` directory. Please ensure proper attribution if the dataset is from a public source.*

## Machine Learning Approach

### 1. Text Preprocessing
- Lowercasing all text
- Removing URLs, user mentions, and special characters
- Expanding contractions (e.g., "don't" → "do not")
- Tokenization and removal of stopwords
- Optional: Lemmatization/Stemming

### 2. Feature Extraction
- **TF-IDF Vectorization**: Converts processed text into numerical features by weighing term frequency and inverse document frequency.
- **n-gram Range**: Utilizes both unigrams and bigrams to capture phrase-level context.

### 3. Model Selection
- **Algorithm**: Logistic Regression (primary model due to good performance on text classification and interpretability).
- **Alternatives**: Multinomial Naive Bayes, SVM, and Random Forest are also explored.
- **Validation**: Stratified K-Fold Cross-Validation to ensure generalization.

### 4. Model Evaluation
Performance is measured using:
- Accuracy Score
- Precision, Recall, and F1-Score (per class)
- Confusion Matrix
- Classification Report

## Installation Instructions

### Prerequisites
- Python 3.8 or higher
- pip (Python package manager)
- Git

### Steps
1. **Clone the Repository**
   ```bash
   git clone https://github.com/Er-Sourabh-Patil/Sentiment-analysis-of-social-media-reviews.git
   cd Sentiment-analysis-of-social-media-reviews
2. **Create virtual environment**
   ```bash
   python -m venv venv
   venv\Scripts\activate.ps1
3. **Install dependencies**
   ```bash
   pip install -r requirements.txt

4. **Run the Flask Web Application**
   ```bash
   cd flask_app
   python app.py
