import os
import pickle
from src.data_loader import combine_datasets
from src.preprocessing import clean_text, preprocess_and_tokenize
from src.model import create_lstm_model
from sklearn.model_selection import train_test_split

import numpy as np
import pandas as pd

import tensorflow as tf

def main():
    print("\n✅ Loading and combining datasets...")
    df = combine_datasets()
    print("Initial dataset shape:", df.shape)

    df.dropna(inplace=True)
    df['text'] = df['text'].apply(clean_text)
    df = df[df['text'].str.strip() != ""]
    print("Cleaned dataset shape:", df.shape)
    print("Label distribution:\n", df['label'].value_counts())

    if df.empty:
        raise ValueError("Combined dataset is empty after preprocessing.")

    print("\n✅ Tokenizing text...")
    X, tokenizer = preprocess_and_tokenize(df['text'])
    y = df['label'].astype(int).values

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    print("\n✅ Building and training model...")
    model = create_lstm_model()
    model.fit(X_train, y_train, epochs=25, validation_data=(X_test, y_test))

    os.makedirs("models", exist_ok=True)
    model.save("models/lstm_model.keras", save_format="keras")
    with open("models/tokenizer.pkl", "wb") as f:
        pickle.dump(tokenizer, f)

    print("\n✅ Training complete. Model and tokenizer saved in /models")

if __name__ == "__main__":
    main()
