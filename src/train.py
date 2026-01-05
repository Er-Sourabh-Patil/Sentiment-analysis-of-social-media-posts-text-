import pandas as pd
import pickle
from sklearn.model_selection import train_test_split
from src.data_loader import combine_datasets
from src.preprocessing import clean_text, preprocess_and_tokenize
from src.model import create_lstm_model

def main():
    df = combine_datasets()
    print("Combined dataset shape BEFORE preprocessing:", df.shape)
    print("Label distribution before cleaning:\n", df['label'].value_counts())
    print(df.head())

    df.dropna(inplace=True)
    df['text'] = df['text'].apply(clean_text)
    print("Empty rows after cleaning:", (df['text'].str.strip() == "").sum())

    df = df[df['text'].str.strip() != ""]
    print("Dataset shape AFTER cleaning:", df.shape)
    print("Label distribution AFTER cleaning:\n", df['label'].value_counts())

    if df.empty:
        raise ValueError("Combined dataset is empty after preprocessing.")

    X, tokenizer = preprocess_and_tokenize(df['text'])
    y = df['label'].astype(int).values

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    model = create_lstm_model()
    model.fit(X_train, y_train, epochs=25, validation_data=(X_test, y_test))

    model.save("models/lstm_model.keras", save_format="keras")
    with open("models/tokenizer.pkl", "wb") as f:
        pickle.dump(tokenizer, f)

if __name__ == "__main__":
    main()

