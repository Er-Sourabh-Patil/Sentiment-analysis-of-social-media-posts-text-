import pandas as pd
import os

def load_twitter_data():
    path = "data/Twitter_Data.csv"
    if not os.path.exists(path):
        raise FileNotFoundError(f"Twitter dataset not found at {path}")
    df = pd.read_csv(path, encoding='latin1')
    df.columns = df.columns.str.lower().str.strip()
    df = df[['text', 'sentiment']].rename(columns={'text': 'text', 'sentiment': 'label'})
    df['label'] = df['label'].astype(str).str.lower().str.strip()
    df = df[df['label'].isin(['positive', 'negative'])]
    df['label'] = df['label'].map({'positive': 1, 'negative': 0})
    return df

def load_google_data():
    path = "data/reviews.csv"
    if not os.path.exists(path):
        raise FileNotFoundError(f"Google Reviews dataset not found at {path}")
    df = pd.read_csv(path, encoding='latin1')
    df.columns = df.columns.str.lower().str.strip()
    df = df[['translated_review', 'sentiment']].rename(columns={'translated_review': 'text', 'sentiment': 'label'})
    df['label'] = df['label'].astype(str).str.lower().str.strip()
    df = df[df['label'].isin(['positive', 'negative'])]
    df['label'] = df['label'].map({'positive': 1, 'negative': 0})
    return df

def combine_datasets():
    print("\n📥 Loading Twitter data...")
    twitter_df = load_twitter_data()
    print("Twitter shape:", twitter_df.shape)

    print("\n📥 Loading Google reviews data...")
    google_df = load_google_data()
    print("Google shape:", google_df.shape)

    combined_df = pd.concat([twitter_df, google_df], ignore_index=True)
    print("\n📊 Combined dataset shape:", combined_df.shape)
    return combined_df
