# ─────────────────────────────────────────────────────────
# Phase 4 — Data Preprocessing
# Run: python src/preprocess.py
# ─────────────────────────────────────────────────────────

import pandas as pd
import numpy as np
import re
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from nltk.tokenize import TweetTokenizer
import pickle
from tqdm import tqdm

# Download NLTK data (run once)
nltk.download("stopwords", quiet=True)
nltk.download("wordnet",   quiet=True)
nltk.download("omw-1.4",   quiet=True)

# ── Tools ─────────────────────────────────────────────────
tokenizer   = TweetTokenizer(preserve_case=False, strip_handles=True, reduce_len=True)
lemmatizer  = WordNetLemmatizer()
STOP_WORDS  = set(stopwords.words("english"))

# Keep some sentiment-bearing negations
KEEP_WORDS  = {"no", "not", "nor", "never", "nothing", "nobody",
               "nowhere", "neither", "cannot", "won't", "don't",
               "can't", "shouldn't", "wouldn't", "couldn't"}
STOP_WORDS  -= KEEP_WORDS

# ─────────────────────────────────────────────────────────
# Core cleaning function
# ─────────────────────────────────────────────────────────
def clean_tweet(text):
    """
    Full preprocessing pipeline for a single tweet.
    Steps:
    1. Lowercase
    2. Remove URLs
    3. Remove @mentions
    4. Remove hashtag symbol (keep word)
    5. Remove special characters / numbers
    6. Tokenize (Twitter-aware)
    7. Remove stopwords
    8. Lemmatize
    9. Rejoin tokens
    """
    text = str(text).lower()

    # 2. Remove URLs
    text = re.sub(r"http\S+|www\S+|https\S+", "", text, flags=re.MULTILINE)

    # 3. Remove @mentions
    text = re.sub(r"@\w+", "", text)

    # 4. Remove # symbol but keep hashtag word
    text = re.sub(r"#(\w+)", r"\1", text)

    # 5. Remove numbers, punctuation (keep letters + spaces)
    text = re.sub(r"[^a-zA-Z\s]", "", text)

    # 6. Tokenize
    tokens = tokenizer.tokenize(text)

    # 7. Remove stopwords + short tokens
    tokens = [t for t in tokens if t not in STOP_WORDS and len(t) > 1]

    # 8. Lemmatize
    tokens = [lemmatizer.lemmatize(t) for t in tokens]

    # 9. Rejoin
    return " ".join(tokens)

# ─────────────────────────────────────────────────────────
# Main pipeline
# ─────────────────────────────────────────────────────────
def preprocess_dataset(input_path="data/sentiment140_raw.csv",
                        output_path="data/sentiment140_clean.csv",
                        sample_size=200000):
    print("Loading raw data...")
    df = pd.read_csv(input_path)

    # Use a sample for faster development
    # Remove .sample() to process all 1.6M tweets (takes ~30 min)
    df = df.sample(n=min(sample_size, len(df)), random_state=42).reset_index(drop=True)
    print(f"Processing {len(df):,} tweets...")

    # Apply cleaning with progress bar
    tqdm.pandas(desc="Cleaning")
    df["clean_text"] = df["text"].progress_apply(clean_tweet)

    # Remove empty tweets after cleaning
    df = df[df["clean_text"].str.strip() != ""]
    df = df[df["clean_text"].str.split().str.len() >= 2]

    # Basic stats
    print(f"\nAfter cleaning: {len(df):,} tweets remain")
    print(f"Avg clean length: {df['clean_text'].str.len().mean():.1f} chars")

    # Save
    df[["target", "text", "clean_text"]].to_csv(output_path, index=False)
    print(f"Saved to {output_path}")

    # Show before/after examples
    print("\n--- Before / After Examples ---")
    for i in range(5):
        print(f"\nRaw  : {df['text'].iloc[i]}")
        print(f"Clean: {df['clean_text'].iloc[i]}")

    return df

# ─────────────────────────────────────────────────────────
# Train/val/test split
# ─────────────────────────────────────────────────────────
def split_dataset(df, test_size=0.2, val_size=0.1):
    from sklearn.model_selection import train_test_split

    X = df["clean_text"]
    y = df["target"]

    # First split: train+val vs test
    X_trainval, X_test, y_trainval, y_test = train_test_split(
        X, y, test_size=test_size, stratify=y, random_state=42
    )

    # Second split: train vs val
    val_relative = val_size / (1 - test_size)
    X_train, X_val, y_train, y_val = train_test_split(
        X_trainval, y_trainval, test_size=val_relative, stratify=y_trainval, random_state=42
    )

    print(f"\nSplit sizes:")
    print(f"  Train : {len(X_train):,}")
    print(f"  Val   : {len(X_val):,}")
    print(f"  Test  : {len(X_test):,}")

    # Save splits
    for name, X, y in [("train", X_train, y_train),
                        ("val",   X_val,   y_val),
                        ("test",  X_test,  y_test)]:
        split_df = pd.DataFrame({"clean_text": X, "target": y})
        split_df.to_csv(f"data/split_{name}.csv", index=False)
        print(f"Saved data/split_{name}.csv")

    return X_train, X_val, X_test, y_train, y_val, y_test

if __name__ == "__main__":
    df = preprocess_dataset()
    X_train, X_val, X_test, y_train, y_val, y_test = split_dataset(df)
    print("\nPreprocessing complete!")
