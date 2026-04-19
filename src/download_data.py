# ─────────────────────────────────────────────────────────
# Phase 2 — Download & Load Dataset
# Run this file once: python src/download_data.py
# ─────────────────────────────────────────────────────────

import os
import zipfile
import pandas as pd

# ── OPTION A: Download via Kaggle API ────────────────────
# 1. Go to kaggle.com → Account → Create New API Token
# 2. Place kaggle.json in C:\Users\<you>\.kaggle\  (Windows)
#    or ~/.kaggle/ (Mac/Linux)
# 3. Then run the lines below

def download_via_kaggle():
    os.system("kaggle datasets download -d kazanova/sentiment140 -p data/")
    with zipfile.ZipFile("data/sentiment140.zip", "r") as z:
        z.extractall("data/")
    print("Downloaded and extracted via Kaggle API.")

# ── OPTION B: Manual Download ─────────────────────────────
# 1. Visit: https://www.kaggle.com/datasets/kazanova/sentiment140
# 2. Click Download
# 3. Unzip and place training.1600000.processed.noemoticon.csv
#    inside the  data/  folder of this project

# ── Load Dataset ─────────────────────────────────────────
def load_dataset(filepath="data/training.1600000.processed.noemoticon.csv"):
    """
    Sentiment140 columns:
    0 = target (0=negative, 4=positive)
    1 = tweet id
    2 = date
    3 = query
    4 = user
    5 = text
    """
    print(f"Loading dataset from {filepath} ...")
    df = pd.read_csv(
        filepath,
        encoding="latin-1",
        header=None,
        names=["target", "id", "date", "query", "user", "text"]
    )

    # Map labels: 4 → 1 (positive), 0 → 0 (negative)
    df["target"] = df["target"].map({4: 1, 0: 0})

    print(f"Total tweets loaded: {len(df):,}")
    print(f"Positive: {df['target'].sum():,} | Negative: {(df['target']==0).sum():,}")
    print(df.head(3))
    return df

if __name__ == "__main__":
    # Uncomment if using Kaggle API:
    # download_via_kaggle()

    df = load_dataset()
    df.to_csv("data/sentiment140_raw.csv", index=False)
    print("\nSaved to data/sentiment140_raw.csv")
