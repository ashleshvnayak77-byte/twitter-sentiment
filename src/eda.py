# ─────────────────────────────────────────────────────────
# Phase 3 — Exploratory Data Analysis (EDA)
# Run in VS Code as a .py file OR copy cells into Jupyter
# ─────────────────────────────────────────────────────────

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from wordcloud import WordCloud
from collections import Counter
import re
import warnings
warnings.filterwarnings("ignore")

# ── Style config ─────────────────────────────────────────
plt.rcParams.update({
    "figure.facecolor": "#FAFAFA",
    "axes.facecolor":   "#FAFAFA",
    "font.family":      "DejaVu Sans",
    "axes.spines.top":  False,
    "axes.spines.right":False,
})
COLORS = {"positive": "#2A7A4F", "negative": "#B5441E", "neutral": "#9A7B2E"}

# ── Load data ─────────────────────────────────────────────
print("Loading data...")
df = pd.read_csv("data/sentiment140_raw.csv")
# Use a sample for faster EDA (remove .sample() for full run)
df_sample = df.sample(n=50000, random_state=42).reset_index(drop=True)
print(f"Working with {len(df_sample):,} tweets")

# ─────────────────────────────────────────────────────────
# 1. CLASS DISTRIBUTION
# ─────────────────────────────────────────────────────────
fig, axes = plt.subplots(1, 2, figsize=(12, 4))
fig.suptitle("1. Sentiment Class Distribution", fontsize=14, fontweight="bold")

# Bar chart
counts = df_sample["target"].value_counts().sort_index()
bars = axes[0].bar(["Negative (0)", "Positive (1)"],
                   counts.values,
                   color=[COLORS["negative"], COLORS["positive"]],
                   width=0.5, edgecolor="white")
for bar in bars:
    axes[0].text(bar.get_x() + bar.get_width()/2,
                 bar.get_height() + 300,
                 f"{bar.get_height():,}", ha="center", fontsize=11, fontweight="bold")
axes[0].set_ylabel("Number of tweets")
axes[0].set_title("Count per class")

# Pie chart
axes[1].pie(counts.values,
            labels=["Negative", "Positive"],
            colors=[COLORS["negative"], COLORS["positive"]],
            autopct="%1.1f%%", startangle=90,
            wedgeprops={"edgecolor": "white", "linewidth": 2})
axes[1].set_title("Class proportion")

plt.tight_layout()
plt.savefig("outputs/01_class_distribution.png", dpi=150, bbox_inches="tight")
plt.show()
print("Saved: outputs/01_class_distribution.png")

# ─────────────────────────────────────────────────────────
# 2. TWEET LENGTH ANALYSIS
# ─────────────────────────────────────────────────────────
df_sample["tweet_length"]   = df_sample["text"].str.len()
df_sample["word_count"]     = df_sample["text"].str.split().str.len()

fig, axes = plt.subplots(1, 2, figsize=(12, 4))
fig.suptitle("2. Tweet Length Analysis", fontsize=14, fontweight="bold")

for label, color in [(0, COLORS["negative"]), (1, COLORS["positive"])]:
    subset = df_sample[df_sample["target"] == label]
    name   = "Negative" if label == 0 else "Positive"
    axes[0].hist(subset["tweet_length"], bins=40, alpha=0.6,
                 color=color, label=name, edgecolor="white")
    axes[1].hist(subset["word_count"],   bins=30, alpha=0.6,
                 color=color, label=name, edgecolor="white")

axes[0].set_xlabel("Character count"); axes[0].set_ylabel("Frequency")
axes[0].set_title("Character length distribution"); axes[0].legend()
axes[1].set_xlabel("Word count");      axes[1].set_ylabel("Frequency")
axes[1].set_title("Word count distribution"); axes[1].legend()

plt.tight_layout()
plt.savefig("outputs/02_tweet_length.png", dpi=150, bbox_inches="tight")
plt.show()

# Print stats
print("\nTweet Length Stats:")
print(df_sample.groupby("target")[["tweet_length", "word_count"]].describe().round(1))

# ─────────────────────────────────────────────────────────
# 3. TOP WORDS PER CLASS (before preprocessing)
# ─────────────────────────────────────────────────────────
STOPWORDS = {"the","a","an","is","it","in","on","at","to","and",
             "of","for","are","was","be","this","that","with","i",
             "my","you","me","we","he","she","they","do","not",
             "but","or","so","if","as","by","from","have","had",
             "has","rt","via","http","https","amp","just","im","ur"}

def get_top_words(texts, n=20):
    words = []
    for t in texts:
        clean = re.sub(r"http\S+|@\S+|#\S+|[^a-zA-Z\s]", "", str(t).lower())
        words.extend([w for w in clean.split() if w not in STOPWORDS and len(w) > 2])
    return Counter(words).most_common(n)

pos_words = get_top_words(df_sample[df_sample["target"]==1]["text"])
neg_words = get_top_words(df_sample[df_sample["target"]==0]["text"])

fig, axes = plt.subplots(1, 2, figsize=(14, 5))
fig.suptitle("3. Top Words by Sentiment Class (raw text)", fontsize=14, fontweight="bold")

for ax, words, color, title in [
    (axes[0], pos_words, COLORS["positive"], "Positive tweets"),
    (axes[1], neg_words, COLORS["negative"], "Negative tweets")
]:
    words_list, counts = zip(*words)
    y_pos = range(len(words_list))
    ax.barh(y_pos, counts, color=color, alpha=0.8)
    ax.set_yticks(y_pos)
    ax.set_yticklabels(words_list, fontsize=10)
    ax.invert_yaxis()
    ax.set_xlabel("Frequency")
    ax.set_title(title)

plt.tight_layout()
plt.savefig("outputs/03_top_words.png", dpi=150, bbox_inches="tight")
plt.show()

# ─────────────────────────────────────────────────────────
# 4. WORD CLOUDS
# ─────────────────────────────────────────────────────────
def make_cloud(texts, color, title, filename):
    all_text = " ".join(texts)
    all_text = re.sub(r"http\S+|@\S+|#\S+", "", all_text.lower())
    wc = WordCloud(
        width=800, height=400,
        background_color="white",
        colormap="RdYlGn" if color == "green" else "OrRd",
        max_words=150,
        stopwords=STOPWORDS,
        collocations=False
    ).generate(all_text)
    plt.figure(figsize=(10, 5))
    plt.imshow(wc, interpolation="bilinear")
    plt.axis("off")
    plt.title(title, fontsize=14, fontweight="bold", pad=12)
    plt.tight_layout()
    plt.savefig(filename, dpi=150, bbox_inches="tight")
    plt.show()

make_cloud(df_sample[df_sample["target"]==1]["text"],
           "green", "Word Cloud — Positive Tweets",
           "outputs/04a_wordcloud_positive.png")

make_cloud(df_sample[df_sample["target"]==0]["text"],
           "red",   "Word Cloud — Negative Tweets",
           "outputs/04b_wordcloud_negative.png")

# ─────────────────────────────────────────────────────────
# 5. HASHTAG & MENTION ANALYSIS
# ─────────────────────────────────────────────────────────
def extract_pattern(texts, pattern):
    items = []
    for t in texts:
        items.extend(re.findall(pattern, str(t).lower()))
    return Counter(items).most_common(15)

pos_texts = df_sample[df_sample["target"]==1]["text"]
neg_texts = df_sample[df_sample["target"]==0]["text"]

pos_hashtags = extract_pattern(pos_texts, r"#(\w+)")
neg_hashtags = extract_pattern(neg_texts, r"#(\w+)")

if pos_hashtags and neg_hashtags:
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    fig.suptitle("5. Top Hashtags by Sentiment", fontsize=14, fontweight="bold")
    for ax, ht, color, title in [
        (axes[0], pos_hashtags, COLORS["positive"], "Positive hashtags"),
        (axes[1], neg_hashtags, COLORS["negative"], "Negative hashtags")
    ]:
        if ht:
            tags, counts = zip(*ht)
            ax.barh(range(len(tags)), counts, color=color, alpha=0.8)
            ax.set_yticks(range(len(tags)))
            ax.set_yticklabels(["#"+t for t in tags], fontsize=9)
            ax.invert_yaxis()
            ax.set_title(title)
    plt.tight_layout()
    plt.savefig("outputs/05_hashtags.png", dpi=150, bbox_inches="tight")
    plt.show()

# ─────────────────────────────────────────────────────────
# 6. SUMMARY STATS TABLE
# ─────────────────────────────────────────────────────────
print("\n" + "="*50)
print("EDA SUMMARY")
print("="*50)
print(f"Total tweets     : {len(df_sample):,}")
print(f"Positive tweets  : {(df_sample['target']==1).sum():,} ({(df_sample['target']==1).mean()*100:.1f}%)")
print(f"Negative tweets  : {(df_sample['target']==0).sum():,} ({(df_sample['target']==0).mean()*100:.1f}%)")
print(f"Avg tweet length : {df_sample['tweet_length'].mean():.1f} chars")
print(f"Avg word count   : {df_sample['word_count'].mean():.1f} words")
print(f"Unique users     : {df_sample['user'].nunique():,}")
print("="*50)
print("\nEDA complete! Check the outputs/ folder for all charts.")
