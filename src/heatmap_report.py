# ─────────────────────────────────────────────────────────
# Phase 8 — Semantic Weighted Index + Heat Map + Report
# Run: python src/heatmap_report.py
# ─────────────────────────────────────────────────────────

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import seaborn as sns
from scipy.stats import gaussian_kde
import re, warnings
warnings.filterwarnings("ignore")

plt.rcParams.update({
    "figure.facecolor": "#FAFAFA",
    "axes.facecolor":   "#FAFAFA",
    "font.family":      "DejaVu Sans",
    "axes.spines.top":  False,
    "axes.spines.right":False,
})

# ─────────────────────────────────────────────────────────
# SEMANTIC WEIGHTED INDEX
# Adapted from Jamaludin et al. (2024) flood methodology
# ─────────────────────────────────────────────────────────

# Table 3 — Semantic Intensity Weights
INTENSITY_WORDS = {
    5: ["hate", "devastated", "obsessed", "addicted", "crying",
        "blessed", "ruined", "heartbroken", "amazing", "disgusting",
        "love", "perfect", "horrible", "incredible"],
    4: ["broken", "scam", "refund", "fraud", "flawless",
        "lifechanger", "must-have", "pathetic", "excellent",
        "terrible", "awful", "brilliant", "worst", "best"],
    3: ["apple", "amazon", "google", "twitter", "netflix",
        "starbucks", "uber", "airline", "iphone", "samsung"],
    2: ["good", "bad", "great", "sad", "happy",
        "boring", "exciting", "nice", "ugly", "beautiful"],
    1: []  # default: everything else = 1
}

# Table 4 — Confidence Depth Weights
def get_confidence_weight(text):
    """Assign confidence weight based on tweet features."""
    text = str(text)
    score = 1   # default: uncertain
    if re.search(r"http\S+", text):
        score = 2   # has URL (possible)
    if re.search(r"#\w+", text):
        score = max(score, 2)   # has hashtag
    # Simulating media attachment heuristic:
    # tweets with "pic", "photo", "image", "video" → higher confidence
    if re.search(r"\b(pic|photo|image|video|screenshot)\b", text.lower()):
        score = max(score, 3)   # likely
    if len(text) > 120:
        score = max(score, 3)   # long tweet = more context
    return score

def get_intensity_weight(text):
    """Assign semantic intensity weight based on word match."""
    words = str(text).lower().split()
    max_wt = 1
    for wt, word_list in INTENSITY_WORDS.items():
        if any(w in words for w in word_list):
            max_wt = max(max_wt, wt)
    return max_wt

def compute_weighted_score(text):
    """Final weighted score = intensity × confidence."""
    return get_intensity_weight(text) * get_confidence_weight(text)

# ─────────────────────────────────────────────────────────
# Load data and compute scores
# ─────────────────────────────────────────────────────────
print("Loading data...")
df = pd.read_csv("data/sentiment140_clean.csv").dropna()
df = df.sample(n=min(20000, len(df)), random_state=42)

print("Computing semantic weighted scores...")
df["intensity_weight"]  = df["text"].apply(get_intensity_weight)
df["confidence_weight"] = df["text"].apply(get_confidence_weight)
df["weighted_score"]    = df["intensity_weight"] * df["confidence_weight"]

print(f"\nWeighted Score Distribution:")
print(df["weighted_score"].describe().round(2))
print(f"\nScore breakdown:")
print(df["weighted_score"].value_counts().sort_index())

# ─────────────────────────────────────────────────────────
# TOPIC CLASSIFICATION (simple keyword-based)
# ─────────────────────────────────────────────────────────
TOPIC_KEYWORDS = {
    "Product/Brand":   ["product","buy","purchase","brand","quality","price","amazon","apple"],
    "Entertainment":   ["movie","music","song","concert","show","tv","series","film","game"],
    "Food & Life":     ["food","eat","coffee","restaurant","lunch","dinner","cook","taste"],
    "Politics":        ["government","president","election","vote","political","congress","policy"],
    "Tech":            ["phone","app","software","tech","update","bug","feature","website"],
    "Health":          ["health","doctor","hospital","sick","medicine","covid","vaccine"],
    "Sports":          ["game","team","match","player","win","lose","score","football","cricket"],
}

def classify_topic(text):
    text = str(text).lower()
    for topic, keywords in TOPIC_KEYWORDS.items():
        if any(k in text for k in keywords):
            return topic
    return "General"

df["topic"] = df["text"].apply(classify_topic)
print(f"\nTopic distribution:")
print(df["topic"].value_counts())

# ─────────────────────────────────────────────────────────
# VISUALIZATION 1 — Weighted Score Distribution
# ─────────────────────────────────────────────────────────
fig, axes = plt.subplots(1, 2, figsize=(13, 5))
fig.suptitle("8.1 Semantic Weighted Score Distribution", fontsize=13, fontweight="bold")

# By sentiment class
for label, color, name in [(1,"#2A7A4F","Positive"), (0,"#B5441E","Negative")]:
    subset = df[df["target"]==label]["weighted_score"]
    axes[0].hist(subset, bins=20, alpha=0.65, color=color, label=name, edgecolor="white")
axes[0].set_xlabel("Weighted Score (Intensity × Confidence)")
axes[0].set_ylabel("Count")
axes[0].set_title("Score distribution by sentiment class")
axes[0].legend()

# Score vs accuracy (boxplot)
df.boxplot(column="weighted_score", by="target", ax=axes[1],
           boxprops=dict(color="#1A5C8A"),
           medianprops=dict(color="#B5441E", linewidth=2))
axes[1].set_title("Score by class")
axes[1].set_xlabel("Sentiment class (0=Neg, 1=Pos)")
axes[1].set_ylabel("Weighted Score")
plt.suptitle("")

plt.tight_layout()
plt.savefig("outputs/08a_weighted_scores.png", dpi=150, bbox_inches="tight")
plt.show()
print("Saved: outputs/08a_weighted_scores.png")

# ─────────────────────────────────────────────────────────
# VISUALIZATION 2 — Topic × Sentiment Heat Map
# (Analogous to the flood zone heat map)
# ─────────────────────────────────────────────────────────
pivot = df.groupby(["topic", "target"])["weighted_score"].mean().unstack(fill_value=0)
pivot.columns = ["Negative", "Positive"]
pivot = pivot.loc[pivot.sum(axis=1).sort_values(ascending=False).index]

fig, axes = plt.subplots(1, 2, figsize=(14, 6))
fig.suptitle("8.2 Sentiment Intensity Heat Map — Topic × Class\n(analogous to flood zone heat map)",
             fontsize=13, fontweight="bold")

# Positive heat map
sns.heatmap(pivot[["Positive"]], ax=axes[0],
            cmap="YlOrRd", annot=True, fmt=".2f",
            linewidths=0.5, linecolor="white",
            cbar_kws={"label": "Avg Weighted Score"},
            vmin=0, vmax=pivot.max().max())
axes[0].set_title("Positive Sentiment Intensity", color="#2A7A4F", fontweight="bold")
axes[0].set_ylabel("Topic Domain")
axes[0].set_xticklabels(["Positive"], rotation=0)

# Negative heat map
sns.heatmap(pivot[["Negative"]], ax=axes[1],
            cmap="Blues", annot=True, fmt=".2f",
            linewidths=0.5, linecolor="white",
            cbar_kws={"label": "Avg Weighted Score"},
            vmin=0, vmax=pivot.max().max())
axes[1].set_title("Negative Sentiment Intensity", color="#1A5C8A", fontweight="bold")
axes[1].set_ylabel("")
axes[1].set_xticklabels(["Negative"], rotation=0)

plt.tight_layout()
plt.savefig("outputs/08b_heatmap_topic.png", dpi=150, bbox_inches="tight")
plt.show()
print("Saved: outputs/08b_heatmap_topic.png")

# ─────────────────────────────────────────────────────────
# VISUALIZATION 3 — Full Topic × Score Grid Heat Map
# ─────────────────────────────────────────────────────────
full_pivot = df.groupby(["topic", "weighted_score"])["target"].count().unstack(fill_value=0)

plt.figure(figsize=(12, 6))
sns.heatmap(full_pivot,
            cmap="RdYlGn",
            annot=True, fmt="d",
            linewidths=0.5, linecolor="#eee",
            cbar_kws={"label": "Tweet Count"})
plt.title("8.3 Tweet Count — Topic × Weighted Score\n(KDE-equivalent density grid)",
          fontsize=12, fontweight="bold")
plt.xlabel("Weighted Score (Intensity × Confidence)")
plt.ylabel("Topic Domain")
plt.tight_layout()
plt.savefig("outputs/08c_density_grid.png", dpi=150, bbox_inches="tight")
plt.show()
print("Saved: outputs/08c_density_grid.png")

# ─────────────────────────────────────────────────────────
# VISUALIZATION 4 — Radar chart (topic domain coverage)
# ─────────────────────────────────────────────────────────
topics_list = list(TOPIC_KEYWORDS.keys()) + ["General"]
pos_scores  = [df[(df["topic"]==t)&(df["target"]==1)]["weighted_score"].mean() for t in topics_list]
neg_scores  = [df[(df["topic"]==t)&(df["target"]==0)]["weighted_score"].mean() for t in topics_list]

# Handle NaN
pos_scores = [s if not np.isnan(s) else 0 for s in pos_scores]
neg_scores = [s if not np.isnan(s) else 0 for s in neg_scores]

angles = np.linspace(0, 2*np.pi, len(topics_list), endpoint=False).tolist()
angles += angles[:1]
pos_scores += pos_scores[:1]
neg_scores += neg_scores[:1]

fig, ax = plt.subplots(figsize=(8, 8), subplot_kw=dict(polar=True))
ax.plot(angles, pos_scores, "o-", color="#2A7A4F", linewidth=2, label="Positive")
ax.fill(angles, pos_scores, color="#2A7A4F", alpha=0.15)
ax.plot(angles, neg_scores, "o-", color="#B5441E", linewidth=2, label="Negative")
ax.fill(angles, neg_scores, color="#B5441E", alpha=0.1)
ax.set_xticks(angles[:-1])
ax.set_xticklabels(topics_list, fontsize=10)
ax.set_title("8.4 Sentiment Intensity Radar — Topic Domain Coverage\n(ETU overlay equivalent)",
             fontsize=12, fontweight="bold", pad=20)
ax.legend(loc="upper right", bbox_to_anchor=(1.3, 1.1))
plt.tight_layout()
plt.savefig("outputs/08d_radar_chart.png", dpi=150, bbox_inches="tight")
plt.show()
print("Saved: outputs/08d_radar_chart.png")

# ─────────────────────────────────────────────────────────
# FINAL SUMMARY TABLE
# ─────────────────────────────────────────────────────────
print("\n" + "="*60)
print("FINAL WEIGHTED INDEX SUMMARY")
print("="*60)
summary = df.groupby(["topic","target"]).agg(
    count=("weighted_score","count"),
    avg_score=("weighted_score","mean"),
    max_score=("weighted_score","max")
).round(2)
summary.index.names = ["Topic", "Sentiment (0=Neg, 1=Pos)"]
print(summary.to_string())
summary.to_csv("outputs/weighted_index_summary.csv")
print("\nSaved: outputs/weighted_index_summary.csv")

print("\nPhase 8 complete! All heat maps and charts saved to outputs/")
