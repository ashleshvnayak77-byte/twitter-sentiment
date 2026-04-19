# ─────────────────────────────────────────────────────────
# Phase 5 + 6 — Feature Extraction & Classical ML Models
# Run: python src/classical_models.py
# ─────────────────────────────────────────────────────────

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pickle, time, warnings
warnings.filterwarnings("ignore")

from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn.naive_bayes    import MultinomialNB, ComplementNB
from sklearn.linear_model   import LogisticRegression
from sklearn.svm            import LinearSVC
from sklearn.pipeline       import Pipeline
from sklearn.metrics        import (classification_report, confusion_matrix,
                                     accuracy_score, f1_score, roc_auc_score,
                                     ConfusionMatrixDisplay)
from sklearn.model_selection import cross_val_score

# ─────────────────────────────────────────────────────────
# Load splits
# ─────────────────────────────────────────────────────────
print("Loading data splits...")
train = pd.read_csv("data/split_train.csv").dropna()
val   = pd.read_csv("data/split_val.csv").dropna()
test  = pd.read_csv("data/split_test.csv").dropna()

X_train, y_train = train["clean_text"], train["target"]
X_val,   y_val   = val["clean_text"],   val["target"]
X_test,  y_test  = test["clean_text"],  test["target"]

print(f"Train: {len(X_train):,} | Val: {len(X_val):,} | Test: {len(X_test):,}")

# ─────────────────────────────────────────────────────────
# PHASE 5 — Feature Extraction
# ─────────────────────────────────────────────────────────

# ── A: TF-IDF Vectorizer ─────────────────────────────────
print("\n[Phase 5] Fitting TF-IDF vectorizer...")
tfidf = TfidfVectorizer(
    max_features=50000,      # top 50K terms
    ngram_range=(1, 2),      # unigrams + bigrams
    min_df=3,                # ignore very rare terms
    max_df=0.95,             # ignore very common terms
    sublinear_tf=True        # log normalization
)
X_train_tfidf = tfidf.fit_transform(X_train)
X_val_tfidf   = tfidf.transform(X_val)
X_test_tfidf  = tfidf.transform(X_test)

print(f"TF-IDF matrix shape: {X_train_tfidf.shape}")
print(f"Top 10 features: {tfidf.get_feature_names_out()[:10]}")

# Save vectorizer
with open("models/tfidf_vectorizer.pkl", "wb") as f:
    pickle.dump(tfidf, f)
print("Saved: models/tfidf_vectorizer.pkl")

# ── B: Bag of Words (for Naive Bayes) ────────────────────
bow = CountVectorizer(max_features=30000, ngram_range=(1, 2), min_df=3)
X_train_bow = bow.fit_transform(X_train)
X_val_bow   = bow.transform(X_val)
X_test_bow  = bow.transform(X_test)

# ─────────────────────────────────────────────────────────
# PHASE 6 — Classical ML Models
# ─────────────────────────────────────────────────────────

def evaluate_model(name, model, X_tr, y_tr, X_te, y_te, feature_name="TF-IDF"):
    """Train, predict, and return metrics dict."""
    print(f"\n{'─'*50}")
    print(f"Training: {name} [{feature_name}]")
    t0 = time.time()
    model.fit(X_tr, y_tr)
    train_time = time.time() - t0

    y_pred = model.predict(X_te)
    acc    = accuracy_score(y_te, y_pred)
    f1     = f1_score(y_te, y_pred)
    try:
        if hasattr(model, "predict_proba"):
            auc = roc_auc_score(y_te, model.predict_proba(X_te)[:,1])
        else:
            auc = roc_auc_score(y_te, model.decision_function(X_te))
    except:
        auc = 0.0

    print(f"  Accuracy : {acc:.4f}")
    print(f"  F1 Score : {f1:.4f}")
    print(f"  ROC-AUC  : {auc:.4f}")
    print(f"  Train time: {train_time:.1f}s")
    print(classification_report(y_te, y_pred, target_names=["Negative","Positive"]))

    return {"model": name, "features": feature_name,
            "accuracy": acc, "f1": f1, "auc": auc, "time": train_time,
            "trained_model": model}

results = []

# ── Model 1: Naive Bayes (BoW) ───────────────────────────
nb = MultinomialNB(alpha=0.1)
r = evaluate_model("Naive Bayes", nb, X_train_bow, y_train, X_val_bow, y_val, "BoW")
results.append(r)

# ── Model 2: Complement Naive Bayes (TF-IDF) ─────────────
cnb = ComplementNB(alpha=0.1)
r = evaluate_model("Complement NB", cnb, X_train_tfidf, y_train, X_val_tfidf, y_val, "TF-IDF")
results.append(r)

# ── Model 3: Logistic Regression (TF-IDF) ────────────────
lr = LogisticRegression(C=1.0, max_iter=1000, solver="saga", n_jobs=-1)
r = evaluate_model("Logistic Regression", lr, X_train_tfidf, y_train, X_val_tfidf, y_val, "TF-IDF")
results.append(r)

# ── Model 4: Linear SVM (TF-IDF) ─────────────────────────
svm = LinearSVC(C=0.5, max_iter=2000)
r = evaluate_model("Linear SVM", svm, X_train_tfidf, y_train, X_val_tfidf, y_val, "TF-IDF")
results.append(r)

# ─────────────────────────────────────────────────────────
# Test set evaluation — best model
# ─────────────────────────────────────────────────────────
best = max(results, key=lambda x: x["f1"])
print(f"\n{'='*50}")
print(f"BEST MODEL: {best['model']} [{best['features']}]")
print(f"Test set evaluation:")

model = best["trained_model"]
if best["features"] == "TF-IDF":
    X_te_final = X_test_tfidf
else:
    X_te_final = X_test_bow

y_pred_test = model.predict(X_te_final)
print(classification_report(y_test, y_pred_test, target_names=["Negative","Positive"]))

# Save best model
with open(f"models/best_classical_{best['model'].replace(' ','_').lower()}.pkl", "wb") as f:
    pickle.dump(model, f)
print(f"Saved best model to models/")

# ─────────────────────────────────────────────────────────
# VISUALIZATIONS
# ─────────────────────────────────────────────────────────
fig, axes = plt.subplots(1, 3, figsize=(16, 5))
fig.suptitle("Phase 6 — Classical Model Comparison", fontsize=13, fontweight="bold")

names   = [r["model"] for r in results]
accs    = [r["accuracy"] for r in results]
f1s     = [r["f1"] for r in results]
colors  = ["#B5441E","#CC7A2E","#2A7A4F","#1A5C8A"]

# Accuracy
axes[0].barh(names, accs, color=colors, alpha=0.85)
axes[0].set_xlim(0.5, 1.0)
for i, v in enumerate(accs):
    axes[0].text(v + 0.002, i, f"{v:.3f}", va="center", fontsize=9)
axes[0].set_title("Accuracy"); axes[0].set_xlabel("Score")

# F1
axes[1].barh(names, f1s, color=colors, alpha=0.85)
axes[1].set_xlim(0.5, 1.0)
for i, v in enumerate(f1s):
    axes[1].text(v + 0.002, i, f"{v:.3f}", va="center", fontsize=9)
axes[1].set_title("F1 Score"); axes[1].set_xlabel("Score")

# Confusion matrix of best model
cm = confusion_matrix(y_test, y_pred_test)
sns.heatmap(cm, annot=True, fmt="d", cmap="RdYlGn",
            xticklabels=["Neg","Pos"],
            yticklabels=["Neg","Pos"], ax=axes[2])
axes[2].set_title(f"Confusion Matrix\n({best['model']})")
axes[2].set_ylabel("True"); axes[2].set_xlabel("Predicted")

plt.tight_layout()
plt.savefig("outputs/06_classical_models.png", dpi=150, bbox_inches="tight")
plt.show()
print("Saved: outputs/06_classical_models.png")

# ─────────────────────────────────────────────────────────
# TOP PREDICTIVE WORDS (Logistic Regression)
# ─────────────────────────────────────────────────────────
if isinstance(model, LogisticRegression):
    feat_names = np.array(tfidf.get_feature_names_out())
    coefs      = model.coef_[0]
    top_pos    = feat_names[np.argsort(coefs)[-20:][::-1]]
    top_neg    = feat_names[np.argsort(coefs)[:20]]
    top_pos_c  = coefs[np.argsort(coefs)[-20:][::-1]]
    top_neg_c  = coefs[np.argsort(coefs)[:20]]

    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    fig.suptitle("Most Predictive Words (Logistic Regression coefficients)", fontsize=12, fontweight="bold")

    axes[0].barh(range(20), top_pos_c, color="#2A7A4F", alpha=0.85)
    axes[0].set_yticks(range(20)); axes[0].set_yticklabels(top_pos, fontsize=9)
    axes[0].invert_yaxis(); axes[0].set_title("Positive signal words")

    axes[1].barh(range(20), np.abs(top_neg_c), color="#B5441E", alpha=0.85)
    axes[1].set_yticks(range(20)); axes[1].set_yticklabels(top_neg, fontsize=9)
    axes[1].invert_yaxis(); axes[1].set_title("Negative signal words")

    plt.tight_layout()
    plt.savefig("outputs/06b_predictive_words.png", dpi=150, bbox_inches="tight")
    plt.show()

print("\nPhase 5 + 6 complete!")
