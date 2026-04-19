# Twitter Sentiment Analysis — Full Project
## VS Code · Python · BERT · Heat Map · From Scratch

---

## Project Structure

```
twitter_sentiment/
├── data/                        ← Dataset files go here
│   ├── training.1600000.processed.noemoticon.csv   ← raw download
│   ├── sentiment140_raw.csv     ← after download_data.py
│   ├── sentiment140_clean.csv   ← after preprocess.py
│   ├── split_train.csv          ← 80% training data
│   ├── split_val.csv            ← 10% validation
│   └── split_test.csv           ← 10% test
│
├── src/
│   ├── download_data.py         ← Phase 2: Load dataset
│   ├── eda.py                   ← Phase 3: Exploratory analysis
│   ├── preprocess.py            ← Phase 4: Clean + tokenize
│   ├── classical_models.py      ← Phase 5+6: TF-IDF + ML models
│   ├── bert_model.py            ← Phase 7: BERT fine-tuning
│   └── heatmap_report.py        ← Phase 8: Heat map + report
│
├── models/                      ← Saved model files
├── outputs/                     ← All charts and results
├── requirements.txt             ← All dependencies
└── README.md                    ← This file
```

---

## PHASE 1 — Environment Setup

### Step 1.1 — Install Python
Download Python 3.10+ from https://python.org
Make sure to check "Add to PATH" during install.

### Step 1.2 — Open VS Code
Install the Python extension from the Extensions panel (Ctrl+Shift+X)

### Step 1.3 — Open terminal in VS Code
Press Ctrl+` (backtick) to open the integrated terminal

### Step 1.4 — Create virtual environment
```bash
# Navigate to your project folder
cd path/to/twitter_sentiment

# Create venv
python -m venv venv

# Activate it
# Windows:
venv\Scripts\activate
# Mac/Linux:
source venv/bin/activate

# You should now see (venv) in your terminal
```

### Step 1.5 — Install all packages
```bash
pip install -r requirements.txt
```

This installs: pandas, numpy, sklearn, matplotlib, seaborn, 
nltk, transformers, torch, datasets, kaggle, wordcloud, etc.

Expect this to take 5–10 minutes.

### Step 1.6 — Test installation
```bash
python -c "import pandas, sklearn, transformers, torch; print('All good!')"
```

---

## PHASE 2 — Get the Dataset

### Option A — Kaggle API (recommended)
1. Go to https://www.kaggle.com → Account → Create New API Token
2. A file `kaggle.json` will download
3. Place it at:
   - Windows: `C:\Users\<YourName>\.kaggle\kaggle.json`
   - Mac/Linux: `~/.kaggle/kaggle.json`
4. Run:
```bash
python src/download_data.py
```

### Option B — Manual download
1. Go to: https://www.kaggle.com/datasets/kazanova/sentiment140
2. Click "Download"
3. Unzip the file
4. Place `training.1600000.processed.noemoticon.csv` inside the `data/` folder
5. Then run:
```bash
python src/download_data.py
```

Expected output:
```
Loading dataset from data/training.1600000.processed.noemoticon.csv ...
Total tweets loaded: 1,600,000
Positive: 800,000 | Negative: 800,000
```

---

## PHASE 3 — Exploratory Data Analysis

```bash
python src/eda.py
```

This generates 5 charts saved to `outputs/`:
- `01_class_distribution.png` — pie + bar chart of sentiment classes
- `02_tweet_length.png`       — character and word count distributions
- `03_top_words.png`          — top 20 words per sentiment class
- `04a_wordcloud_positive.png` — positive word cloud
- `04b_wordcloud_negative.png` — negative word cloud
- `05_hashtags.png`            — top hashtags per class

---

## PHASE 4 — Preprocessing

```bash
python src/preprocess.py
```

Steps performed automatically:
1. Lowercase all text
2. Remove URLs (http://...)
3. Remove @mentions
4. Remove # symbol (keep word)
5. Remove numbers and punctuation
6. Tokenize using NLTK TweetTokenizer
7. Remove stopwords (but keep sentiment negations: "not", "never" etc.)
8. Lemmatize each word

Outputs:
- `data/sentiment140_clean.csv`
- `data/split_train.csv` (80%)
- `data/split_val.csv`   (10%)
- `data/split_test.csv`  (10%)

---

## PHASE 5 + 6 — Feature Extraction + Classical ML

```bash
python src/classical_models.py
```

Phase 5 — Feature extraction:
- TF-IDF vectorizer (50K features, unigrams + bigrams)
- Bag of Words vectorizer

Phase 6 — Models trained:
1. Multinomial Naive Bayes (BoW)
2. Complement Naive Bayes (TF-IDF)
3. Logistic Regression (TF-IDF) ← usually best
4. Linear SVM (TF-IDF)

Outputs:
- `outputs/06_classical_models.png` — accuracy + F1 + confusion matrix
- `outputs/06b_predictive_words.png` — most predictive words
- `models/tfidf_vectorizer.pkl`
- `models/best_classical_*.pkl`

Expected results (approximate):
| Model | Accuracy | F1 |
|---|---|---|
| Naive Bayes | ~78% | ~0.79 |
| Logistic Reg | ~83% | ~0.83 |
| Linear SVM | ~84% | ~0.84 |

---

## PHASE 7 — BERT Fine-Tuning

### If you have a GPU (Nvidia):
```bash
python src/bert_model.py
```

### If you are on CPU only (slower):
Open `src/bert_model.py` and set:
```python
USE_TINY_BERT = True    # Uses DistilBERT (2x faster)
SAMPLE_SIZE   = 3000    # Use fewer tweets for speed
EPOCHS        = 2
```
Then run:
```bash
python src/bert_model.py
```

### OR use Google Colab (FREE GPU):
1. Go to https://colab.research.google.com
2. Upload your preprocessed CSV files
3. Copy-paste `src/bert_model.py` into a code cell
4. Runtime → Change runtime type → GPU
5. Run!

Expected training time:
- GPU (T4): ~15 minutes for 10K tweets, 3 epochs
- CPU only: ~2–3 hours (use smaller sample)

Expected results:
- BERT accuracy: ~92%
- BERT F1: ~0.91

Saved model: `models/bert_sentiment/`

---

## PHASE 8 — Heat Map + Weighted Index + Report

```bash
python src/heatmap_report.py
```

This implements the full semantic weighted index system:
- Intensity weight (1–5) based on keyword category
- Confidence weight (1–4) based on URL/hashtag/length
- Final score = Intensity × Confidence

Outputs:
- `outputs/08a_weighted_scores.png`   — score distribution by class
- `outputs/08b_heatmap_topic.png`     — MAIN heat map (topic × sentiment)
- `outputs/08c_density_grid.png`      — KDE equivalent density grid
- `outputs/08d_radar_chart.png`       — ETU overlay radar chart
- `outputs/weighted_index_summary.csv`— full summary table

---

## Run Order (quick reference)

```bash
# Activate env first
venv\Scripts\activate          # Windows
source venv/bin/activate       # Mac/Linux

# Run in this order:
python src/download_data.py    # Phase 2
python src/eda.py              # Phase 3
python src/preprocess.py       # Phase 4
python src/classical_models.py # Phase 5+6
python src/bert_model.py       # Phase 7
python src/heatmap_report.py   # Phase 8
```

---

## Troubleshooting

**"ModuleNotFoundError"**
→ Make sure your venv is activated and you ran `pip install -r requirements.txt`

**"FileNotFoundError: data/..."**
→ Make sure you placed the CSV in the `data/` folder and ran `download_data.py` first

**BERT training is too slow**
→ Set `USE_TINY_BERT = True` and `SAMPLE_SIZE = 2000` in `bert_model.py`
→ Or use Google Colab with free GPU

**Out of memory during BERT**
→ Reduce `BATCH_SIZE` from 32 to 16 or 8

**NLTK errors**
→ Run this once: `python -c "import nltk; nltk.download('all')"`

---

## Stack Used

| Tool | Purpose |
|---|---|
| Python 3.10+ | Language |
| pandas / numpy | Data handling |
| NLTK | Tokenization, stopwords, lemmatization |
| scikit-learn | TF-IDF, ML models, metrics |
| HuggingFace Transformers | BERT fine-tuning |
| PyTorch | Deep learning backend |
| matplotlib / seaborn | All visualizations |
| WordCloud | Word cloud generation |

---

Dataset: Sentiment140 (Go, Bhayani & Huang, 2009) — 1.6M tweets
Methodology adapted from: Jamaludin et al. (2024) J. Hydrology 628: 130519
