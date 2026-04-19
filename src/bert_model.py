# ─────────────────────────────────────────────────────────
# Phase 7 — BERT Fine-Tuning
# Run: python src/bert_model.py
#
# GPU recommended (Google Colab free tier works great)
# If no GPU: set USE_TINY_BERT = True below for CPU training
# ─────────────────────────────────────────────────────────

import pandas as pd
import numpy as np
import torch
import matplotlib.pyplot as plt
import warnings, os, time
warnings.filterwarnings("ignore")

from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    TrainingArguments,
    Trainer,
    EarlyStoppingCallback
)
from datasets import Dataset
from sklearn.metrics import accuracy_score, f1_score, classification_report

# ── Config ────────────────────────────────────────────────
USE_TINY_BERT = False   # Set True if no GPU → uses distilbert (much faster)
SAMPLE_SIZE   = 10000   # Tweets to train on (increase for better accuracy)
MAX_LEN       = 128     # Max token length
BATCH_SIZE    = 32
EPOCHS        = 3
LR            = 2e-5

MODEL_NAME = "distilbert-base-uncased" if USE_TINY_BERT else "bert-base-uncased"

device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Using device: {device}")
print(f"Model: {MODEL_NAME}")
print(f"Training on {SAMPLE_SIZE:,} tweets for {EPOCHS} epochs")

# ─────────────────────────────────────────────────────────
# Load data
# ─────────────────────────────────────────────────────────
print("\nLoading data splits...")
train = pd.read_csv("data/split_train.csv").dropna()
val   = pd.read_csv("data/split_val.csv").dropna()
test  = pd.read_csv("data/split_test.csv").dropna()

# Sample for faster training
train = train.sample(n=min(SAMPLE_SIZE, len(train)), random_state=42)
val   = val.sample(n=min(SAMPLE_SIZE//4, len(val)), random_state=42)
test  = test.sample(n=min(SAMPLE_SIZE//4, len(test)), random_state=42)

print(f"Train: {len(train):,} | Val: {len(val):,} | Test: {len(test):,}")

# ─────────────────────────────────────────────────────────
# Tokenizer
# ─────────────────────────────────────────────────────────
print(f"\nLoading tokenizer: {MODEL_NAME}")
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

def tokenize_function(examples):
    return tokenizer(
        examples["clean_text"],
        padding="max_length",
        truncation=True,
        max_length=MAX_LEN
    )

# Convert to HuggingFace Dataset format
def to_hf_dataset(df):
    ds = Dataset.from_pandas(df[["clean_text", "target"]].rename(columns={"target": "labels"}))
    return ds.map(tokenize_function, batched=True)

print("Tokenizing datasets...")
train_ds = to_hf_dataset(train)
val_ds   = to_hf_dataset(val)
test_ds  = to_hf_dataset(test)

# ─────────────────────────────────────────────────────────
# Model
# ─────────────────────────────────────────────────────────
print(f"\nLoading model: {MODEL_NAME}")
model = AutoModelForSequenceClassification.from_pretrained(
    MODEL_NAME,
    num_labels=2,
    id2label={0: "NEGATIVE", 1: "POSITIVE"},
    label2id={"NEGATIVE": 0, "POSITIVE": 1}
)

# ─────────────────────────────────────────────────────────
# Metrics function
# ─────────────────────────────────────────────────────────
def compute_metrics(eval_pred):
    logits, labels = eval_pred
    preds = np.argmax(logits, axis=-1)
    acc = accuracy_score(labels, preds)
    f1  = f1_score(labels, preds)
    return {"accuracy": acc, "f1": f1}

# ─────────────────────────────────────────────────────────
# Training arguments
# ─────────────────────────────────────────────────────────
training_args = TrainingArguments(
    output_dir              = "models/bert_checkpoints",
    num_train_epochs        = EPOCHS,
    per_device_train_batch_size = BATCH_SIZE,
    per_device_eval_batch_size  = BATCH_SIZE,
    learning_rate           = LR,
    weight_decay            = 0.01,
    warmup_ratio            = 0.1,
    evaluation_strategy     = "epoch",
    save_strategy           = "epoch",
    load_best_model_at_end  = True,
    metric_for_best_model   = "f1",
    logging_steps           = 100,
    logging_dir             = "outputs/bert_logs",
    fp16                    = (device == "cuda"),  # Mixed precision if GPU
    dataloader_num_workers  = 0,
    report_to               = "none"               # Disable wandb
)

# ─────────────────────────────────────────────────────────
# Trainer
# ─────────────────────────────────────────────────────────
trainer = Trainer(
    model           = model,
    args            = training_args,
    train_dataset   = train_ds,
    eval_dataset    = val_ds,
    compute_metrics = compute_metrics,
    callbacks       = [EarlyStoppingCallback(early_stopping_patience=2)]
)

# ─────────────────────────────────────────────────────────
# Fine-tune
# ─────────────────────────────────────────────────────────
print("\n" + "="*50)
print("Starting BERT fine-tuning...")
print("="*50)
t0 = time.time()
trainer.train()
print(f"\nTraining complete in {(time.time()-t0)/60:.1f} minutes")

# ─────────────────────────────────────────────────────────
# Evaluate on test set
# ─────────────────────────────────────────────────────────
print("\nEvaluating on test set...")
results = trainer.evaluate(test_ds)
print(f"\nTest Accuracy : {results['eval_accuracy']:.4f}")
print(f"Test F1 Score : {results['eval_f1']:.4f}")

# Detailed classification report
preds_out   = trainer.predict(test_ds)
y_pred      = np.argmax(preds_out.predictions, axis=-1)
y_true      = test["target"].values

print("\nClassification Report:")
print(classification_report(y_true, y_pred,
                             target_names=["Negative", "Positive"]))

# ─────────────────────────────────────────────────────────
# Save model + tokenizer
# ─────────────────────────────────────────────────────────
save_path = "models/bert_sentiment"
trainer.save_model(save_path)
tokenizer.save_pretrained(save_path)
print(f"\nModel saved to: {save_path}/")

# ─────────────────────────────────────────────────────────
# Plot training history
# ─────────────────────────────────────────────────────────
log_history = trainer.state.log_history
train_loss  = [x["loss"] for x in log_history if "loss" in x and "eval_loss" not in x]
eval_acc    = [x["eval_accuracy"] for x in log_history if "eval_accuracy" in x]
eval_f1     = [x["eval_f1"]       for x in log_history if "eval_f1" in x]

fig, axes = plt.subplots(1, 3, figsize=(15, 4))
fig.suptitle("Phase 7 — BERT Fine-tuning Progress", fontsize=13, fontweight="bold")

if train_loss:
    axes[0].plot(train_loss, color="#1A5C8A", linewidth=2)
    axes[0].set_title("Training Loss"); axes[0].set_xlabel("Step")

if eval_acc:
    axes[1].plot(eval_acc, color="#2A7A4F", marker="o", linewidth=2)
    axes[1].set_ylim(0, 1)
    axes[1].set_title("Validation Accuracy"); axes[1].set_xlabel("Epoch")

if eval_f1:
    axes[2].plot(eval_f1, color="#B5441E", marker="o", linewidth=2)
    axes[2].set_ylim(0, 1)
    axes[2].set_title("Validation F1 Score"); axes[2].set_xlabel("Epoch")

plt.tight_layout()
plt.savefig("outputs/07_bert_training.png", dpi=150, bbox_inches="tight")
plt.show()
print("Saved: outputs/07_bert_training.png")

# ─────────────────────────────────────────────────────────
# Quick inference demo
# ─────────────────────────────────────────────────────────
print("\n--- Inference Demo ---")
sample_tweets = [
    "I absolutely love this product, it changed my life!",
    "This is the worst experience I have ever had, completely broken.",
    "Just got my delivery, pretty good overall",
    "Never buying from this brand again. Total scam.",
    "Had an amazing day at the beach with family!"
]
from transformers import pipeline
classifier = pipeline("text-classification", model=save_path, device=0 if device=="cuda" else -1)
for tweet in sample_tweets:
    result = classifier(tweet)[0]
    emoji  = "+" if result["label"] == "POSITIVE" else "-"
    print(f"[{emoji}] {result['label']:8s} ({result['score']:.2%}) | {tweet[:60]}")

print("\nPhase 7 complete!")
