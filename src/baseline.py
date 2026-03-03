r"""Baseline: RoBERTa-base binary classifier, 1 epoch, cross-entropy.

Replicates the notebook baseline (Reconstruct_and_RoBERTa_baseline_train_dev_dataset.ipynb):
  - Downsampled training set: all positives + 2x negatives (mirrors notebook 2:1 ratio)
  - 1 epoch of fine-tuning
  - Standard cross-entropy loss (no focal loss)
  - No auxiliary category head
  - Default threshold 0.5 (no tuning)

Outputs dev predictions to predictions/baseline_dev.txt for use with src/evaluate.py.

Usage:
    python BestModel/baseline.py
"""

import random
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(REPO_ROOT / "src"))

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset as TorchDataset
from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    Trainer,
    TrainingArguments,
)
from sklearn.metrics import f1_score, precision_score, recall_score

from data_utils import load_train, load_dev


# ---------------------------------------------------------------------------
# Hyperparameters — match notebook: 1 epoch, standard CE, downsampled 2:1
# ---------------------------------------------------------------------------

MODEL_NAME = "roberta-base"
MAX_LENGTH = 128
NUM_EPOCHS = 1
TRAIN_BATCH_SIZE = 16
EVAL_BATCH_SIZE = 32
LEARNING_RATE = 2e-5
WEIGHT_DECAY = 0.01
THRESHOLD = 0.5

CHECKPOINT_DIR = REPO_ROOT / "checkpoints" / "roberta_baseline"
CHECKPOINT_DIR.mkdir(parents=True, exist_ok=True)

PREDICTIONS_DIR = REPO_ROOT / "predictions"
PREDICTIONS_DIR.mkdir(parents=True, exist_ok=True)


# ---------------------------------------------------------------------------
# Dataset
# ---------------------------------------------------------------------------

class PCLDataset(TorchDataset):
    def __init__(self, records: list[dict], tokenizer, max_length: int = MAX_LENGTH):
        self.labels = [r["label"] for r in records]
        encodings = tokenizer(
            [r["text"] for r in records],
            truncation=True,
            padding="max_length",
            max_length=max_length,
            return_tensors="pt",
        )
        self.input_ids = encodings["input_ids"]
        self.attention_mask = encodings["attention_mask"]

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        return {
            "input_ids": self.input_ids[idx],
            "attention_mask": self.attention_mask[idx],
            "labels": torch.tensor(self.labels[idx], dtype=torch.long),
        }


# ---------------------------------------------------------------------------
# Metrics
# ---------------------------------------------------------------------------

def compute_metrics(eval_pred):
    logits, labels = eval_pred
    preds = np.argmax(logits, axis=-1)
    return {
        "f1_pos": f1_score(labels, preds, pos_label=1, zero_division=0),
        "precision_pos": precision_score(labels, preds, pos_label=1, zero_division=0),
        "recall_pos": recall_score(labels, preds, pos_label=1, zero_division=0),
    }


# ---------------------------------------------------------------------------
# Inference
# ---------------------------------------------------------------------------

def get_probs(records: list[dict], model, tokenizer, device: torch.device) -> np.ndarray:
    encodings = tokenizer(
        [r["text"] for r in records],
        truncation=True,
        padding="max_length",
        max_length=MAX_LENGTH,
        return_tensors="pt",
    )
    input_ids = encodings["input_ids"]
    attention_mask = encodings["attention_mask"]

    all_probs = []
    model.eval()
    with torch.no_grad():
        for start in range(0, len(input_ids), EVAL_BATCH_SIZE):
            batch = {
                "input_ids": input_ids[start:start + EVAL_BATCH_SIZE].to(device),
                "attention_mask": attention_mask[start:start + EVAL_BATCH_SIZE].to(device),
            }
            logits = model(**batch).logits
            probs = torch.softmax(logits, dim=-1)[:, 1].cpu().numpy()
            all_probs.append(probs)
    return np.concatenate(all_probs)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    train_records = load_train()
    dev_records = load_dev()

    # Downsample negatives: keep all positives + 2x negatives (notebook approach)
    positives = [r for r in train_records if r["label"] == 1]
    negatives = [r for r in train_records if r["label"] == 0]
    random.seed(42)
    negatives_sampled = random.sample(negatives, min(len(positives) * 2, len(negatives)))
    downsampled = positives + negatives_sampled
    random.shuffle(downsampled)

    n_pos = len(positives)
    n_neg = len(negatives_sampled)
    print(f"Train (original):  {len(train_records)} ({sum(r['label']==1 for r in train_records)} pos)")
    print(f"Train (downsampled): {len(downsampled)} ({n_pos} pos / {n_neg} neg, ratio {n_neg/n_pos:.1f}:1)")
    print(f"Dev:  {len(dev_records)}")

    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    train_dataset = PCLDataset(downsampled, tokenizer)
    dev_dataset = PCLDataset(dev_records, tokenizer)

    model = AutoModelForSequenceClassification.from_pretrained(MODEL_NAME, num_labels=2)

    training_args = TrainingArguments(
        output_dir=str(CHECKPOINT_DIR),
        num_train_epochs=NUM_EPOCHS,
        per_device_train_batch_size=TRAIN_BATCH_SIZE,
        per_device_eval_batch_size=EVAL_BATCH_SIZE,
        learning_rate=LEARNING_RATE,
        weight_decay=WEIGHT_DECAY,
        eval_strategy="epoch",
        save_strategy="epoch",
        load_best_model_at_end=False,
        logging_steps=50,
        fp16=torch.cuda.is_available(),
        report_to="none",
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=dev_dataset,
        compute_metrics=compute_metrics,
        processing_class=tokenizer,
    )

    print("\nTraining baseline (1 epoch, cross-entropy, 2:1 downsampled) ...")
    trainer.train()

    model.to(device)

    # Evaluate on dev at threshold 0.5
    dev_probs = get_probs(dev_records, model, tokenizer, device)
    dev_preds = (dev_probs >= THRESHOLD).astype(int)
    dev_labels = [r["label"] for r in dev_records]

    dev_f1 = f1_score(dev_labels, dev_preds, pos_label=1, zero_division=0)
    dev_prec = precision_score(dev_labels, dev_preds, pos_label=1, zero_division=0)
    dev_rec = recall_score(dev_labels, dev_preds, pos_label=1, zero_division=0)

    print(f"\nDev results (threshold={THRESHOLD}):")
    print(f"  F1={dev_f1:.4f}  Precision={dev_prec:.4f}  Recall={dev_rec:.4f}")

    # Write predictions
    out_path = PREDICTIONS_DIR / "baseline_dev.txt"
    out_path.write_text("\n".join(str(p) for p in dev_preds.tolist()) + "\n")
    print(f"\nWrote {len(dev_preds)} predictions -> {out_path}")


if __name__ == "__main__":
    main()
