"""Best model — self-contained script for the PCL classification task.

DeBERTa-v3-base fine-tuned with class-weighted cross-entropy on the full
training set, with decision threshold optimised on the dev set.

This script loads the dataset, trains the model end-to-end, and writes
dev.txt and test.txt to the repository root.

Usage:
    python BestModel/model.py
"""

import json
import sys
from pathlib import Path

# Allow importing from src/
REPO_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(REPO_ROOT / "src"))

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset as TorchDataset
from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    Trainer,
    TrainingArguments,
)
from sklearn.metrics import f1_score, precision_score, recall_score

from data_utils import load_train, load_dev, load_test


# ---------------------------------------------------------------------------
# Hyperparameters
# ---------------------------------------------------------------------------

MODEL_NAME = "microsoft/deberta-v3-base"
MAX_LENGTH = 128
NUM_EPOCHS = 4
TRAIN_BATCH_SIZE = 16
EVAL_BATCH_SIZE = 32
LEARNING_RATE = 2e-5
WEIGHT_DECAY = 0.01
WARMUP_RATIO = 0.1

CHECKPOINT_DIR = REPO_ROOT / "checkpoints" / "deberta_best"
CHECKPOINT_DIR.mkdir(parents=True, exist_ok=True)


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
        self.token_type_ids = encodings.get("token_type_ids")

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        item = {
            "input_ids": self.input_ids[idx],
            "attention_mask": self.attention_mask[idx],
            "labels": torch.tensor(self.labels[idx], dtype=torch.long),
        }
        if self.token_type_ids is not None:
            item["token_type_ids"] = self.token_type_ids[idx]
        return item


class InferenceDataset(TorchDataset):
    def __init__(self, texts: list[str], tokenizer, max_length: int = MAX_LENGTH):
        encodings = tokenizer(
            texts,
            truncation=True,
            padding="max_length",
            max_length=max_length,
            return_tensors="pt",
        )
        self.input_ids = encodings["input_ids"]
        self.attention_mask = encodings["attention_mask"]
        self.token_type_ids = encodings.get("token_type_ids")

    def __len__(self):
        return len(self.input_ids)

    def __getitem__(self, idx):
        item = {
            "input_ids": self.input_ids[idx],
            "attention_mask": self.attention_mask[idx],
        }
        if self.token_type_ids is not None:
            item["token_type_ids"] = self.token_type_ids[idx]
        return item


# ---------------------------------------------------------------------------
# Custom Trainer with class-weighted loss
# ---------------------------------------------------------------------------

class WeightedTrainer(Trainer):
    def __init__(self, class_weight: torch.Tensor, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.class_weight = class_weight

    def compute_loss(self, model, inputs, return_outputs=False, **kwargs):
        labels = inputs.pop("labels")
        outputs = model(**inputs)
        logits = outputs.logits
        weight = self.class_weight.to(logits.device)
        loss = F.cross_entropy(logits, labels, weight=weight)
        return (loss, outputs) if return_outputs else loss


# ---------------------------------------------------------------------------
# Metrics & inference
# ---------------------------------------------------------------------------

def compute_metrics(eval_pred):
    logits, labels = eval_pred
    preds = np.argmax(logits, axis=-1)
    return {
        "f1_pos": f1_score(labels, preds, pos_label=1, zero_division=0),
        "precision_pos": precision_score(labels, preds, pos_label=1, zero_division=0),
        "recall_pos": recall_score(labels, preds, pos_label=1, zero_division=0),
    }


def get_probs(records: list[dict], model, tokenizer, device: torch.device) -> np.ndarray:
    texts = [r["text"] for r in records]
    dataset = InferenceDataset(texts, tokenizer)
    loader = DataLoader(dataset, batch_size=EVAL_BATCH_SIZE)
    all_probs = []
    model.eval()
    with torch.no_grad():
        for batch in loader:
            batch = {k: v.to(device) for k, v in batch.items()}
            logits = model(**batch).logits
            probs = torch.softmax(logits, dim=-1)[:, 1].cpu().numpy()
            all_probs.append(probs)
    return np.concatenate(all_probs)


def find_best_threshold(probs: np.ndarray, labels: list[int]) -> tuple[float, float]:
    best_tau, best_f1 = 0.5, 0.0
    for tau in np.arange(0.05, 0.96, 0.05):
        preds = (probs >= tau).astype(int)
        f1 = f1_score(labels, preds, pos_label=1, zero_division=0)
        if f1 > best_f1:
            best_f1, best_tau = f1, float(tau)
    return best_tau, best_f1


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    train_records = load_train()
    dev_records = load_dev()
    test_records = load_test()
    print(f"Train: {len(train_records)}  Dev: {len(dev_records)}  Test: {len(test_records)}")

    # Class weights from training distribution
    n_neg = sum(1 for r in train_records if r["label"] == 0)
    n_pos = sum(1 for r in train_records if r["label"] == 1)
    pos_weight = n_neg / n_pos
    class_weight = torch.tensor([1.0, pos_weight], dtype=torch.float32)
    print(f"Class weights: [No-PCL=1.00, PCL={pos_weight:.2f}]")

    # Tokeniser & datasets
    print(f"\nLoading tokeniser: {MODEL_NAME}")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    train_dataset = PCLDataset(train_records, tokenizer)
    dev_dataset = PCLDataset(dev_records, tokenizer)

    # Model
    print(f"Loading model: {MODEL_NAME}")
    model = AutoModelForSequenceClassification.from_pretrained(MODEL_NAME, num_labels=2)

    # Training args
    total_steps = (len(train_dataset) // TRAIN_BATCH_SIZE) * NUM_EPOCHS
    warmup_steps = int(total_steps * WARMUP_RATIO)

    training_args = TrainingArguments(
        output_dir=str(CHECKPOINT_DIR),
        num_train_epochs=NUM_EPOCHS,
        per_device_train_batch_size=TRAIN_BATCH_SIZE,
        per_device_eval_batch_size=EVAL_BATCH_SIZE,
        learning_rate=LEARNING_RATE,
        warmup_steps=warmup_steps,
        weight_decay=WEIGHT_DECAY,
        eval_strategy="epoch",
        save_strategy="epoch",
        load_best_model_at_end=True,
        metric_for_best_model="f1_pos",
        greater_is_better=True,
        logging_steps=50,
        fp16=torch.cuda.is_available(),
        report_to="none",
    )

    trainer = WeightedTrainer(
        class_weight=class_weight,
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=dev_dataset,
        compute_metrics=compute_metrics,
        tokenizer=tokenizer,
    )

    print("\nTraining ...")
    trainer.train()
    trainer.save_model(str(CHECKPOINT_DIR))
    tokenizer.save_pretrained(str(CHECKPOINT_DIR))

    # Reload best checkpoint for inference
    model = AutoModelForSequenceClassification.from_pretrained(str(CHECKPOINT_DIR))
    model.to(device)

    # Threshold optimisation on dev
    print("\nOptimising threshold on dev set ...")
    dev_probs = get_probs(dev_records, model, tokenizer, device)
    dev_labels = [r["label"] for r in dev_records]
    tau, dev_f1 = find_best_threshold(dev_probs, dev_labels)
    print(f"Best threshold: {tau:.2f}  ->  dev F1 (positive class): {dev_f1:.4f}")

    # Save threshold
    (CHECKPOINT_DIR / "threshold.json").write_text(
        json.dumps({"threshold": tau, "dev_f1": dev_f1}, indent=2)
    )

    # Write dev predictions
    dev_preds = (dev_probs >= tau).astype(int).tolist()
    dev_out = REPO_ROOT / "dev.txt"
    dev_out.write_text("\n".join(str(p) for p in dev_preds) + "\n")
    print(f"Wrote {len(dev_preds)} dev predictions -> {dev_out}")

    # Write test predictions
    test_probs = get_probs(test_records, model, tokenizer, device)
    test_preds = (test_probs >= tau).astype(int).tolist()
    test_out = REPO_ROOT / "test.txt"
    test_out.write_text("\n".join(str(p) for p in test_preds) + "\n")
    print(f"Wrote {len(test_preds)} test predictions -> {test_out}")

    print(f"\nDone. Dev F1: {dev_f1:.4f}")


if __name__ == "__main__":
    main()
