"""Model training script (Exercise 4).

Trains DeBERTa-v3-base with class-weighted cross-entropy on the full train
split, evaluates on dev each epoch, and saves the best checkpoint (by dev F1
of the positive class) to checkpoints/deberta_best/.

Usage:
    python src/train.py
"""

from pathlib import Path
import json

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset as TorchDataset
from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    Trainer,
    TrainingArguments,
)
from sklearn.metrics import f1_score, precision_score, recall_score

from data_utils import load_train, load_dev


MODEL_NAME = "microsoft/deberta-v3-base"
MAX_LENGTH = 128
CHECKPOINT_DIR = Path(__file__).parent.parent / "checkpoints" / "deberta_best"
CHECKPOINT_DIR.mkdir(parents=True, exist_ok=True)


# ---------------------------------------------------------------------------
# Dataset wrapper
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
        # DeBERTa-v3 does not use token_type_ids
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


# ---------------------------------------------------------------------------
# Custom Trainer with class-weighted loss
# ---------------------------------------------------------------------------

class WeightedTrainer(Trainer):
    """Trainer that applies class weights to cross-entropy to address imbalance."""

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
# Main training function
# ---------------------------------------------------------------------------

def train(train_records: list[dict], dev_records: list[dict]) -> None:
    # --- Class weights ---
    n_neg = sum(1 for r in train_records if r["label"] == 0)
    n_pos = sum(1 for r in train_records if r["label"] == 1)
    pos_weight = n_neg / n_pos
    class_weight = torch.tensor([1.0, pos_weight], dtype=torch.float32)
    print(f"Class weights: [No-PCL=1.00, PCL={pos_weight:.2f}]  (n_neg={n_neg}, n_pos={n_pos})")

    # --- Tokeniser & datasets ---
    print(f"Loading tokeniser: {MODEL_NAME}")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

    print("Tokenising train ...")
    train_dataset = PCLDataset(train_records, tokenizer)
    print("Tokenising dev ...")
    dev_dataset = PCLDataset(dev_records, tokenizer)

    # --- Model ---
    print(f"Loading model: {MODEL_NAME}")
    model = AutoModelForSequenceClassification.from_pretrained(MODEL_NAME, num_labels=2)

    # --- Training arguments ---
    total_steps = (len(train_dataset) // 16) * 4
    warmup_steps = int(total_steps * 0.1)

    training_args = TrainingArguments(
        output_dir=str(CHECKPOINT_DIR),
        num_train_epochs=4,
        per_device_train_batch_size=16,
        per_device_eval_batch_size=32,
        learning_rate=2e-5,
        warmup_steps=warmup_steps,
        weight_decay=0.01,
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

    # --- Train ---
    print("\nStarting training ...")
    trainer.train()

    # --- Save best model ---
    trainer.save_model(str(CHECKPOINT_DIR))
    tokenizer.save_pretrained(str(CHECKPOINT_DIR))
    print(f"\nBest model saved to {CHECKPOINT_DIR}")

    # --- Final dev evaluation ---
    metrics = trainer.evaluate()
    best_f1 = metrics.get("eval_f1_pos", 0.0)
    print(f"Best dev F1 (positive class): {best_f1:.4f}")

    # Persist key info for predict.py
    info = {"model_name": MODEL_NAME, "max_length": MAX_LENGTH, "dev_f1": best_f1}
    (CHECKPOINT_DIR / "train_info.json").write_text(json.dumps(info, indent=2))


if __name__ == "__main__":
    train_records = load_train()
    dev_records = load_dev()

    print(f"Train: {len(train_records)} examples")
    print(f"Dev:   {len(dev_records)} examples")

    train(train_records, dev_records)
