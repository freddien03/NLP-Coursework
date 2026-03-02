r"""Best model -- Multi-task RoBERTa-base with focal loss for PCL detection.

Architecture:
  RoBERTa-base encoder (shared)
        |
    [CLS] token + Dropout(0.1)
      /              \
  Binary Head       Category Head
  (768 -> 2)        (768 -> 7)
  Focal Loss        BCE Loss
      \              /
   Combined: L = L_focal + lambda * L_bce

The binary head detects PCL (0/1). The category head predicts which of the
7 fine-grained PCL categories are present (multi-label). At inference only
the binary head is used; the category head acts as a training-time regulariser.

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
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset as TorchDataset
from transformers import (
    AutoTokenizer,
    AutoModel,
    Trainer,
    TrainingArguments,
)
from sklearn.metrics import f1_score, precision_score, recall_score

from data_utils import load_train, load_dev, load_test, load_fine_grained_labels, TRAIN_CSV, DEV_CSV


# ---------------------------------------------------------------------------
# Hyperparameters
# ---------------------------------------------------------------------------

MODEL_NAME = "roberta-base"
MAX_LENGTH = 128
NUM_EPOCHS = 5
TRAIN_BATCH_SIZE = 16
EVAL_BATCH_SIZE = 32
LEARNING_RATE = 2e-5
WEIGHT_DECAY = 0.01
WARMUP_RATIO = 0.1

FOCAL_GAMMA = 2.0
FOCAL_ALPHA = [1.0, 9.55]  # [weight_neg, weight_pos]
AUX_LOSS_WEIGHT = 0.5      # lambda for auxiliary BCE loss

NUM_CATEGORIES = 7

CHECKPOINT_DIR = REPO_ROOT / "checkpoints" / "roberta_best"
CHECKPOINT_DIR.mkdir(parents=True, exist_ok=True)


# ---------------------------------------------------------------------------
# Focal Loss
# ---------------------------------------------------------------------------

class FocalLoss(nn.Module):
    """Focal loss for binary/multi-class classification.

    FL(p_t) = -alpha_t * (1 - p_t)^gamma * log(p_t)

    Focuses training on hard, misclassified examples by down-weighting
    well-classified ones.
    """
    def __init__(self, gamma: float = 2.0, alpha: list[float] | None = None):
        super().__init__()
        self.gamma = gamma
        if alpha is not None:
            self.register_buffer("alpha", torch.tensor(alpha, dtype=torch.float32))
        else:
            self.alpha = None

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        ce_loss = F.cross_entropy(logits, targets, reduction="none")
        pt = torch.exp(-ce_loss)  # p_t = probability of correct class
        focal_weight = (1 - pt) ** self.gamma

        if self.alpha is not None:
            alpha_t = self.alpha.to(logits.device)[targets]
            focal_weight = alpha_t * focal_weight

        return (focal_weight * ce_loss).mean()


# ---------------------------------------------------------------------------
# Multi-Task Model
# ---------------------------------------------------------------------------

class MultiTaskModel(nn.Module):
    """Transformer encoder with two classification heads.

    - binary_head: 2-class PCL detection
    - category_head: 7-class multi-label PCL category prediction
    """
    def __init__(self, model_name: str = MODEL_NAME, num_categories: int = NUM_CATEGORIES,
                 prior_prob: float = 0.0948):
        super().__init__()
        self.encoder = AutoModel.from_pretrained(model_name, dtype=torch.float32)
        hidden_size = self.encoder.config.hidden_size
        self.dropout = nn.Dropout(0.1)
        self.binary_head = nn.Linear(hidden_size, 2)
        self.category_head = nn.Linear(hidden_size, num_categories)
        # Initialise binary head bias to class prior so the model predicts ~π at
        # the start of training rather than ~0.5. This prevents the large initial
        # loss spike from the majority class from collapsing training (Lin et al. 2017).
        import math
        self.binary_head.bias.data[1] = math.log(prior_prob / (1 - prior_prob))

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        token_type_ids: torch.Tensor | None = None,
        labels: torch.Tensor | None = None,
        category_labels: torch.Tensor | None = None,
    ) -> dict:
        encoder_kwargs = {"input_ids": input_ids, "attention_mask": attention_mask}
        if token_type_ids is not None:
            encoder_kwargs["token_type_ids"] = token_type_ids

        outputs = self.encoder(**encoder_kwargs)
        cls_output = self.dropout(outputs.last_hidden_state[:, 0])

        binary_logits = self.binary_head(cls_output)
        category_logits = self.category_head(cls_output)

        return {
            "binary_logits": binary_logits,
            "category_logits": category_logits,
        }


# ---------------------------------------------------------------------------
# Dataset
# ---------------------------------------------------------------------------

class MultiTaskPCLDataset(TorchDataset):
    """Dataset returning tokenized inputs + binary label + 7-element category labels."""

    def __init__(
        self,
        records: list[dict],
        fine_grained: dict[str, list[int]],
        tokenizer,
        max_length: int = MAX_LENGTH,
    ):
        self.labels = [r["label"] for r in records]
        # Map par_id -> category vector; default to all-zeros for non-PCL
        self.category_labels = [
            fine_grained.get(r["par_id"], [0] * NUM_CATEGORIES)
            for r in records
        ]
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
            "category_labels": torch.tensor(self.category_labels[idx], dtype=torch.float32),
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
# Custom Trainer with multi-task loss
# ---------------------------------------------------------------------------

class MultiTaskTrainer(Trainer):
    """Trainer that computes focal loss (binary) + BCE loss (category)."""

    def __init__(self, focal_loss: FocalLoss, aux_weight: float, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.focal_loss = focal_loss
        self.aux_weight = aux_weight

    def compute_loss(self, model, inputs, return_outputs=False, **kwargs):
        labels = inputs.pop("labels")
        category_labels = inputs.pop("category_labels")

        outputs = model(**inputs)
        binary_logits = outputs["binary_logits"]
        category_logits = outputs["category_logits"]

        # Primary loss: focal loss on binary classification
        loss_focal = self.focal_loss(binary_logits, labels)

        # Auxiliary loss: BCE on 7-category multi-label classification
        loss_bce = F.binary_cross_entropy_with_logits(category_logits, category_labels)

        loss = loss_focal + self.aux_weight * loss_bce

        return (loss, outputs) if return_outputs else loss

    def save_model(self, output_dir=None, _internal_call=False):
        """Save custom model via torch.save since it's not a PreTrainedModel."""
        if output_dir is None:
            output_dir = self.args.output_dir
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        torch.save(self.model.state_dict(), output_dir / "model.pt")


# ---------------------------------------------------------------------------
# Metrics & inference
# ---------------------------------------------------------------------------

def compute_metrics(eval_pred):
    predictions, label_ids = eval_pred
    # predictions is a tuple: (binary_logits, category_logits)
    # label_ids is a tuple: (binary_labels, category_labels)
    if isinstance(predictions, tuple):
        binary_logits = predictions[0]
    else:
        binary_logits = predictions

    if isinstance(label_ids, tuple):
        labels = label_ids[0]
    else:
        labels = label_ids

    preds = np.argmax(binary_logits, axis=-1)
    return {
        "f1_pos": f1_score(labels, preds, pos_label=1, zero_division=0),
        "precision_pos": precision_score(labels, preds, pos_label=1, zero_division=0),
        "recall_pos": recall_score(labels, preds, pos_label=1, zero_division=0),
    }


def get_probs(
    records: list[dict], model, tokenizer, device: torch.device
) -> np.ndarray:
    """Return softmax P(label=1) for each record using the binary head."""
    texts = [r["text"] for r in records]
    dataset = InferenceDataset(texts, tokenizer)
    loader = DataLoader(dataset, batch_size=EVAL_BATCH_SIZE)
    all_probs = []
    model.eval()
    with torch.no_grad():
        for batch in loader:
            batch = {k: v.to(device) for k, v in batch.items()}
            outputs = model(**batch)
            binary_logits = outputs["binary_logits"]
            probs = torch.softmax(binary_logits, dim=-1)[:, 1].cpu().numpy()
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

    # Load data
    train_records = load_train()
    dev_records = load_dev()
    test_records = load_test()
    print(f"Train: {len(train_records)}  Dev: {len(dev_records)}  Test: {len(test_records)}")

    # Load fine-grained category labels
    train_fg = load_fine_grained_labels(TRAIN_CSV)
    dev_fg = load_fine_grained_labels(DEV_CSV)
    print(f"Fine-grained labels: {len(train_fg)} train, {len(dev_fg)} dev")

    # Class distribution
    n_neg = sum(1 for r in train_records if r["label"] == 0)
    n_pos = sum(1 for r in train_records if r["label"] == 1)
    pos_weight = n_neg / n_pos
    print(f"Class ratio: {n_neg} neg / {n_pos} pos = {pos_weight:.2f}:1")

    # Tokeniser & datasets
    print(f"\nLoading tokeniser: {MODEL_NAME}")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    train_dataset = MultiTaskPCLDataset(train_records, train_fg, tokenizer)
    dev_dataset = MultiTaskPCLDataset(dev_records, dev_fg, tokenizer)

    # Model
    print(f"Loading model: {MODEL_NAME}")
    prior_prob = n_pos / (n_pos + n_neg)
    model = MultiTaskModel(MODEL_NAME, prior_prob=prior_prob)

    # Focal loss
    focal_loss = FocalLoss(gamma=FOCAL_GAMMA, alpha=FOCAL_ALPHA)

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
        load_best_model_at_end=False,  # manual loading since custom model
        metric_for_best_model="f1_pos",
        greater_is_better=True,
        save_total_limit=1,
        logging_steps=50,
        fp16=torch.cuda.is_available(),
        report_to="none",
    )

    trainer = MultiTaskTrainer(
        focal_loss=focal_loss,
        aux_weight=AUX_LOSS_WEIGHT,
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=dev_dataset,
        compute_metrics=compute_metrics,
        processing_class=tokenizer,
    )

    print("\nTraining ...")
    trainer.train()

    # Save final model
    torch.save(model.state_dict(), CHECKPOINT_DIR / "model.pt")
    tokenizer.save_pretrained(str(CHECKPOINT_DIR))

    # Load best checkpoint
    best_ckpt = trainer.state.best_model_checkpoint
    if best_ckpt:
        print(f"\nLoading best checkpoint: {best_ckpt}")
        state_dict = torch.load(Path(best_ckpt) / "model.pt", map_location=device, weights_only=True)
        model.load_state_dict(state_dict)
        # Also save the best weights to the main checkpoint dir
        torch.save(state_dict, CHECKPOINT_DIR / "model.pt")
    else:
        print("\nNo best checkpoint found, using final model.")

    model.to(device)

    # Threshold optimisation on dev
    print("\nOptimising threshold on dev set ...")
    dev_probs = get_probs(dev_records, model, tokenizer, device)
    dev_labels = [r["label"] for r in dev_records]
    tau, dev_f1 = find_best_threshold(dev_probs, dev_labels)
    print(f"Best threshold: {tau:.2f}  ->  dev F1 (positive class): {dev_f1:.4f}")

    # Save threshold and training info
    (CHECKPOINT_DIR / "threshold.json").write_text(
        json.dumps({"threshold": tau, "dev_f1": dev_f1}, indent=2)
    )
    (CHECKPOINT_DIR / "train_info.json").write_text(
        json.dumps({
            "model": MODEL_NAME,
            "architecture": "multi-task (binary + 7-category)",
            "loss": f"focal(gamma={FOCAL_GAMMA}) + {AUX_LOSS_WEIGHT}*BCE",
            "epochs": NUM_EPOCHS,
            "batch_size": TRAIN_BATCH_SIZE,
            "lr": LEARNING_RATE,
            "threshold": tau,
            "dev_f1": dev_f1,
        }, indent=2)
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
