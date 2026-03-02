"""Generate dev.txt and test.txt prediction files (Exercise 5.1).

Loads the best multi-task DeBERTa checkpoint, sweeps the decision threshold on
the dev set to maximise positive-class F1, then writes predictions for both
dev and test splits.

Output format: one prediction per line, 0 or 1.
Files are written to the repository root as required by the spec.

Usage:
    python src/predict.py
"""

import sys
from pathlib import Path
import json

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset as TorchDataset
from transformers import AutoTokenizer
from sklearn.metrics import f1_score

from data_utils import load_dev, load_test

REPO_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(REPO_ROOT / "BestModel"))

from model import MultiTaskModel

CHECKPOINT_DIR = REPO_ROOT / "checkpoints" / "roberta_best"
MAX_LENGTH = 128
BATCH_SIZE = 32


# ---------------------------------------------------------------------------
# Minimal inference dataset (no labels required)
# ---------------------------------------------------------------------------

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
# Inference helpers
# ---------------------------------------------------------------------------

def _get_probs(
    records: list[dict],
    model,
    tokenizer,
    device: torch.device,
) -> np.ndarray:
    """Return softmax P(label=1) for each record using the binary head."""
    texts = [r["text"] for r in records]
    dataset = InferenceDataset(texts, tokenizer)
    loader = DataLoader(dataset, batch_size=BATCH_SIZE)

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


def find_best_threshold(
    dev_records: list[dict],
    model,
    tokenizer,
    device: torch.device,
) -> tuple[float, float, np.ndarray]:
    """Sweep threshold on dev set; return (best_tau, best_f1, probs)."""
    probs = _get_probs(dev_records, model, tokenizer, device)
    labels = [r["label"] for r in dev_records]

    best_tau, best_f1 = 0.5, 0.0
    for tau in np.arange(0.05, 0.96, 0.05):
        preds = (probs >= tau).astype(int)
        f1 = f1_score(labels, preds, pos_label=1, zero_division=0)
        if f1 > best_f1:
            best_f1, best_tau = f1, float(tau)

    return best_tau, best_f1, probs


def predict(records: list[dict], tau: float | None = None) -> list[int]:
    """Load checkpoint, apply threshold, return binary predictions."""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    tokenizer = AutoTokenizer.from_pretrained(str(CHECKPOINT_DIR))
    model = MultiTaskModel()
    state_dict = torch.load(
        CHECKPOINT_DIR / "model.pt", map_location=device, weights_only=True
    )
    model.load_state_dict(state_dict)
    model.to(device)

    # Threshold: use provided value, or read persisted one, or default 0.5
    if tau is None:
        tau_path = CHECKPOINT_DIR / "threshold.json"
        if tau_path.exists():
            tau = json.loads(tau_path.read_text())["threshold"]
        else:
            tau = 0.5

    probs = _get_probs(records, model, tokenizer, device)
    return (probs >= tau).astype(int).tolist()


def write_predictions(preds: list[int], out_path: Path) -> None:
    out_path.write_text("\n".join(str(p) for p in preds) + "\n", encoding="utf-8")
    print(f"Wrote {len(preds)} predictions to {out_path}")


if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    tokenizer = AutoTokenizer.from_pretrained(str(CHECKPOINT_DIR))
    model = MultiTaskModel()
    state_dict = torch.load(
        CHECKPOINT_DIR / "model.pt", map_location=device, weights_only=True
    )
    model.load_state_dict(state_dict)
    model.to(device)

    # Tune threshold on dev
    dev_records = load_dev()
    print(f"Sweeping threshold on {len(dev_records)} dev examples ...")
    tau, dev_f1, dev_probs = find_best_threshold(dev_records, model, tokenizer, device)
    print(f"Best threshold: {tau:.2f}  ->  dev F1 (positive class): {dev_f1:.4f}")

    # Persist threshold
    (CHECKPOINT_DIR / "threshold.json").write_text(
        json.dumps({"threshold": tau, "dev_f1": dev_f1}, indent=2)
    )

    # Dev predictions (reuse probs already computed during threshold sweep)
    dev_preds = (dev_probs >= tau).astype(int).tolist()
    write_predictions(dev_preds, REPO_ROOT / "dev.txt")

    # Test predictions
    test_records = load_test()
    print(f"Predicting on {len(test_records)} test examples ...")
    test_probs = _get_probs(test_records, model, tokenizer, device)
    test_preds = (test_probs >= tau).astype(int).tolist()
    write_predictions(test_preds, REPO_ROOT / "test.txt")
