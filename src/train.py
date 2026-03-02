"""Train the multi-task DeBERTa model for PCL detection.

Delegates to BestModel/model.py which contains the full training pipeline.

Usage:
    python src/train.py
"""

import sys
from pathlib import Path

REPO_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(REPO_ROOT / "BestModel"))

from data_utils import load_train, load_dev
from model import main as run_training


def train(train_records: list[dict], dev_records: list[dict]) -> None:
    # The full pipeline (data loading, training, prediction) is handled
    # by BestModel/model.py's main(). It loads data internally, so we
    # just call it directly.
    run_training()


if __name__ == "__main__":
    train_records = load_train()
    dev_records = load_dev()

    print(f"Train: {len(train_records)} examples")
    print(f"Dev:   {len(dev_records)} examples")

    train(train_records, dev_records)
