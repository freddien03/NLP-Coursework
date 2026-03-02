"""Error analysis and ablation studies (Exercise 5.2).

Loads best model and baseline predictions on the dev set and analyses
failure cases across four scenarios:
  - Both correct
  - Both incorrect
  - Only best model correct
  - Only baseline correct

Usage:
    python src/evaluate.py
"""

from pathlib import Path

from data_utils import load_dev, load_train, load_fine_grained_labels, DEV_CSV


REPO_ROOT = Path(__file__).parent.parent


def read_predictions(path: Path) -> list[int]:
    return [int(line.strip()) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def error_analysis(
    records: list[dict],
    baseline_preds: list[int],
    best_preds: list[int],
) -> None:
    """TODO: print and save error analysis across the four scenarios."""
    raise NotImplementedError


def ablation_study(train_records: list[dict], dev_records: list[dict]) -> None:
    """TODO: systematically remove model components and report dev F1."""
    raise NotImplementedError


if __name__ == "__main__":
    dev_records = load_dev()

    baseline_preds = read_predictions(REPO_ROOT / "predictions" / "baseline_dev.txt")
    best_preds = read_predictions(REPO_ROOT / "dev.txt")

    error_analysis(dev_records, baseline_preds, best_preds)
    ablation_study(load_train(), dev_records)
