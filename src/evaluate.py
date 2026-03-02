"""Error analysis and per-keyword evaluation (Exercise 5.2).

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

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay, f1_score, precision_score, recall_score

from data_utils import load_dev, load_fine_grained_labels, DEV_CSV


REPO_ROOT = Path(__file__).parent.parent

# 7 fine-grained PCL category names (same order as the label vector)
PCL_CATEGORIES = [
    "Unbalanced power relations",
    "Shallow solution",
    "Presupposition",
    "Authority voice",
    "Metaphor",
    "Compassion",
    "The poorer the merrier",
]


def read_predictions(path: Path) -> list[int]:
    return [int(line.strip()) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def error_analysis(
    records: list[dict],
    baseline_preds: list[int],
    best_preds: list[int],
) -> None:
    """Print 4-bucket error analysis and save confusion matrix figure."""
    assert len(records) == len(baseline_preds) == len(best_preds), (
        f"Length mismatch: records={len(records)}, baseline={len(baseline_preds)}, best={len(best_preds)}"
    )

    fine_grained = load_fine_grained_labels(DEV_CSV)
    gold = [r["label"] for r in records]

    # 4 buckets
    both_correct    = []
    both_wrong      = []
    only_best       = []   # best correct, baseline wrong
    only_baseline   = []   # baseline correct, best wrong

    for i, (rec, bp, bsp, g) in enumerate(zip(records, best_preds, baseline_preds, gold)):
        best_ok     = (bp  == g)
        base_ok     = (bsp == g)
        if best_ok and base_ok:
            both_correct.append(i)
        elif not best_ok and not base_ok:
            both_wrong.append(i)
        elif best_ok and not base_ok:
            only_best.append(i)
        else:
            only_baseline.append(i)

    total = len(records)
    print("=" * 60)
    print("Error Analysis — 4-Bucket Summary")
    print("=" * 60)
    print(f"Total dev examples:         {total}")
    print(f"Both correct:               {len(both_correct):4d}  ({100*len(both_correct)/total:.1f}%)")
    print(f"Both wrong:                 {len(both_wrong):4d}  ({100*len(both_wrong)/total:.1f}%)")
    print(f"Only best model correct:    {len(only_best):4d}  ({100*len(only_best)/total:.1f}%)")
    print(f"Only baseline correct:      {len(only_baseline):4d}  ({100*len(only_baseline)/total:.1f}%)")
    print()

    def _active_cats(par_id: str) -> str:
        vec = fine_grained.get(par_id, [0] * 7)
        cats = [PCL_CATEGORIES[j] for j, v in enumerate(vec) if v]
        return ", ".join(cats) if cats else "—"

    def _show_examples(indices: list[int], label: str, n: int = 5):
        print(f"--- {label} (showing up to {n}) ---")
        for idx in indices[:n]:
            rec = records[idx]
            snippet = rec["text"][:120].replace("\n", " ")
            gold_lbl = gold[idx]
            cats     = _active_cats(rec["par_id"])
            print(f"  [{idx}] gold={gold_lbl}  cats=[{cats}]")
            print(f"       \"{snippet}\"")
        print()

    _show_examples(only_best,     "Only Best Correct (improvements over baseline)")
    _show_examples(only_baseline, "Only Baseline Correct (regressions)")
    _show_examples(both_wrong,    "Both Wrong (hard cases)")

    # --- Confusion matrices ---
    figures_dir = REPO_ROOT / "writeup" / "figures"
    figures_dir.mkdir(parents=True, exist_ok=True)

    fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    for ax, preds, title in zip(
        axes,
        [baseline_preds, best_preds],
        ["Baseline (RoBERTa, 1-epoch CE)", "Best Model (RoBERTa multi-task, focal)"],
    ):
        cm = confusion_matrix(gold, preds)
        disp = ConfusionMatrixDisplay(cm, display_labels=["No PCL", "PCL"])
        disp.plot(ax=ax, colorbar=False, cmap="Blues")
        f1 = f1_score(gold, preds, pos_label=1, zero_division=0)
        ax.set_title(f"{title}\nDev F1={f1:.4f}", fontsize=10)

    plt.tight_layout()
    out_path = figures_dir / "confusion_matrices.png"
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved confusion matrices -> {out_path}")


def per_keyword_analysis(records: list[dict], best_preds: list[int]) -> None:
    """Compute precision/recall/F1 per keyword and save horizontal bar chart."""
    gold = [r["label"] for r in records]

    # Group indices by keyword
    keyword_indices: dict[str, list[int]] = {}
    for i, rec in enumerate(records):
        kw = rec["keyword"]
        keyword_indices.setdefault(kw, []).append(i)

    print("=" * 60)
    print("Per-Keyword Analysis (best model, dev set)")
    print("=" * 60)
    print(f"{'Keyword':<20} {'N+':>4} {'N-':>5} {'P':>6} {'R':>6} {'F1':>6}")
    print("-" * 60)

    kw_f1s    = {}
    kw_labels = {}

    for kw, idxs in sorted(keyword_indices.items()):
        g    = [gold[i]       for i in idxs]
        p    = [best_preds[i] for i in idxs]
        n_pos = sum(g)
        n_neg = len(g) - n_pos
        prec  = precision_score(g, p, pos_label=1, zero_division=0)
        rec   = recall_score(g,    p, pos_label=1, zero_division=0)
        f1    = f1_score(g,        p, pos_label=1, zero_division=0)
        kw_f1s[kw]    = f1
        kw_labels[kw] = f"{kw} (n+={n_pos})"
        print(f"  {kw:<18} {n_pos:4d} {n_neg:5d} {prec:6.3f} {rec:6.3f} {f1:6.3f}")

    print()

    # --- Horizontal bar chart ---
    sorted_kws = sorted(kw_f1s, key=kw_f1s.get)
    labels = [kw_labels[k] for k in sorted_kws]
    values = [kw_f1s[k]    for k in sorted_kws]

    fig, ax = plt.subplots(figsize=(8, 5))
    bars = ax.barh(labels, values, color="steelblue", edgecolor="white")
    ax.bar_label(bars, fmt="%.3f", padding=3, fontsize=9)
    ax.set_xlabel("F1 (positive class)")
    ax.set_title("Per-Keyword F1 — Best Model on Dev Set")
    ax.set_xlim(0, 1.05)
    ax.axvline(x=np.mean(values), color="crimson", linestyle="--", linewidth=1,
               label=f"mean={np.mean(values):.3f}")
    ax.legend(fontsize=9)
    plt.tight_layout()

    figures_dir = REPO_ROOT / "writeup" / "figures"
    figures_dir.mkdir(parents=True, exist_ok=True)
    out_path = figures_dir / "per_keyword_f1.png"
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved per-keyword F1 chart -> {out_path}")


if __name__ == "__main__":
    dev_records = load_dev()

    baseline_preds = read_predictions(REPO_ROOT / "predictions" / "baseline_dev.txt")
    best_preds     = read_predictions(REPO_ROOT / "dev.txt")

    error_analysis(dev_records, baseline_preds, best_preds)
    per_keyword_analysis(dev_records, best_preds)
