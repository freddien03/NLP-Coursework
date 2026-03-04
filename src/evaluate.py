"""Error analysis and per-keyword evaluation (Exercise 5.2).

Loads best model, ablation, and baseline predictions on the dev set and analyses
failure cases across four scenarios:
  - Both correct
  - Both incorrect
  - Only best model correct
  - Only other model correct

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


def _four_buckets(records, gold, preds_a, preds_b):
    """Return (both_correct, both_wrong, only_a, only_b) index lists."""
    both_correct = []
    both_wrong   = []
    only_a       = []
    only_b       = []
    for i, (rec, pa, pb, g) in enumerate(zip(records, preds_a, preds_b, gold)):
        a_ok = (pa == g)
        b_ok = (pb == g)
        if a_ok and b_ok:
            both_correct.append(i)
        elif not a_ok and not b_ok:
            both_wrong.append(i)
        elif a_ok and not b_ok:
            only_a.append(i)
        else:
            only_b.append(i)
    return both_correct, both_wrong, only_a, only_b


def error_analysis(
    records: list[dict],
    baseline_preds: list[int],
    ablation_preds: list[int],
    best_preds: list[int],
) -> None:
    """Print 4-bucket error analysis for two comparisons and save confusion matrix figure."""
    assert len(records) == len(baseline_preds) == len(ablation_preds) == len(best_preds), (
        f"Length mismatch: records={len(records)}, baseline={len(baseline_preds)}, "
        f"ablation={len(ablation_preds)}, best={len(best_preds)}"
    )

    fine_grained = load_fine_grained_labels(DEV_CSV)
    gold = [r["label"] for r in records]

    # --- Summary metrics table ---
    print("=" * 60)
    print("Model Summary — Dev Set Metrics")
    print("=" * 60)
    print(f"{'Model':<30} {'F1':>6} {'P':>6} {'R':>6} {'Threshold':>10}")
    print("-" * 60)
    for name, preds, tau in [
        ("Baseline (1-epoch CE)", baseline_preds, 0.50),
        ("Ablation (no cat. head)", ablation_preds, 0.65),
        ("Best model (multi-task)", best_preds, 0.25),
    ]:
        f1  = f1_score(gold, preds, pos_label=1, zero_division=0)
        p   = precision_score(gold, preds, pos_label=1, zero_division=0)
        r   = recall_score(gold, preds, pos_label=1, zero_division=0)
        print(f"  {name:<28} {f1:6.4f} {p:6.4f} {r:6.4f} {tau:10.2f}")
    print()

    total = len(records)

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

    def _print_bucket_table(label_a, label_b, both_correct, both_wrong, only_a, only_b):
        print("=" * 60)
        print(f"4-Bucket Comparison: {label_a} vs {label_b}")
        print("=" * 60)
        print(f"Total dev examples: {total}")
        print(f"Both correct:              {len(both_correct):4d}  ({100*len(both_correct)/total:.1f}%)")
        print(f"Both wrong:                {len(both_wrong):4d}  ({100*len(both_wrong)/total:.1f}%)")
        print(f"Only {label_a} correct:  {len(only_a):4d}  ({100*len(only_a)/total:.1f}%)")
        print(f"Only {label_b} correct: {len(only_b):4d}  ({100*len(only_b)/total:.1f}%)")
        print()

    # --- Best vs Baseline ---
    bc_bl, bw_bl, only_best_bl, only_base = _four_buckets(records, gold, best_preds, baseline_preds)
    _print_bucket_table("best", "baseline", bc_bl, bw_bl, only_best_bl, only_base)
    _show_examples(only_best_bl, "Only Best Correct (improvements over baseline)")
    _show_examples(only_base,    "Only Baseline Correct (regressions vs baseline)")
    _show_examples(bw_bl,        "Both Wrong — Best vs Baseline (hard cases)")

    # --- Best vs Ablation ---
    bc_ab, bw_ab, only_best_ab, only_abl = _four_buckets(records, gold, best_preds, ablation_preds)
    _print_bucket_table("best", "ablation", bc_ab, bw_ab, only_best_ab, only_abl)
    _show_examples(only_best_ab, "Only Best Correct (improvements over ablation)")
    _show_examples(only_abl,     "Only Ablation Correct (regressions vs ablation)")
    _show_examples(bw_ab,        "Both Wrong — Best vs Ablation (hard cases)")

    # --- Confusion matrices (3 panels) ---
    figures_dir = REPO_ROOT / "writeup" / "figures"
    figures_dir.mkdir(parents=True, exist_ok=True)

    fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    for ax, preds, title in zip(
        axes,
        [baseline_preds, best_preds],
        ["Baseline (1-epoch CE)", "Best Model (multi-task)"],
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
    ablation_preds = read_predictions(REPO_ROOT / "predictions" / "ablation_no_category_dev.txt")
    best_preds     = read_predictions(REPO_ROOT / "dev.txt")

    error_analysis(dev_records, baseline_preds, ablation_preds, best_preds)
    per_keyword_analysis(dev_records, best_preds)
