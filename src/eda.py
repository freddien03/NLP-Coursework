"""Exploratory Data Analysis for the PCL dataset (Exercise 2).

Run this script to produce figures saved to figures/.
Two EDA techniques are applied:
  1. Class Distribution + Token Length (Basic Statistical Profiling)
  2. N-gram Analysis — Top Bigrams by Class (Lexical Analysis)
"""

import collections
import re
import string
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import numpy as np

from data_utils import load_train


FIGURES_DIR = Path(__file__).parent.parent / "writeup" / "figures"
FIGURES_DIR.mkdir(exist_ok=True)

# Compact stop-word list — suppresses filler words so content bigrams surface
_STOP_WORDS = {
    "a", "an", "the", "and", "or", "but", "in", "on", "at", "to", "for",
    "of", "with", "by", "from", "as", "is", "was", "are", "were", "be",
    "been", "being", "have", "has", "had", "do", "does", "did", "will",
    "would", "could", "should", "may", "might", "shall", "that", "this",
    "it", "its", "he", "she", "they", "we", "i", "you", "his", "her",
    "their", "our", "my", "your", "not", "no", "so", "if", "up", "out",
    "about", "who", "which", "than", "more", "also", "can", "into",
    "there", "when", "all", "said", "s", "re", "t",
}


def technique_1(train_records: list[dict]) -> None:
    """Class Distribution + Token Length Distribution.

    Produces a two-panel figure:
      Left:  bar chart of label-0 vs label-1 counts with exact counts/percentages
      Right: overlapping normalised histograms of whitespace token lengths
             with vertical mean lines for each class

    Saves: figures/technique1_class_dist_token_length.png
    """
    # --- 1. Compute statistics ---
    records_0 = [r for r in train_records if r["label"] == 0]
    records_1 = [r for r in train_records if r["label"] == 1]
    n0, n1 = len(records_0), len(records_1)
    total = len(train_records)

    lengths_0 = [len(r["text"].split()) for r in records_0]
    lengths_1 = [len(r["text"].split()) for r in records_1]

    mean_0, median_0, p95_0 = np.mean(lengths_0), np.median(lengths_0), np.percentile(lengths_0, 95)
    mean_1, median_1, p95_1 = np.mean(lengths_1), np.median(lengths_1), np.percentile(lengths_1, 95)

    # --- 2. Print summary ---
    print("=== Technique 1: Class Distribution + Token Length ===")
    print(f"  Label 0 (No PCL): {n0:,} examples ({100*n0/total:.1f}%)")
    print(f"  Label 1 (PCL):    {n1:,} examples ({100*n1/total:.1f}%)")
    print(f"  Imbalance ratio:  {n0/n1:.2f}:1")
    print(f"  Token lengths (whitespace-split words):")
    print(f"    No PCL — mean: {mean_0:.1f}  median: {median_0:.0f}  p95: {p95_0:.0f}")
    print(f"    PCL    — mean: {mean_1:.1f}  median: {median_1:.0f}  p95: {p95_1:.0f}")

    # --- 3. Build figure ---
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    fig.suptitle(
        "PCL Dataset — Class Distribution and Token Length (Training Split)",
        fontsize=13, fontweight="bold",
    )

    # Left panel: class distribution bar chart
    ax = axes[0]
    bars = ax.bar(
        ["No PCL (0)", "PCL (1)"],
        [n0, n1],
        color=["steelblue", "tomato"],
        edgecolor="black",
        linewidth=0.8,
    )
    for bar, count in zip(bars, [n0, n1]):
        pct = 100.0 * count / total
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + max(n0, n1) * 0.01,
            f"{count:,}\n({pct:.1f}%)",
            ha="center", va="bottom", fontsize=10,
        )
    ax.set_title("Class Distribution", fontsize=11)
    ax.set_ylabel("Number of examples")
    ax.set_ylim(0, max(n0, n1) * 1.2)
    ax.yaxis.set_major_formatter(ticker.FuncFormatter(lambda x, _: f"{int(x):,}"))

    # Right panel: overlapping token length histograms
    ax = axes[1]
    clip = int(np.percentile(lengths_0 + lengths_1, 98))
    bins = np.arange(0, clip + 10, 10)
    ax.hist(lengths_0, bins=bins, density=True, alpha=0.55,
            color="steelblue", label=f"No PCL (n={n0:,})")
    ax.hist(lengths_1, bins=bins, density=True, alpha=0.55,
            color="tomato", label=f"PCL (n={n1:,})")
    ax.axvline(mean_0, color="steelblue", linestyle="--", linewidth=1.4,
               label=f"Mean No PCL: {mean_0:.0f}")
    ax.axvline(mean_1, color="tomato", linestyle="--", linewidth=1.4,
               label=f"Mean PCL: {mean_1:.0f}")
    ax.set_title("Token Length Distribution by Class", fontsize=11)
    ax.set_xlabel("Token count (whitespace-split)")
    ax.set_ylabel("Density")
    ax.set_xlim(0, clip)
    ax.legend(fontsize=9)

    plt.tight_layout()
    out_path = FIGURES_DIR / "technique1_class_dist_token_length.png"
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {out_path}")


def technique_2(train_records: list[dict]) -> None:
    """N-gram Analysis — Top Bigrams by Class (Lexical Analysis).

    Compares the most common bigrams in PCL=1 vs PCL=0 examples after
    filtering stop words, revealing discriminative phrases and whether
    specific word patterns trivially identify the class.

    Produces a side-by-side horizontal bar chart of the top 15 bigrams
    per class.

    Saves: figures/technique2_ngram_analysis.png
    """
    TOP_N = 15

    def _tokenise(text: str) -> list[str]:
        text = text.lower()
        text = text.translate(str.maketrans("", "", string.punctuation))
        return [w for w in text.split() if w not in _STOP_WORDS and len(w) > 1]

    def _bigrams(tokens: list[str]) -> list[tuple[str, str]]:
        return [(tokens[i], tokens[i + 1]) for i in range(len(tokens) - 1)]

    # --- 1. Count bigrams per class ---
    counts: dict[int, collections.Counter] = {0: collections.Counter(), 1: collections.Counter()}
    for r in train_records:
        tokens = _tokenise(r["text"])
        counts[r["label"]].update(_bigrams(tokens))

    top: dict[int, list[tuple[tuple, int]]] = {
        lbl: counts[lbl].most_common(TOP_N) for lbl in (0, 1)
    }

    # --- 2. Print summary ---
    print("\n=== Technique 2: N-gram Analysis (Top Bigrams by Class) ===")
    for lbl, label_name in [(1, "PCL=1"), (0, "No PCL=0")]:
        print(f"  Top 10 bigrams for {label_name}:")
        for (w1, w2), cnt in top[lbl][:10]:
            print(f"    {w1} {w2}  ({cnt})")

    # --- 3. Build figure ---
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    fig.suptitle(
        "Top Bigrams by Class (stop words removed, Training Split)",
        fontsize=13, fontweight="bold",
    )

    palette = {0: "steelblue", 1: "tomato"}
    titles  = {0: f"No PCL (label=0) — Top {TOP_N} bigrams",
               1: f"PCL (label=1) — Top {TOP_N} bigrams"}

    for ax, lbl in zip(axes, [1, 0]):   # PCL on left, No PCL on right
        items = top[lbl]
        labels = [f"{w1} {w2}" for (w1, w2), _ in items]
        values = [cnt for _, cnt in items]

        # Plot ascending so highest bar is at top
        ax.barh(range(TOP_N), values[::-1], color=palette[lbl],
                edgecolor="black", linewidth=0.6, height=0.7)
        ax.set_yticks(range(TOP_N))
        ax.set_yticklabels(labels[::-1], fontsize=9)
        ax.set_xlabel("Frequency", fontsize=10)
        ax.set_title(titles[lbl], fontsize=11)
        ax.xaxis.set_major_formatter(ticker.FuncFormatter(lambda x, _: f"{int(x):,}"))

    plt.tight_layout()
    out_path = FIGURES_DIR / "technique2_ngram_analysis.png"
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {out_path}")


if __name__ == "__main__":
    train_records = load_train()
    print(f"Loaded {len(train_records)} training examples")

    technique_1(train_records)
    technique_2(train_records)

    print(f"\nFigures saved to {FIGURES_DIR}")
