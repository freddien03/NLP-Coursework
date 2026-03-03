# NLP-Coursework

Binary classification of Patronising and Condescending Language (PCL) using the DontPatronizeMe dataset (SemEval Task 4).

**GitHub:** [github.com/freddienunn/NLP-Coursework](https://github.com/freddienunn/NLP-Coursework)

---

## Repository Structure

```text
NLP-Coursework/
├── data/                    # Dataset files (gitignored — add manually)
│   ├── dontpatronizeme_pcl.tsv         # Full labelled corpus (train + dev)
│   ├── train_semeval_parids-labels.csv # Train split par_ids + fine-grained labels
│   ├── dev_semeval_parids-labels.csv   # Dev split par_ids + fine-grained labels
│   └── task4_test.tsv                  # Unlabelled test set (3,832 examples)
├── src/
│   ├── data_utils.py        # Data loading and train/dev/test splits
│   ├── eda.py               # Exploratory data analysis (Exercise 2)
│   ├── train.py             # Training entry point (delegates to BestModel/model.py)
│   ├── evaluate.py          # Error analysis: best model vs baseline (Exercise 5.2)
│   ├── predict.py           # Generate dev.txt / test.txt from saved checkpoint
│   ├── baseline.py          # Baseline: RoBERTa, 1 epoch, cross-entropy, 2:1 downsampled
│   └── ablation_no_category.py  # Ablation: focal loss only, no category head
├── BestModel/
│   └── model.py             # Best model: multi-task RoBERTa, focal loss + category head
├── predictions/
│   └── baseline_dev.txt     # Baseline dev predictions (for evaluate.py)
├── Reconstruct_and_RoBERTa_baseline_train_dev_dataset.ipynb  # Original baseline notebook
├── dev.txt                  # Best model dev predictions (0/1 per line)
└── test.txt                 # Best model test predictions (0/1 per line)
```

## Quickstart

```bash
# Install dependencies
pip install -r requirements.txt

# 1. Place dataset files in data/

# 2. Run EDA
python src/eda.py

# 3. Train best model (saves checkpoint to checkpoints/roberta_best/)
python src/train.py
# Or directly:
python BestModel/model.py

# 4. Generate dev.txt and test.txt from saved checkpoint
python src/predict.py

# 5. Run baseline (saves predictions/baseline_dev.txt)
python src/baseline.py

# 6. Error analysis: best model vs baseline
python src/evaluate.py
```

## Model Summary

| Model | Dev F1 | Notes |
|---|---|---|
| Best model (`BestModel/model.py`) | 0.6119 | Multi-task RoBERTa-base, focal loss (γ=2, α=[1,9.55]), auxiliary category head (BCE, λ=0.5), 5 epochs, threshold-tuned |
| Baseline (`BestModel/baseline.py`) | 0.5073 | RoBERTa-base, cross-entropy, 2:1 downsampled negatives, 1 epoch, threshold=0.5 |
| Ablation: no category head (`BestModel/ablation_no_category.py`) | — | Focal loss only, no auxiliary head |
