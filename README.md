# NLP-Coursework

Binary classification of Patronising and Condescending Language (PCL) using the DontPatronizeMe dataset.

**Leaderboard name:** <!-- add ≤20 char name here -->

**GitHub:** [github.com/freddienunn/NLP-Coursework](https://github.com/freddienunn/NLP-Coursework) <!-- update URL -->

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
│   ├── train.py             # Model training (Exercise 4)
│   ├── evaluate.py          # Error analysis and ablations (Exercise 5.2)
│   └── predict.py           # Generate dev.txt / test.txt (Exercise 5.1)
├── figures/                 # Saved EDA and analysis plots (gitignored)
├── BestModel/
│   ├── model.py             # Self-contained best model script
│   └── README.md            # Approach description
├── dev.txt                  # Dev set predictions (0/1 per line)
└── test.txt                 # Test set predictions (0/1 per line)
```

## Quickstart

```bash
# 1. Place dataset files in data/
# 2. Run EDA
python src/eda.py

# 3. Train best model
python src/train.py

# 4. Generate predictions
python src/predict.py
```
