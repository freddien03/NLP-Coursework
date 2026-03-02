# Best Model

**Approach:** DeBERTa-v3-base fine-tuned with class-weighted cross-entropy (pos\_weight ≈ 9.55), trained for 4 epochs on the full training set (8,375 examples), with decision threshold optimised on the dev set to maximise positive-class F1.

**Dev F1:** _fill in after training_

**Test F1:** _fill in after training_

## Key design choices

| Component | Baseline | This model |
|---|---|---|
| Backbone | RoBERTa-base | DeBERTa-v3-base |
| Class handling | Downsample negatives 2:1 | Weighted CE loss (w ≈ 9.55) |
| Training epochs | 1 | 4 |
| LR schedule | None | Linear warmup + linear decay |
| Decision threshold | Fixed 0.5 | Grid-searched on dev |
| Training examples used | 2,382 | 8,375 |

## Reproduce results

```bash
pip install -r requirements.txt
python BestModel/model.py
```

This trains the model end-to-end and writes `dev.txt` and `test.txt` to the repository root.

Alternatively, to train and predict separately:

```bash
python src/train.py    # trains and saves checkpoint to checkpoints/deberta_best/
python src/predict.py  # tunes threshold and writes dev.txt + test.txt
```
