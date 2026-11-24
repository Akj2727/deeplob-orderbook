# deeplob-orderbook

Predicting the direction of the next mid-price move from limit order book data, comparing deep models (DeepLOB and a small transformer) against feature-based baselines. The data is FI-2010, the standard public benchmark of Nasdaq Nordic order book events.

## Results

Test set: days 8 to 10 of FI-2010, 139,587 events. Macro F1 over up / stationary / down, so a model that always predicts "stationary" scores about 0.28 to 0.33.

| Horizon (events) | Transformer | XGBoost | Logistic | DeepLOB | XGBoost, book only |
| --- | --- | --- | --- | --- | --- |
| k = 10 | **0.809** | 0.771 | 0.785 | 0.694 | 0.358 |
| k = 50 | **0.877** | 0.820 | 0.697 | 0.627 | 0.384 |
| k = 100 | **0.920** | 0.863 | 0.514 | 0.719 | 0.400 |

Single-window inference latency on one CPU thread (p50): transformer about 0.75 ms, DeepLOB 2.7 to 4.1 ms. Full tables are in [`results/summary.md`](results/summary.md).

What the numbers say:
- **The transformer is the best model at every horizon.** It beats XGBoost by 3.8 to 5.7 F1 points, with 103k parameters and sub-millisecond CPU inference.
- **Almost all the signal comes from the recent price path, not the book.** XGBoost on the current book alone (spread, imbalance at 1, 3, 5 and 10 levels, depth, book slope) scores 0.36 to 0.40, close to guessing. Adding past returns and volatility lifts it to 0.77 to 0.86. FI-2010 labels compare a smoothed future mid-price with a smoothed past one, so part of each label is already visible in recent prices. This is a known property of the dataset and explains why long horizons look easier here than they would be in live trading.
- **DeepLOB underperformed in this budget.** Each model got at most 10 epochs of 32,768 sampled windows on a 2-core CPU, which is far less training than the DeepLOB paper used (it reports F1 around 0.83, 0.80 and 0.77). Its validation F1 also swung from epoch to epoch. The transformer converged much faster under the same budget.

## Pipeline

1. **Data** (`data.py`). FI-2010, no-auction, decimal-precision normalised version, downloaded automatically. Days 1 to 7 are used for training, with the last 20% held out in time order for early stopping. Days 8 to 10 are the test set.
2. **Inputs** (`train.py`, `Windows`). Each sample is the last 100 events × 40 columns (10 levels of ask price, ask volume, bid price, bid volume). FI-2010 stacks five stocks with different price levels, so prices are converted to basis points relative to the window's last mid-price, and volumes are log-transformed and z-scored with training-set statistics only. Batches are built with one tensor gather instead of per-sample Python.
3. **Models** (`models.py`):
   - **DeepLOB** (Zhang, Zohren and Roberts, 2019): convolutions that merge price with volume, bid with ask and then all levels, an inception module over time, and an LSTM.
   - **Transformer:** a linear embedding, sinusoidal positions and 2 pre-norm encoder layers, predicting from the last event.
4. **Training.** Adam, cross-entropy weighted against the class imbalance (which changes a lot by horizon), a fresh random sample of windows each epoch, and keeping the weights with the best validation macro F1.
5. **Baselines** (`features.py`). Logistic regression and XGBoost on point-in-time features: spread, microprice gap, order imbalance at several depths, depth ratio, book slope, past returns over 1 to 50 events, rolling volatility and smoothed imbalance. The "book only" ablation drops every past-price feature.

## Run

```
pip install -e ".[test]"
pytest
python -m lob.run                                   # all models, k = 10, 50, 100 (about 3 hours on 2 CPU cores)
python -m lob.run --models transformer --horizons 50
python -m lob.run --summary-only                    # rebuild results/summary.md
```

Each model and horizon is saved to `results/runs/` as soon as it finishes, so an interrupted run resumes where it stopped.

## Tests

- windows end exactly at the labelled event
- features never use future events
- prices in a window are relative to its last mid-price
- both models output three classes
- training recovers a planted pattern

## Limitations

- **FI-2010 is small and old:** 10 days in June 2010, 5 stocks. Its smoothed labels make the task easier than predicting raw next moves.
- **No trading test.** Prices are normalised and the labels use a fixed threshold, so the results measure classification quality, not profit after costs.
- **One seed, short training.** Repeating runs over seeds and training DeepLOB longer would tighten the comparison.
- **Stock boundaries.** The five stocks are concatenated in the files, so a few windows span two stocks.
