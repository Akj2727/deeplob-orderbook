"""Training and evaluation for the deep models, plus the feature baselines."""
from __future__ import annotations

import time

import numpy as np
import torch
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score, precision_score, recall_score
from sklearn.preprocessing import StandardScaler
from torch import nn
from .data import WINDOW, Split
from .features import make_features


def metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict:
    return {
        "accuracy": accuracy_score(y_true, y_pred),
        "f1_macro": f1_score(y_true, y_pred, average="macro"),
        "precision_macro": precision_score(y_true, y_pred, average="macro", zero_division=0),
        "recall_macro": recall_score(y_true, y_pred, average="macro", zero_division=0),
        "confusion": confusion_matrix(y_true, y_pred, labels=[0, 1, 2]).tolist(),
    }


class Windows:
    """Builds batches of [B, 1, window, 40] windows with one tensor gather, no per-item Python."""

    def __init__(self, split: Split, horizon_idx: int, window: int = WINDOW):
        self.x = torch.from_numpy(split.lob)
        self.y = torch.from_numpy(split.labels[:, horizon_idx])
        self.window = window
        self.offsets = torch.arange(window)

    def __len__(self) -> int:
        return len(self.x) - self.window + 1

    def batch(self, starts: torch.Tensor):
        idx = starts[:, None] + self.offsets
        return self.x[idx].unsqueeze(1), self.y[starts + self.window - 1]


@torch.no_grad()
def predict(model: nn.Module, w: Windows, stride: int = 1,
            batch_size: int = 1024) -> tuple[np.ndarray, np.ndarray]:
    model.eval()
    starts = torch.arange(0, len(w), stride)
    preds, ys = [], []
    for chunk in starts.split(batch_size):
        x, y = w.batch(chunk)
        preds.append(model(x).argmax(1).numpy())
        ys.append(y.numpy())
    return np.concatenate(ys), np.concatenate(preds)


def fit(model: nn.Module, train: Split, val: Split, horizon_idx: int, epochs: int = 10,
        samples_per_epoch: int = 32768, batch_size: int = 256, lr: float = 1e-3,
        patience: int = 3, val_stride: int = 4, seed: int = 0, log=print) -> dict:
    """Adam with class-weighted cross-entropy; keeps the weights with the best validation macro F1.

    Each epoch trains on a fresh random sample of training windows.
    """
    gen = torch.Generator().manual_seed(seed)
    torch.manual_seed(seed)
    tr, va = Windows(train, horizon_idx), Windows(val, horizon_idx)
    # class weights offset the label imbalance, which differs by horizon
    counts = np.bincount(train.labels[WINDOW - 1:, horizon_idx], minlength=3)
    weights = torch.tensor(counts.sum() / (3 * counts), dtype=torch.float32)
    loss_fn = nn.CrossEntropyLoss(weight=weights)
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    n = min(samples_per_epoch, len(tr))
    best, best_state, bad, history = -1.0, None, 0, []
    for epoch in range(1, epochs + 1):
        model.train()
        t0, total = time.time(), 0.0
        starts = torch.randperm(len(tr), generator=gen)[:n]
        for chunk in starts.split(batch_size):
            x, y = tr.batch(chunk)
            opt.zero_grad()
            loss = loss_fn(model(x), y)
            loss.backward()
            opt.step()
            total += loss.item() * len(y)
        yv, pv = predict(model, va, stride=val_stride)
        f1 = f1_score(yv, pv, average="macro")
        history.append({"epoch": epoch, "train_loss": total / n, "val_f1": f1,
                        "seconds": time.time() - t0})
        log(f"  epoch {epoch}: loss {total / n:.4f}, val F1 {f1:.4f}, {time.time() - t0:.0f}s")
        if f1 > best:
            best, bad = f1, 0
            best_state = {k: v.clone() for k, v in model.state_dict().items()}
        else:
            bad += 1
            if bad >= patience:
                break
    model.load_state_dict(best_state)
    return {"best_val_f1": best, "history": history}


@torch.no_grad()
def latency_us(model: nn.Module, n: int = 300) -> dict:
    """Single-window inference latency on CPU, one thread, in microseconds."""
    model.eval()
    threads = torch.get_num_threads()
    torch.set_num_threads(1)
    x = torch.randn(1, 1, WINDOW, 40)
    for _ in range(30):
        model(x)
    times = []
    for _ in range(n):
        t = time.perf_counter()
        model(x)
        times.append((time.perf_counter() - t) * 1e6)
    torch.set_num_threads(threads)
    return {"p50_us": float(np.percentile(times, 50)), "p99_us": float(np.percentile(times, 99))}


def _features(split: Split) -> np.ndarray:
    return make_features(split.lob).to_numpy()


def baselines(train: Split, test: Split, horizon_idx: int, seed: int = 0) -> dict:
    """Logistic regression and XGBoost on hand-built features of the current book."""
    from xgboost import XGBClassifier

    xtr, xte = _features(train), _features(test)
    ytr, yte = train.labels[:, horizon_idx], test.labels[:, horizon_idx]
    # score on the same events the deep models are scored on
    xtr, ytr, xte, yte = xtr[WINDOW - 1:], ytr[WINDOW - 1:], xte[WINDOW - 1:], yte[WINDOW - 1:]
    scaler = StandardScaler().fit(xtr)
    out = {}
    lr = LogisticRegression(max_iter=2000, class_weight="balanced")
    lr.fit(scaler.transform(xtr), ytr)
    out["logistic"] = metrics(yte, lr.predict(scaler.transform(xte)))
    counts = np.bincount(ytr, minlength=3)
    w = (counts.sum() / (3 * counts))[ytr]
    xgb = XGBClassifier(n_estimators=300, max_depth=6, learning_rate=0.05, subsample=0.8,
                        colsample_bytree=0.8, n_jobs=2, random_state=seed, tree_method="hist")
    xgb.fit(xtr, ytr, sample_weight=w)
    out["xgboost"] = metrics(yte, xgb.predict(xte))
    # ablation: the same model on the current book only, with no past-price features
    cols = [i for i, c in enumerate(make_features(train.lob[:200]).columns)
            if not c.startswith(("ret_", "vol_", "imbalance_1_ma"))]
    xgb_book = XGBClassifier(n_estimators=300, max_depth=6, learning_rate=0.05, subsample=0.8,
                             colsample_bytree=0.8, n_jobs=2, random_state=seed, tree_method="hist")
    xgb_book.fit(xtr[:, cols], ytr, sample_weight=w)
    out["xgboost_book_only"] = metrics(yte, xgb_book.predict(xte[:, cols]))
    return out
