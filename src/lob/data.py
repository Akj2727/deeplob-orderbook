"""FI-2010 limit order book data: loading, the standard split and sliding windows.

FI-2010 (Ntakaris et al., 2018) has 10 days of Nasdaq Nordic order book events for
5 stocks, with 10 price levels on each side. We use the decimal-precision normalised,
no-auction version. In the raw files each column is one event and each row a field:

  * rows 0..39: level by level, ask price, ask volume, bid price, bid volume
  * last 5 rows: mid-price direction labels for horizons k = 10, 20, 30, 50, 100 events
    (1 = up, 2 = stationary, 3 = down)

Standard setup (as in DeepLOB): days 1-7 for training (the last 20% kept for
validation, in time order), days 8-10 for testing.
"""
from __future__ import annotations

import io
import urllib.request
import zipfile
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import Dataset

HORIZONS = (10, 20, 30, 50, 100)
N_LOB = 40
WINDOW = 100
DATA_URL = ("https://raw.githubusercontent.com/zcakhaa/DeepLOB-Deep-Convolutional-"
            "Neural-Networks-for-Limit-Order-Books/master/data/data.zip")
TRAIN_FILE = "Train_Dst_NoAuction_DecPre_CF_7.txt"
TEST_FILES = tuple(f"Test_Dst_NoAuction_DecPre_CF_{d}.txt" for d in (7, 8, 9))


@dataclass
class Split:
    lob: np.ndarray     # [events, 40] float32
    labels: np.ndarray  # [events, 5] int64 in {0: up, 1: stationary, 2: down}


def download(dest: Path) -> None:
    dest.mkdir(parents=True, exist_ok=True)
    if (dest / TRAIN_FILE).exists():
        return
    raw = urllib.request.urlopen(DATA_URL, timeout=300).read()
    zipfile.ZipFile(io.BytesIO(raw)).extractall(dest)


def _read(path: Path) -> Split:
    cache = path.with_suffix(".npz")
    if cache.exists():
        z = np.load(cache)
        return Split(z["lob"], z["labels"])
    raw = np.loadtxt(path, dtype=np.float64)
    lob = raw[:N_LOB].T.astype(np.float32)
    labels = (raw[-5:].T - 1).astype(np.int64)
    np.savez(cache, lob=lob, labels=labels)
    return Split(lob, labels)


def load(data_dir: Path, val_frac: float = 0.2) -> tuple[Split, Split, Split]:
    """(train, validation, test). Validation is the last part of days 1-7, in time order."""
    full = _read(data_dir / TRAIN_FILE)
    cut = int(len(full.lob) * (1 - val_frac))
    train = Split(full.lob[:cut], full.labels[:cut])
    val = Split(full.lob[cut:], full.labels[cut:])
    tests = [_read(data_dir / f) for f in TEST_FILES]
    test = Split(np.concatenate([t.lob for t in tests]), np.concatenate([t.labels for t in tests]))
    return train, val, test


class WindowDataset(Dataset):
    """Item i is the last `window` events ending at event i+window-1, with that event's label."""

    def __init__(self, split: Split, horizon_idx: int, window: int = WINDOW):
        self.x = torch.from_numpy(split.lob)
        self.y = torch.from_numpy(split.labels[:, horizon_idx])
        self.window = window

    def __len__(self) -> int:
        return len(self.x) - self.window + 1

    def __getitem__(self, i: int):
        return self.x[i: i + self.window].unsqueeze(0), self.y[i + self.window - 1]
