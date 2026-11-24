"""Hand-built order book features for the baselines.

Every feature at event t uses events up to and including t only.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

LEVELS = 10


def book_arrays(lob: np.ndarray):
    """Split a [events, 40] block into ask price, ask volume, bid price, bid volume [events, 10]."""
    blk = lob.reshape(len(lob), LEVELS, 4)
    return blk[:, :, 0], blk[:, :, 1], blk[:, :, 2], blk[:, :, 3]


def make_features(lob: np.ndarray) -> pd.DataFrame:
    ap, av, bp, bv = book_arrays(lob.astype(np.float64))
    mid = (ap[:, 0] + bp[:, 0]) / 2
    f = {
        "spread": (ap[:, 0] - bp[:, 0]) / mid,
        "microprice_gap": ((ap[:, 0] * bv[:, 0] + bp[:, 0] * av[:, 0]) / (av[:, 0] + bv[:, 0]) - mid) / mid,
    }
    for k in (1, 3, 5, 10):
        b, a = bv[:, :k].sum(1), av[:, :k].sum(1)
        f[f"imbalance_{k}"] = (b - a) / (b + a)
    f["depth_ratio"] = np.log(bv.sum(1) / av.sum(1))
    f["ask_slope"] = (ap[:, -1] - ap[:, 0]) / mid
    f["bid_slope"] = (bp[:, 0] - bp[:, -1]) / mid
    df = pd.DataFrame(f)
    logmid = pd.Series(np.log(mid))
    for k in (1, 5, 10, 50):
        df[f"ret_{k}"] = logmid.diff(k)
    df["vol_50"] = logmid.diff().rolling(50).std()
    for k in (5, 20):
        df[f"imbalance_1_ma{k}"] = df["imbalance_1"].rolling(k).mean()
    return df.replace([np.inf, -np.inf], np.nan).fillna(0.0)
