import numpy as np
import torch

from lob.data import Split, WindowDataset


def split(n=150):
    lob = np.arange(n * 40, dtype=np.float32).reshape(n, 40)
    labels = np.stack([np.arange(n) % 3] * 5, axis=1).astype(np.int64)
    return Split(lob, labels)


def test_window_ends_at_the_labelled_event():
    s = split()
    ds = WindowDataset(s, horizon_idx=0, window=100)
    assert len(ds) == 51
    x, y = ds[7]
    assert x.shape == (1, 100, 40)
    assert torch.equal(x[0, -1], torch.from_numpy(s.lob[106]))
    assert y.item() == s.labels[106, 0]
