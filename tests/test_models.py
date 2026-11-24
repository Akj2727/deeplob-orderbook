import numpy as np
import torch

from lob.data import Split
from lob.models import build
from lob.train import Windows, fit, metrics


def test_models_output_three_logits():
    x = torch.randn(4, 1, 100, 40)
    for name in ("deeplob", "transformer"):
        assert build(name)(x).shape == (4, 3)


def test_metrics_on_perfect_predictions():
    y = np.array([0, 1, 2, 1])
    m = metrics(y, y)
    assert m["accuracy"] == 1.0 and m["f1_macro"] == 1.0


def test_fit_learns_a_planted_pattern():
    rng = np.random.default_rng(0)
    n = 1200
    lob = rng.uniform(1.0, 2.0, size=(n, 40)).astype(np.float32)  # positive prices and volumes
    lob[:, 1] = np.exp(rng.normal(size=n)).astype(np.float32)     # best ask volume drives the label
    z = np.log(lob[:, 1])
    labels = np.where(z > 0.5, 0, np.where(z < -0.5, 2, 1))
    s = Split(lob, np.stack([labels] * 5, axis=1).astype(np.int64))
    model = build("transformer")
    info = fit(model, s, s, horizon_idx=0, epochs=12, samples_per_epoch=2048, batch_size=64,
               lr=3e-3, val_stride=4, patience=12, log=lambda m: None)
    assert info["best_val_f1"] > 0.6


def test_window_prices_are_relative_to_the_last_mid():
    from lob.data import Split
    lob = np.full((150, 40), 2.0, dtype=np.float32)
    lob[:, 0], lob[:, 2] = 10.1, 9.9   # best ask and bid around a mid of 10
    s = Split(lob, np.zeros((150, 5), dtype=np.int64))
    x, _ = Windows(s, 0).batch(torch.tensor([0, 20]))
    assert torch.allclose(x[:, 0, -1, 0], torch.full((2,), 10.0), atol=0.01)  # +100 bp, in tens of bp
    assert torch.allclose(x[:, 0, -1, 2], torch.full((2,), -10.0), atol=0.01)
