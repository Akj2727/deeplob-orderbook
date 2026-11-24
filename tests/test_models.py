import numpy as np
import torch

from lob.data import Split
from lob.models import build
from lob.train import fit, metrics


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
    lob = rng.normal(size=(n, 40)).astype(np.float32)
    labels = np.where(lob[:, 0] > 0.5, 0, np.where(lob[:, 0] < -0.5, 2, 1))
    s = Split(lob, np.stack([labels] * 5, axis=1).astype(np.int64))
    model = build("transformer")
    info = fit(model, s, s, horizon_idx=0, epochs=4, samples_per_epoch=1024, batch_size=64,
               lr=3e-3, val_stride=4, log=lambda m: None)
    assert info["best_val_f1"] > 0.6
