import numpy as np

from lob.features import make_features


def book(n=300, seed=0):
    rng = np.random.default_rng(seed)
    mid = 10 + np.cumsum(rng.normal(0, 0.01, n))
    lob = np.zeros((n, 40))
    for lvl in range(10):
        lob[:, 4 * lvl] = mid + 0.01 * (lvl + 1)        # ask price
        lob[:, 4 * lvl + 1] = rng.uniform(1, 5, n)      # ask volume
        lob[:, 4 * lvl + 2] = mid - 0.01 * (lvl + 1)    # bid price
        lob[:, 4 * lvl + 3] = rng.uniform(1, 5, n)      # bid volume
    return lob


def test_features_do_not_look_ahead():
    a = book()
    b = a.copy()
    b[200:] *= 1.5  # change the future only
    fa, fb = make_features(a), make_features(b)
    assert np.allclose(fa.iloc[:200].values, fb.iloc[:200].values)


def test_imbalance_sign():
    lob = book(60)
    lob[:, 3] = 100.0  # large best-bid volume
    f = make_features(lob)
    assert (f["imbalance_1"] > 0.9).all()
