from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

import kairos as k


@pytest.fixture
def panel(dates, rng):
    cov = np.array([[1.0, 0.5, 0.2], [0.5, 1.0, 0.3], [0.2, 0.3, 1.0]]) * 1e-4
    x = rng.multivariate_normal(np.zeros(3), cov, len(dates))
    return pd.DataFrame(x, index=dates, columns=["a", "b", "c"])


def test_correlation_min_obs(panel):
    p = panel.copy()
    p.iloc[5:, 2] = np.nan
    c = k.correlation(p, min_obs=10)
    assert np.isnan(c.loc["a", "c"])
    assert c.loc["a", "b"] == pytest.approx(panel["a"].corr(panel["b"]))
    assert c.attrs["kairos"]["pair_n"].loc["a", "c"] == 5


def test_ewma_cov_diagonal_matches_ewma_vol(panel):
    c = k.ewma_cov(panel, halflife=20)
    for col in panel.columns:
        assert np.sqrt(c.loc[col, col]) == pytest.approx(
            k.ewma_vol(panel[col], 20).value
        )
    assert np.allclose(c, c.T)


def test_ledoit_wolf_properties(rng, dates):
    x = pd.DataFrame(rng.normal(0, 0.01, (30, 50)), index=dates[:30])
    lw = k.ledoit_wolf(x)
    s = lw.attrs["kairos"]["shrinkage"]
    assert 0 < s <= 1
    assert np.linalg.eigvalsh(lw.to_numpy()).min() > 0  # p > n, still PD


def test_ledoit_wolf_reference_value():
    # Hand-checkable 2x2 case: identical-variance, uncorrelated in sample.
    x = pd.DataFrame({"a": [1.0, -1.0, 1.0, -1.0], "b": [1.0, 1.0, -1.0, -1.0]})
    lw = k.ledoit_wolf(x)
    assert np.allclose(lw, np.eye(2))


def test_nearest_psd_repairs_and_keeps_diag():
    m = np.array([[1.0, 0.9, -0.9], [0.9, 1.0, 0.9], [-0.9, 0.9, 1.0]])
    assert np.linalg.eigvalsh(m).min() < 0
    fixed = k.nearest_psd(m)
    assert np.linalg.eigvalsh(fixed).min() > -1e-12
    assert np.allclose(np.diag(fixed), 1.0)
    df = pd.DataFrame(m, index=list("xyz"), columns=list("xyz"))
    assert list(k.nearest_psd(df).columns) == list("xyz")
