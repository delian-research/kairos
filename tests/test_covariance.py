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


@pytest.fixture
def linked(dates, rng):
    """Two assets that share a market factor and nothing else."""
    m = pd.Series(rng.normal(0, 0.01, len(dates)), index=dates, name="mkt")
    a = 1.5 * m + rng.normal(0, 0.01, len(dates))
    b = 1.2 * m + rng.normal(0, 0.01, len(dates))
    return a.rename("a"), b.rename("b"), m


def test_pair_correlation_matches_pandas_with_interval(linked):
    a, b, _ = linked
    est = k.pair_correlation(a, b)
    assert est.value == pytest.approx(a.corr(b))
    assert est.params["lower"] < est.value < est.params["upper"]
    z = np.arctanh(est.value)
    half = 1.959964 / np.sqrt(len(a) - 3)
    assert est.params["upper"] == pytest.approx(np.tanh(z + half), rel=1e-5)
    wide = k.pair_correlation(a.iloc[:40], b.iloc[:40])
    assert (wide.params["upper"] - wide.params["lower"]) > (
        est.params["upper"] - est.params["lower"]
    )


def test_partial_correlation_removes_shared_factor(linked):
    a, b, m = linked
    raw = k.pair_correlation(a, b)
    partial = k.pair_correlation(a, b, controls=m)
    assert raw.value > 0.4
    assert abs(partial.value) < 0.1
    assert partial.params["lower"] < 0 < partial.params["upper"]
    assert partial.method == "partial-correlation"
    assert partial.params["controls"] == ["mkt"]
    assert partial.params["raw_correlation"] == pytest.approx(raw.value)


def test_pair_correlation_rejects_prices_and_constants(dates):
    prices = pd.Series(np.linspace(100, 120, 50), index=dates[:50])
    with pytest.raises(ValueError, match="looks like prices"):
        k.pair_correlation(prices, prices * 2)
    flat = pd.Series(0.0, index=dates[:50])
    with pytest.raises(ValueError, match="constant"):
        k.pair_correlation(flat, k.returns(prices).reindex(dates[:50]).fillna(0.01))


def test_rolling_correlation(linked):
    a, b, _ = linked
    rc = k.rolling_correlation(a, b, 60)
    assert rc.isna().sum() == 59
    assert rc.iloc[-1] == pytest.approx(a.iloc[-60:].corr(b.iloc[-60:]))
    with pytest.raises(ValueError):
        k.rolling_correlation(a.iloc[:10], b.iloc[:10], 60)
