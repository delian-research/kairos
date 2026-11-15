from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

import kairos as k


def test_beta_recovers_known_slope(dates, rng) -> None:
    x = pd.Series(rng.normal(0, 0.01, len(dates)), index=dates)
    y = 0.0002 + 1.5 * x + rng.normal(0, 0.005, len(dates))
    est = k.beta(y, x)
    assert est.value == pytest.approx(1.5, abs=0.1)
    assert est.params["alpha"] == pytest.approx(0.0002, abs=0.0005)
    assert 0 < est.params["r2"] < 1
    assert est.params["se_type"] == "ols"
    assert est.params["t"] == pytest.approx(est.value / est.params["se"])


def test_beta_aligns_and_drops_nans(dates, rng) -> None:
    x = pd.Series(rng.normal(0, 0.01, len(dates)), index=dates)
    y = 2.0 * x
    y.iloc[:10] = np.nan
    est = k.beta(y.iloc[5:], x)
    assert est.n == len(dates) - 10
    assert est.value == pytest.approx(2.0)


def test_zero_variance_benchmark_raises() -> None:
    with pytest.raises(ValueError, match="zero variance"):
        k.beta(pd.Series([0.1, 0.2, 0.3]), pd.Series([0.0, 0.0, 0.0]))


def test_rolling_beta_matches_naive_loop(dates, rng) -> None:
    x = pd.Series(rng.normal(0, 0.01, 200), index=dates[:200])
    y = 0.8 * x + rng.normal(0, 0.01, 200)
    w = 60
    fast = k.rolling_beta(y, x, w)
    for i in (w - 1, 120, 199):
        xs, ys = x.iloc[i - w + 1 : i + 1], y.iloc[i - w + 1 : i + 1]
        naive = np.cov(ys, xs, ddof=1)[0, 1] / np.var(xs, ddof=1)
        assert fast.iloc[i] == pytest.approx(naive)
    assert fast.iloc[: w - 1].isna().all()


def test_hac_se_larger_with_autocorrelated_errors(rng) -> None:
    n = 2000
    x = np.zeros(n)
    e = np.zeros(n)
    for t in range(1, n):  # both regressor and error follow AR(1)
        x[t] = 0.8 * x[t - 1] + rng.normal()
        e[t] = 0.8 * e[t - 1] + rng.normal()
    xs, ys = pd.Series(x), pd.Series(1.0 * x + e)
    ols = k.beta(ys, xs)
    hac = k.beta(ys, xs, hac=True)
    assert hac.value == ols.value
    assert hac.params["se_type"] == "newey-west"
    assert hac.params["lags"] == k.newey_west_lags(n)
    assert hac.params["se"] > 1.5 * ols.params["se"]


def test_newey_west_lags() -> None:
    assert k.newey_west_lags(100) == 4
    assert k.newey_west_lags(252) == 4
    assert k.newey_west_lags(1) == 1
