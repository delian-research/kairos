from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

import kairos as k


def test_probabilistic_sharpe_grows_with_sample(rng):
    long = pd.Series(rng.normal(0.001, 0.01, 2000))
    short = long.iloc[:60]
    p_long = k.probabilistic_sharpe(long)
    assert p_long.value > 0.99
    assert k.probabilistic_sharpe(short).params["se"] > p_long.params["se"]
    zero = pd.Series(rng.normal(0.0, 0.01, 2000))
    assert 0.02 < k.probabilistic_sharpe(zero).value < 0.98


def test_probabilistic_sharpe_at_own_sharpe_is_half(rng):
    r = pd.Series(rng.normal(0.0005, 0.01, 500))
    sr = k.sharpe_ratio(r).value
    assert k.probabilistic_sharpe(r, sr).value == pytest.approx(0.5)
    ann = k.probabilistic_sharpe(r, sr * np.sqrt(252), annualize=True)
    assert ann.value == pytest.approx(0.5)
    assert ann.params["sharpe"] == pytest.approx(sr * np.sqrt(252))


def test_bootstrap_ci_covers_mean_and_is_reproducible(rng):
    r = pd.Series(rng.normal(0.001, 0.01, 500))
    ci = k.bootstrap_ci(r, lambda s: s.mean(), seed=7, block=1)
    assert ci.value == pytest.approx(r.mean())
    assert ci.params["lower"] < r.mean() < ci.params["upper"]
    assert ci.params["se"] == pytest.approx(0.01 / np.sqrt(500), rel=0.2)
    again = k.bootstrap_ci(r, lambda s: s.mean(), seed=7, block=1)
    assert again.params["lower"] == ci.params["lower"]


def test_bootstrap_accepts_estimate_statistics(rng):
    r = pd.Series(rng.normal(0.001, 0.01, 300))
    ci = k.bootstrap_ci(r, k.sharpe_ratio, n_boot=200, seed=1)
    assert ci.value == pytest.approx(k.sharpe_ratio(r).value)
    assert ci.params["block"] == round(300 ** (1 / 3))


def test_block_bootstrap_widens_interval_for_autocorrelated_data(rng):
    e = rng.normal(0, 0.01, 2000)
    x = np.empty(2000)
    x[0] = e[0]
    for i in range(1, 2000):
        x[i] = 0.8 * x[i - 1] + e[i]
    s = pd.Series(x)
    iid = k.bootstrap_ci(s, lambda v: v.mean(), block=1, seed=3)
    blocked = k.bootstrap_ci(s, lambda v: v.mean(), block=50, seed=3)
    assert blocked.params["se"] > 2 * iid.params["se"]
