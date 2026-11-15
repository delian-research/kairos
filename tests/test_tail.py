from __future__ import annotations

from statistics import NormalDist

import numpy as np
import pandas as pd
import pytest

import kairos as k


@pytest.fixture
def normal_r(rng):
    return pd.Series(rng.normal(0.0, 0.01, 200_000))


def test_historical_var_matches_quantile(rng):
    r = pd.Series(rng.normal(0, 0.01, 1000))
    assert k.value_at_risk(r, 0.99).value == pytest.approx(-np.quantile(r, 0.01))


def test_normal_var_and_es_on_normal_data(normal_r):
    z = NormalDist().inv_cdf(0.05)
    v = k.value_at_risk(normal_r, method="normal").value
    assert v == pytest.approx(-z * 0.01, rel=0.01)
    hist = k.value_at_risk(normal_r).value
    assert hist == pytest.approx(v, rel=0.02)
    cf = k.value_at_risk(normal_r, method="cornish-fisher").value
    assert cf == pytest.approx(v, rel=0.02)
    es_n = k.expected_shortfall(normal_r, method="normal").value
    es_h = k.expected_shortfall(normal_r).value
    assert es_n == pytest.approx(0.01 * NormalDist().pdf(z) / 0.05, rel=0.01)
    assert es_h == pytest.approx(es_n, rel=0.02)
    assert es_h > hist


def test_cornish_fisher_widens_for_fat_left_tail(rng):
    r = pd.Series(rng.standard_t(4, 20_000) * 0.01)
    r = r - 0.03 * (rng.random(20_000) < 0.01)  # rare crashes: negative skew
    assert (
        k.value_at_risk(r, 0.99, "cornish-fisher").value
        > k.value_at_risk(r, 0.99, "normal").value
    )


def test_level_validated(normal_r):
    with pytest.raises(ValueError, match="level"):
        k.value_at_risk(normal_r, 95)


def test_moments_match_pandas(rng):
    r = pd.Series(rng.normal(0, 1, 300))
    assert k.skewness(r).value == pytest.approx(r.skew())
    assert k.excess_kurtosis(r).value == pytest.approx(r.kurt())


def test_worst_periods():
    r = pd.Series([0.01, -0.05, 0.02, -0.03], index=list("abcd"))
    w = k.worst_periods(r, 2)
    assert list(w.index) == ["b", "d"]
