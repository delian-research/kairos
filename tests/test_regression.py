from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

import kairos as k


@pytest.fixture
def factors(dates, rng):
    x = rng.normal(0, 0.01, (len(dates), 2))
    return pd.DataFrame(x, index=dates, columns=["mkt", "smb"])


def test_recovers_known_loadings(factors, rng):
    e = rng.normal(0, 0.002, len(factors))
    y = 0.0002 + 1.2 * factors["mkt"] - 0.5 * factors["smb"] + e
    reg = k.factor_regression(y, factors)
    assert list(reg.index) == ["alpha", "mkt", "smb"]
    assert reg.loc["mkt", "coef"] == pytest.approx(1.2, abs=0.03)
    assert reg.loc["smb", "coef"] == pytest.approx(-0.5, abs=0.03)
    assert reg.loc["mkt", "p"] < 1e-6
    meta = reg.attrs["kairos"]
    assert meta["n"] == len(factors) and 0.9 < meta["r2"] < 1
    assert meta["resid_vol"] == pytest.approx(0.002, rel=0.1)


def test_single_factor_matches_beta(factors, rng):
    y = 0.9 * factors["mkt"] + pd.Series(
        rng.normal(0, 0.005, len(factors)), index=factors.index
    )
    for hac in (False, True):
        reg = k.factor_regression(y, factors["mkt"], hac=hac)
        b = k.beta(y, factors["mkt"], hac=hac)
        assert reg.loc["mkt", "coef"] == pytest.approx(b.value)
        assert reg.loc["mkt", "se"] == pytest.approx(b.params["se"])
        assert reg.loc["alpha", "coef"] == pytest.approx(b.params["alpha"])


def test_collinear_factors_raise(factors):
    f = factors.assign(dup=factors["mkt"] * 2)
    with pytest.raises(ValueError, match="collinear"):
        k.factor_regression(factors["mkt"] + 0.0, f)


def test_dimson_beta_recovers_delayed_reaction(dates, rng):
    m = pd.Series(rng.normal(0, 0.01, len(dates)), index=dates)
    stale = 0.6 * m + 0.4 * m.shift(1) + rng.normal(0, 0.002, len(dates))
    plain = k.beta(stale.dropna(), m).value
    dim = k.dimson_beta(stale, m, lags=1)
    assert plain == pytest.approx(0.6, abs=0.05)
    assert dim.value == pytest.approx(1.0, abs=0.05)
    assert len(dim.params["betas"]) == 2


def test_downside_beta(dates, rng):
    m = pd.Series(rng.normal(0, 0.01, len(dates)), index=dates)
    y = np.where(m < 0, 1.5 * m, 0.5 * m) + rng.normal(0, 0.001, len(dates))
    y = pd.Series(y, index=dates)
    assert k.downside_beta(y, m).value == pytest.approx(1.5, abs=0.05)
    assert k.beta(y, m).value < 1.3


def test_adjust_beta(factors, rng):
    y = 1.6 * factors["mkt"] + pd.Series(
        rng.normal(0, 0.01, len(factors)), index=factors.index
    )
    b = k.beta(y, factors["mkt"])
    blume = k.adjust_beta(b)
    assert blume.value == pytest.approx(2 / 3 * b.value + 1 / 3)
    assert blume.params["raw_beta"] == b.value
    vas = k.adjust_beta(b, "vasicek", prior_sd=0.3)
    weight = b.params["se"] ** 2 / (b.params["se"] ** 2 + 0.09)
    assert vas.value == pytest.approx((1 - weight) * b.value + weight)
    assert 1.0 < vas.value < b.value
    with pytest.raises(ValueError, match="prior_sd"):
        k.adjust_beta(b, "vasicek")
