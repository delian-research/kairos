from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

import kairos as k


@pytest.fixture
def simple(dates, rng):
    x = rng.normal(0.0004, 0.01, (len(dates), 3))
    return pd.DataFrame(x, index=dates, columns=["a", "b", "c"])


@pytest.fixture
def cov(rng):
    a = rng.normal(0, 1, (6, 6))
    m = np.dot(a, a.T) / 6 * 1e-4 + np.eye(6) * 1e-5
    names = list("abcdef")
    return pd.DataFrame(m, index=names, columns=names)


def test_constant_weights_is_weighted_sum_of_simple_returns(simple):
    w = {"a": 0.5, "b": 0.3, "c": 0.2}
    p = k.portfolio_returns(simple, w, "simple")
    assert np.allclose(p, np.dot(simple.to_numpy(), [0.5, 0.3, 0.2]))
    assert p.attrs["kairos"]["rebalance"] == "every"


def test_log_input_round_trips_through_simple(simple):
    w = {"a": 0.5, "b": 0.5}
    p_simple = k.portfolio_returns(simple, w, "simple")
    p_log = k.portfolio_returns(np.log1p(simple), w, "log")
    assert np.allclose(np.expm1(p_log), p_simple)


def test_buy_and_hold_matches_wealth_of_holdings(simple):
    w = {"a": 0.6, "b": 0.4}
    p = k.portfolio_returns(simple, w, "simple", rebalance=None)
    wealth = 0.6 * (1 + simple["a"]).cumprod() + 0.4 * (1 + simple["b"]).cumprod()
    assert np.allclose((1 + p).cumprod(), wealth)


def test_periodic_rebalance_sits_between(simple):
    w = {"a": 0.6, "b": 0.4}
    monthly = k.portfolio_returns(simple, w, "simple", rebalance="ME")
    daily = k.portfolio_returns(simple, w, "simple")
    first_month = simple.index.to_period("M") == simple.index[0].to_period("M")
    hold = k.portfolio_returns(simple[first_month], w, "simple", rebalance=None)
    assert np.allclose(monthly[first_month], hold)
    # The first day of every month starts from target weights.
    starts = ~simple.index.to_period("M").duplicated()
    assert np.allclose(monthly[starts], daily[starts])


def test_cash_and_unknown_assets(simple):
    half = k.portfolio_returns(simple, {"a": 0.5}, "simple")
    assert np.allclose(half, 0.5 * simple["a"])
    assert half.attrs["kairos"]["weight_sum"] == 0.5
    with pytest.raises(ValueError, match="not in the data"):
        k.portfolio_returns(simple, {"zzz": 1.0})


def test_risk_contributions_sum_to_portfolio_vol(cov):
    w = pd.Series([0.3, 0.2, 0.2, 0.1, 0.1, 0.1], index=cov.index)
    rc = k.risk_contributions(w, cov)
    sigma = k.portfolio_vol(w, cov)
    assert sigma == pytest.approx(np.sqrt(w @ cov @ w))
    assert rc["contribution"].sum() == pytest.approx(sigma)
    assert rc["pct"].sum() == pytest.approx(1.0)
    assert rc.attrs["kairos"]["portfolio_vol"] == pytest.approx(sigma)
    ann = k.risk_contributions(w, cov, annualize=True)
    assert ann["contribution"].sum() == pytest.approx(sigma * np.sqrt(252))


def test_weights_can_be_a_subset_of_cov(cov):
    assert k.portfolio_vol({"a": 1.0}, cov) == pytest.approx(np.sqrt(cov.loc["a", "a"]))


def test_diversification_ratio(cov):
    assert k.diversification_ratio({"a": 1.0}, cov) == pytest.approx(1.0)
    w = k.inverse_vol_weights(cov)
    assert k.diversification_ratio(w, cov) > 1.0


def test_inverse_vol_weights(cov):
    w = k.inverse_vol_weights(cov)
    assert w.sum() == pytest.approx(1.0)
    assert np.allclose(w * np.sqrt(np.diag(cov)), (w * np.sqrt(np.diag(cov))).iloc[0])


def _projected_gradient_min_var(s, iters=30_000):
    """Slow reference solver: projected gradient on the simplex."""
    n = len(s)
    w = np.full(n, 1 / n)
    step = 1 / (2 * np.linalg.eigvalsh(s).max())
    for _ in range(iters):
        v = w - step * 2 * s @ w
        u = np.sort(v)[::-1]
        css = np.cumsum(u) - 1
        rho = np.nonzero(u - css / np.arange(1, n + 1) > 0)[0][-1]
        w = np.clip(v - css[rho] / (rho + 1), 0, None)
    return w


def test_min_variance_unconstrained_closed_form(cov):
    w = k.min_variance_weights(cov, long_only=False)
    inv = np.linalg.inv(cov.to_numpy())
    assert np.allclose(w, inv.sum(axis=1) / inv.sum())


@pytest.mark.parametrize("seed", [0, 1, 2, 3])
def test_min_variance_long_only_matches_reference(seed):
    rng = np.random.default_rng(seed)
    a = rng.normal(0, 1, (8, 5))  # few factors: forces some zero weights
    m = np.dot(a, a.T) + np.diag(rng.uniform(0.05, 0.5, 8))
    cov = pd.DataFrame(m)
    w = k.min_variance_weights(cov)
    ref = _projected_gradient_min_var(m)
    assert (w >= 0).all() and w.sum() == pytest.approx(1.0)
    assert float(w @ m @ w) <= float(ref @ m @ ref) * (1 + 1e-9)
    assert np.allclose(w, ref, atol=1e-5)


def test_min_variance_singular_cov_raises():
    m = np.ones((3, 3))
    with pytest.raises(ValueError, match="singular"):
        k.min_variance_weights(pd.DataFrame(m))


def test_risk_parity_equalizes_contributions(cov):
    w = k.risk_parity_weights(cov)
    assert (w > 0).all() and w.sum() == pytest.approx(1.0)
    rc = k.risk_contributions(w, cov)
    assert np.allclose(rc["pct"], 1 / 6, atol=1e-7)


def test_risk_parity_with_budget(cov):
    budget = {"a": 3, "b": 1, "c": 1, "d": 1, "e": 1, "f": 1}
    w = k.risk_parity_weights(cov, budget=budget)
    rc = k.risk_contributions(w, cov)
    assert rc.loc["a", "pct"] == pytest.approx(3 / 8, abs=1e-7)
