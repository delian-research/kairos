from __future__ import annotations

import pandas as pd
import pytest

import kairos as k


def test_estimate_str_and_dict() -> None:
    r = pd.Series(
        [0.01, -0.02, 0.015, 0.0], index=pd.bdate_range("2026-01-05", periods=4)
    )
    est = k.realized_vol(r, annualize=True)
    s = str(est)
    assert "close-to-close" in s and "n=4" in s
    assert "2026-01-05 to 2026-01-08" in s and "annualized=True" in s
    d = est.as_dict()
    assert d["value"] == est.value and d["ddof"] == 1
    assert float(est) == est.value


def test_estimate_is_frozen() -> None:
    est = k.Estimate(1.0, 1, None, None, "x")
    with pytest.raises(AttributeError):
        est.value = 2.0  # type: ignore[misc]


def test_annualize_return() -> None:
    assert k.annualize_return(0.001, "log") == pytest.approx(0.252)
    assert k.annualize_return(0.001, "simple") == pytest.approx(1.001**252 - 1)
