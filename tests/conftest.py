from __future__ import annotations

import numpy as np
import pandas as pd
import pytest


@pytest.fixture
def dates() -> pd.DatetimeIndex:
    return pd.bdate_range("2024-01-01", periods=600)


@pytest.fixture
def rng() -> np.random.Generator:
    return np.random.default_rng(12345)
