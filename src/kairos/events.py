"""Event studies: abnormal returns around a list of event dates."""

from __future__ import annotations

import math
from collections.abc import Iterable
from typing import Any

import numpy as np
import pandas as pd

from ._util import align, check_returns, frame_attrs, span


def event_study(
    asset: pd.Series,
    benchmark: pd.Series,
    events: Iterable[Any],
    *,
    window: tuple[int, int] = (-5, 5),
    estimation: int = 120,
    gap: int = 10,
) -> pd.DataFrame:
    """Market-model event study for one asset.

    For each event, a market model ``asset = alpha + beta * benchmark`` is
    fitted on the ``estimation`` periods that end ``gap`` periods before the
    event window, so the event cannot contaminate its own baseline. Abnormal
    return (AR) is the actual return minus the model's prediction; cumulative
    abnormal return (CAR) sums AR from the start of the window.

    Args:
        asset, benchmark: Returns, inner-joined on the index. Log returns
            make CAR an exact sum.
        events: Event dates. An event is placed on the first index date on or
            after it, so an announcement on a non-trading day counts on the
            next session.
        window: ``(start, end)`` offsets in periods relative to the event
            (day 0), counted by position in the index, e.g. ``(-5, 5)``.
        estimation: Length of the estimation window in periods.
        gap: Periods between the estimation window and the event window.

    Returns a DataFrame indexed by relative period with columns ``mean_ar``,
    ``car`` (cumulative mean AR), ``t`` and ``n_events``. With two or more
    events, ``t`` is the cross-sectional t-statistic of CAR across events;
    with one event it is CAR over the estimation residual volatility scaled
    by the square root of elapsed periods.

    ``df.attrs["kairos"]`` holds ``events_used``, ``events_skipped`` (date and
    reason, for events too close to the edge of the data) and ``event_car``
    (each event's CAR over the full window). Overlapping event windows are
    treated as independent, which overstates ``t`` when events cluster.
    """
    lo, hi = window
    if lo > hi:
        raise ValueError("window must be (start, end) with start <= end")
    if estimation < 10:
        raise ValueError("estimation must be at least 10 periods")
    if gap < 0:
        raise ValueError("gap must be non-negative")
    a, b = align(asset, benchmark)
    check_returns(a, "asset")
    check_returns(b, "benchmark")
    if not a.index.is_monotonic_increasing:
        raise ValueError("returns must be sorted by date")
    av, bv = a.to_numpy(), b.to_numpy()
    n = len(a)
    offsets = np.arange(lo, hi + 1)

    ars: list[np.ndarray] = []
    resid_sd: list[float] = []
    used: list[Any] = []
    skipped: list[tuple[Any, str]] = []
    for ev in events:
        pos = int(a.index.searchsorted(ev, side="left"))
        est_end = pos + lo - gap
        est_start = est_end - estimation
        if pos >= n or pos + hi >= n:
            skipped.append((ev, "event window runs past the end of the data"))
            continue
        if est_start < 0:
            skipped.append((ev, "not enough history for the estimation window"))
            continue
        x, y = bv[est_start:est_end], av[est_start:est_end]
        xc = x - x.mean()
        sxx = float(np.dot(xc, xc))
        if sxx == 0:
            skipped.append((ev, "benchmark is constant in the estimation window"))
            continue
        slope = float(np.dot(xc, y - y.mean())) / sxx
        alpha = float(y.mean()) - slope * float(x.mean())
        resid = y - (alpha + slope * x)
        resid_sd.append(math.sqrt(float(np.dot(resid, resid)) / (estimation - 2)))
        sl = slice(pos + lo, pos + hi + 1)
        ars.append(av[sl] - (alpha + slope * bv[sl]))
        used.append(a.index[pos])

    if not ars:
        reasons = "; ".join(f"{d}: {why}" for d, why in skipped) or "no events given"
        raise ValueError(f"no usable events ({reasons})")

    ar = np.vstack(ars)
    cars = np.cumsum(ar, axis=1)
    m = len(ars)
    car = cars.mean(axis=0)
    if m >= 2:
        se = cars.std(axis=0, ddof=1) / math.sqrt(m)
        t_kind = "cross-sectional"
    else:
        se = resid_sd[0] * np.sqrt(np.arange(1, len(offsets) + 1))
        t_kind = "time-series (single event)"
    t = np.divide(car, se, out=np.full(len(offsets), np.nan), where=se > 0)
    out = pd.DataFrame(
        {"mean_ar": ar.mean(axis=0), "car": car, "t": t, "n_events": m},
        index=pd.Index(offsets, name="period"),
    )
    start, end = span(a.index)
    return frame_attrs(
        out,
        method="event-study-market-model",
        window=window,
        estimation=estimation,
        gap=gap,
        t_stat=t_kind,
        n=n,
        start=start,
        end=end,
        events_used=used,
        events_skipped=skipped,
        event_car=pd.Series(cars[:, -1], index=used, name="car"),
    )
