"""Strictly causal technical indicators.

Every indicator here obeys one contract: the value at index t is a function of
bars[0..t] only. That is enforced empirically by `assert_causal`, which is run
over the whole registry in the test suite rather than trusted by inspection.

The classic ways this contract gets broken -- and what the prober catches:
  - centered rolling windows (`center=True`)
  - full-sample normalization (z-scoring against the whole series' mean/std)
  - backward fill (`bfill`) -- pulls a future observation into the past
  - zero-phase filters (`scipy.signal.filtfilt`)
  - an off-by-one `.shift(-1)` where `.shift(1)` was meant

All indicators return a float array the same length as the input, NaN during
warmup. NaN is never filled in: a NaN means "this indicator has no opinion on
this bar", and the engine declines to trade on it.
"""

from __future__ import annotations

from typing import Callable, Mapping

import numpy as np
import pandas as pd

from .types import Bars

INDICATOR_REGISTRY: dict[str, Callable[..., np.ndarray]] = {}


def causal(fn: Callable[..., np.ndarray]) -> Callable[..., np.ndarray]:
    """Register an indicator as claiming causality. The claim is tested."""
    INDICATOR_REGISTRY[fn.__name__] = fn
    return fn


def _s(a: np.ndarray) -> pd.Series:
    return pd.Series(np.asarray(a, dtype=float))


# --------------------------------------------------------------------------
# Trend / location
# --------------------------------------------------------------------------

@causal
def sma(bars: Bars, period: int = 20) -> np.ndarray:
    return _s(bars.c).rolling(period, min_periods=period).mean().to_numpy(copy=True)


@causal
def ema(bars: Bars, period: int = 20) -> np.ndarray:
    out = _s(bars.c).ewm(span=period, adjust=False).mean().to_numpy(copy=True)
    out[: period - 1] = np.nan          # honest warmup
    return out


@causal
def dist_sma(bars: Bars, period: int = 200) -> np.ndarray:
    """Fractional distance of close from its SMA. Scale-free, so it is
    comparable across symbols and across time."""
    m = sma(bars, period)
    with np.errstate(invalid="ignore", divide="ignore"):
        return (np.asarray(bars.c, dtype=float) - m) / m


@causal
def pct_52w(bars: Bars, period: int = 252) -> np.ndarray:
    """Position of close within its trailing range, 0 = at the low, 1 = at the high."""
    c = _s(bars.c)
    hi = c.rolling(period, min_periods=period).max()
    lo = c.rolling(period, min_periods=period).min()
    rng = (hi - lo).replace(0.0, np.nan)
    return ((c - lo) / rng).to_numpy(copy=True)


# --------------------------------------------------------------------------
# Oscillators
# --------------------------------------------------------------------------

@causal
def rsi(bars: Bars, period: int = 14) -> np.ndarray:
    """Wilder's RSI via recursive smoothing (alpha = 1/period)."""
    c = _s(bars.c)
    delta = c.diff()
    gain = delta.clip(lower=0.0)
    loss = (-delta).clip(lower=0.0)
    avg_gain = gain.ewm(alpha=1.0 / period, adjust=False).mean()
    avg_loss = loss.ewm(alpha=1.0 / period, adjust=False).mean()
    rs = avg_gain / avg_loss.replace(0.0, np.nan)
    out = (100.0 - 100.0 / (1.0 + rs)).to_numpy(copy=True)
    # avg_loss == 0 means an unbroken run of gains -> RSI 100, not NaN.
    out[np.asarray(avg_loss.eq(0.0) & avg_gain.gt(0.0))] = 100.0
    out[:period] = np.nan
    return out


@causal
def macd_hist(bars: Bars, fast: int = 12, slow: int = 26, signal: int = 9) -> np.ndarray:
    c = _s(bars.c)
    line = c.ewm(span=fast, adjust=False).mean() - c.ewm(span=slow, adjust=False).mean()
    sig = line.ewm(span=signal, adjust=False).mean()
    out = (line - sig).to_numpy(copy=True)
    out[: slow + signal] = np.nan
    return out


@causal
def bollinger_z(bars: Bars, period: int = 20) -> np.ndarray:
    """Standardized distance from the middle band. +2 == upper band touch."""
    c = _s(bars.c)
    m = c.rolling(period, min_periods=period).mean()
    sd = c.rolling(period, min_periods=period).std(ddof=0).replace(0.0, np.nan)
    return ((c - m) / sd).to_numpy(copy=True)


@causal
def williams_r(bars: Bars, period: int = 14) -> np.ndarray:
    h = _s(bars.h).rolling(period, min_periods=period).max()
    l = _s(bars.l).rolling(period, min_periods=period).min()
    rng = (h - l).replace(0.0, np.nan)
    return (-100.0 * (h - _s(bars.c)) / rng).to_numpy(copy=True)


@causal
def ultimate_osc(bars: Bars, p1: int = 7, p2: int = 14, p3: int = 28) -> np.ndarray:
    c, h, l = _s(bars.c), _s(bars.h), _s(bars.l)
    prev_c = c.shift(1)                       # shift(1) = yesterday. Never shift(-1).
    bp = c - pd.concat([l, prev_c], axis=1).min(axis=1)
    tr = pd.concat([h, prev_c], axis=1).max(axis=1) - pd.concat([l, prev_c], axis=1).min(axis=1)

    def avg(n: int) -> pd.Series:
        return bp.rolling(n, min_periods=n).sum() / tr.rolling(n, min_periods=n).sum().replace(0.0, np.nan)

    out = (100.0 * (4 * avg(p1) + 2 * avg(p2) + avg(p3)) / 7.0).to_numpy(copy=True)
    out[: p3 + 1] = np.nan
    return out


@causal
def vix_fix(bars: Bars, period: int = 22) -> np.ndarray:
    """Williams VIX Fix: synthetic volatility from the price series alone.

    (highest_close(period) - low) / highest_close(period) * 100
    """
    hc = _s(bars.c).rolling(period, min_periods=period).max()
    return (100.0 * (hc - _s(bars.l)) / hc.replace(0.0, np.nan)).to_numpy(copy=True)


@causal
def atr(bars: Bars, period: int = 14) -> np.ndarray:
    h, l, c = _s(bars.h), _s(bars.l), _s(bars.c)
    prev_c = c.shift(1)
    tr = pd.concat(
        [(h - l).abs(), (h - prev_c).abs(), (l - prev_c).abs()], axis=1
    ).max(axis=1)
    out = tr.ewm(alpha=1.0 / period, adjust=False).mean().to_numpy(copy=True)
    out[:period] = np.nan
    return out


@causal
def realized_vol(bars: Bars, period: int = 20) -> np.ndarray:
    """Annualized realized volatility of log returns."""
    r = np.log(_s(bars.c)).diff()
    return (r.rolling(period, min_periods=period).std(ddof=0) * np.sqrt(252.0)).to_numpy(copy=True)


# --------------------------------------------------------------------------
# The causality prober
# --------------------------------------------------------------------------

def assert_causal(
    fn: Callable[..., np.ndarray],
    bars: Bars,
    n_probes: int = 60,
    seed: int = 0,
    tol: float = 1e-9,
    **params,
) -> None:
    """Empirically verify that fn(bars[:t+1])[-1] == fn(bars)[t].

    If an indicator can see the future, truncating the series at t changes the
    value it reports for t. This compares the two directly at random t. It is a
    proof by construction rather than by code review, which is why it runs over
    the entire registry in the test suite.
    """
    n = len(bars)
    full = np.asarray(fn(bars, **params), dtype=float)
    if full.shape[0] != n:
        raise AssertionError(f"{fn.__name__} returned length {full.shape[0]}, expected {n}")

    rng = np.random.default_rng(seed)
    lo = max(2, n // 4)
    if lo >= n:
        return
    for t in rng.choice(np.arange(lo, n), size=min(n_probes, n - lo), replace=False):
        t = int(t)
        truncated = np.asarray(fn(bars.head(t + 1), **params), dtype=float)[-1]
        reference = full[t]
        if np.isnan(truncated) and np.isnan(reference):
            continue
        if np.isnan(truncated) != np.isnan(reference):
            raise AssertionError(
                f"{fn.__name__} is NOT causal at t={t}: "
                f"truncated={truncated!r} full={reference!r} (NaN mismatch)"
            )
        denom = max(1.0, abs(reference))
        if abs(truncated - reference) / denom > tol:
            raise AssertionError(
                f"{fn.__name__} is NOT causal at t={t}: "
                f"truncated={truncated!r} != full={reference!r}"
            )


def probe_all(bars: Bars, seed: int = 0) -> dict[str, str]:
    """Run the causality probe over every registered indicator."""
    results: dict[str, str] = {}
    for name, fn in sorted(INDICATOR_REGISTRY.items()):
        try:
            assert_causal(fn, bars, seed=seed)
            results[name] = "causal"
        except AssertionError as exc:
            results[name] = f"FAIL: {exc}"
    return results
