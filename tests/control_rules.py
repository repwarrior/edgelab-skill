"""Rules used ONLY as experimental controls by the test suite.

None of these is a trading idea. They exist so the harness can be checked from
both directions: on noise every one of them must fail, and on a series with an
edge injected after "close below its 15-bar mean" days, `dip_below_mean` must be
found. Two deliberately broken objects live here too, to prove the safety checks
can actually fire.
"""
from __future__ import annotations

import zlib

import numpy as np

from edgelab import Rule, SignalDecision


def dip_below_mean(w, p):
    """Matches the injected edge in the synthetic data exactly."""
    if len(w.c) < 15:
        return None
    look = w.c[-15:]
    return SignalDecision(1, p["hold"]) if look[-1] < look.mean() * 0.96 else None


def every_nth_bar(w, p):
    return SignalDecision(1, p["hold"]) if w.t % p["n"] == 0 else None


def coin_flip(w, p):
    """Deterministic pseudo-random entries: a hash of (symbol, bar)."""
    u = zlib.crc32(f"{w.symbol}:{w.t}:{p['salt']}".encode()) / 2**32
    return SignalDecision(p["side"], p["hold"]) if u < p["rate"] else None


def new_high(w, p):
    n = p["lookback"]
    if len(w.c) < n + 1:
        return None
    return SignalDecision(1, p["hold"]) if w.c[-1] > np.max(w.h[-n - 1:-1]) else None


def control_catalogue(hold: int = 10) -> list[Rule]:
    return [
        Rule("dip_below_mean", dip_below_mean, 30, {"hold": hold}, hold_bars=hold),
        Rule("every_7th_bar", every_nth_bar, 30, {"n": 7, "hold": hold}, hold_bars=hold),
        Rule("coin_long", coin_flip, 30, {"side": 1, "rate": 0.08, "salt": 1, "hold": hold}, hold_bars=hold),
        Rule("coin_short", coin_flip, 30, {"side": -1, "rate": 0.08, "salt": 2, "hold": hold}, hold_bars=hold),
        Rule("new_20_bar_high", new_high, 30, {"lookback": 20, "hold": hold}, hold_bars=hold),
    ]


# ---- deliberately broken -----------------------------------------------------

def peeking_rule(w, p):
    """Tries to read tomorrow. Must raise IndexError, never return a price."""
    tomorrow = w.c[w.t + 1]
    if tomorrow > w.c[-1]:
        return SignalDecision(1, p.get("hold", 5))
    return None


def noncausal_zscore(bars, period: int = 20) -> np.ndarray:
    """Standardised against the WHOLE sample, future included. Not registered;
    exists so the causality prober can be seen to catch it."""
    c = np.asarray(bars.c, dtype=float)
    return (c - np.nanmean(c)) / (np.nanstd(c) + 1e-12)
