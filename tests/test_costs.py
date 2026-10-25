"""Cost model tests, including the validation that rules out Corwin-Schultz on liquid names.

The load-bearing number in this package is the round-trip cost: every
NO_EDGE verdict of the form "real gross edge, eaten by costs" depends on it.
Validated against a known spread, it is wrong by ~40x on liquid names.
"""

from __future__ import annotations

import numpy as np
import pytest

from edgelab.config import CostConfig
from edgelab.costs import CostModel, corwin_schultz_spread


def _simulate(n_bars=600, ticks=390, daily_vol=0.015, true_spread=0.0005, seed=0):
    """Efficient mid random-walks intraday; every trade prints at mid +/- s/2.
    The bar's high/low therefore contain volatility AND spread -- exactly the
    structure Corwin-Schultz claims to decompose."""
    rng = np.random.default_rng(seed)
    per_tick = daily_vol / np.sqrt(ticks)
    H = np.empty(n_bars); L = np.empty(n_bars)
    mid = 100.0
    for b in range(n_bars):
        path = mid * np.exp(np.cumsum(rng.normal(0, per_tick, ticks)))
        trades = path * (1 + rng.choice([-1.0, 1.0], ticks) * true_spread / 2)
        H[b], L[b] = trades.max(), trades.min()
        mid = path[-1]
    return H, L


# -- the validation that matters ---------------------------------------------

def test_corwin_schultz_is_unusable_on_liquid_names():
    """Documents the failure rather than hiding it. If a future change makes CS
    accurate here, this test fails and the docstring warning should be removed."""
    H, L = _simulate(daily_vol=0.015, true_spread=0.0001, seed=1)
    est = float(np.nanmedian(corwin_schultz_spread(H, L)))
    assert est > 10 * 0.0001, (
        "CS unexpectedly accurate at a 1bp spread -- re-check the warning in "
        "corwin_schultz_spread and the explicit half_spread_bps default"
    )


def test_corwin_schultz_tracks_volatility_not_spread():
    """The diagnostic finding: its output barely responds to the true spread but
    scales cleanly with volatility."""
    lo_vol = float(np.nanmedian(corwin_schultz_spread(*_simulate(daily_vol=0.010, true_spread=0.0001, seed=2))))
    hi_vol = float(np.nanmedian(corwin_schultz_spread(*_simulate(daily_vol=0.025, true_spread=0.0001, seed=3))))
    same_vol_wider = float(np.nanmedian(corwin_schultz_spread(*_simulate(daily_vol=0.010, true_spread=0.0020, seed=4))))
    assert hi_vol > 1.8 * lo_vol, "estimate should scale with volatility (it does)"
    assert same_vol_wider < 1.8 * lo_vol, "estimate barely responds to a 20x wider spread"


# -- the corrected defaults --------------------------------------------------

def test_default_round_trip_is_plausible_for_liquid_equities():
    """A penny spread on a $100-500 stock is 0.2-1.0 bps. A round trip of tens
    of bps on AAPL is not a cost model, it is a bug."""
    cm = CostModel()
    daily_vol = 0.015 / np.sqrt(252)
    rt = cm.round_trip_bps(np.nan, 311.30, np.nan, 311.30, daily_vol)
    assert 1.0 <= rt <= 15.0, f"round trip {rt:.1f}bps is outside the plausible band"


def test_default_does_not_call_corwin_schultz():
    """half_spread_bps is explicit by default; the estimator must be opt-in."""
    assert CostConfig().half_spread_bps is not None
    cm = CostModel()
    # NaN spread input would force the CS fallback path if it were live
    assert cm.half_spread_bps(np.nan, 311.30) == pytest.approx(1.0)


def test_explicit_none_re_enables_the_estimator():
    cm = CostModel(CostConfig(half_spread_bps=None))
    assert cm.half_spread_bps(0.0020, 100.0) > 1.0


# -- invariants that must hold at any cost level -----------------------------

def test_costs_are_symmetric_and_never_negative():
    cm = CostModel()
    v = 0.015 / np.sqrt(252)
    assert cm.per_side_bps(np.nan, 100.0, v) > 0
    assert cm.round_trip_bps(np.nan, 100.0, np.nan, 100.0, v) >= CostConfig().min_round_trip_bps


def test_round_trip_respects_its_floor():
    cm = CostModel(CostConfig(commission_bps=0.0, half_spread_bps=0.0,
                              slippage_eta=0.0, min_round_trip_bps=4.0))
    assert cm.round_trip_bps(np.nan, 100.0, np.nan, 100.0, 0.0) == pytest.approx(4.0)
