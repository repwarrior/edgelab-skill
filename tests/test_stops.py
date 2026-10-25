"""Tests for stop-loss handling.

A stop-loss backtest lies in three specific ways, and each has a test here:
filling at the stop price when the market gapped past it, booking stop exits on
a bar the position had already left, and quietly changing results for strategies
that never asked for a stop at all.
"""

from __future__ import annotations

import numpy as np
import pytest

from edgelab.engine import ExecutionEngine, _first_stop_hit
from edgelab.panel import build_panel
from edgelab.types import Bars
from edgelab.types import RawSeries, Rule, SignalDecision


def _panel(o, h, l, c, symbol="X"):
    n = len(c)
    d0 = np.datetime64("2026-01-01", "D")
    dates = np.array([d0 + np.timedelta64(i, "D") for i in range(n)])
    s = RawSeries(symbol=symbol, dates=dates,
                  o=np.array(o, float), h=np.array(h, float),
                  l=np.array(l, float), c=np.array(c, float),
                  volume=np.full(n, 1e6), dividend=np.zeros(n), split=np.zeros(n))
    return build_panel([s], dates, regime=np.full(n, "bull/calm", dtype=object))


def _rule(stop_atr=None, stop_pct=None, hold=5, side=1, warmup=1):
    def sig(window, p):
        # fire once, on the warmup bar only, so each test has exactly one trade
        if window.t != warmup:
            return None
        return SignalDecision(side=side, hold_bars=hold, tag="t",
                              stop_atr=stop_atr, stop_pct=stop_pct)
    return Rule("r", sig, warmup, {}, "", hold)


FLAT_O = [100.0] * 10
FLAT_H = [101.0] * 10
FLAT_C = [100.0] * 10


# -- the no-stop path must be untouched --------------------------------------

def test_no_stop_leaves_everything_scheduled():
    p = _panel(FLAT_O, FLAT_H, [99.0] * 10, FLAT_C)
    tr = ExecutionEngine().run(_rule(), p, audit=False).trades
    assert len(tr) == 1
    assert tr[0].terminated == "scheduled"
    assert tr[0].stop_px is None
    assert tr[0].gapped_through_stop is False


def test_a_stop_that_never_triggers_matches_no_stop_exactly():
    """Same path, one run with a far-away stop and one with none. If these
    differ, the stop code is changing results it has no business touching."""
    p = _panel(FLAT_O, FLAT_H, [99.0] * 10, FLAT_C)
    eng = ExecutionEngine()
    a = eng.run(_rule(), p, audit=False).trades[0]
    b = eng.run(_rule(stop_pct=0.90), p, audit=False).trades[0]
    assert a.exit_idx == b.exit_idx
    assert a.exit_px == b.exit_px
    assert a.net_ret == pytest.approx(b.net_ret)
    assert b.terminated == "scheduled"


# -- the stop must actually fire ---------------------------------------------

def test_stop_fires_and_fills_at_the_stop_price():
    low = [99.0] * 10
    low[3] = 94.0                       # breach on bar 3
    p = _panel(FLAT_O, FLAT_H, low, FLAT_C)
    tr = ExecutionEngine().run(_rule(stop_pct=0.05), p, audit=False).trades[0]
    assert tr.terminated == "stopped"
    assert tr.stop_px == pytest.approx(95.0)
    assert tr.exit_px == pytest.approx(95.0)
    assert tr.gapped_through_stop is False
    assert tr.exit_idx == 3


def test_stop_exits_earlier_than_the_schedule():
    low = [99.0] * 10
    low[2] = 90.0
    p = _panel(FLAT_O, FLAT_H, low, FLAT_C)
    eng = ExecutionEngine()
    stopped = eng.run(_rule(stop_pct=0.05), p, audit=False).trades[0]
    normal = eng.run(_rule(), p, audit=False).trades[0]
    assert stopped.exit_idx < normal.exit_idx
    assert stopped.bars_held < normal.bars_held


# -- gap risk must NOT be assumed away ---------------------------------------

def test_gap_through_stop_fills_at_the_open_not_the_stop():
    """The market never traded at the stop. Filling there would pretend a stop
    caps losses precisely; it does not."""
    o = list(FLAT_O); o[3] = 88.0
    low = [99.0] * 10; low[3] = 87.0
    p = _panel(o, FLAT_H, low, FLAT_C)
    tr = ExecutionEngine().run(_rule(stop_pct=0.05), p, audit=False).trades[0]
    assert tr.terminated == "stopped"
    assert tr.gapped_through_stop is True
    assert tr.exit_px == pytest.approx(88.0)
    assert tr.exit_px < tr.stop_px          # worse than the stop, as it must be
    assert tr.net_ret < -0.10               # the loss exceeds the stop distance


def test_gap_loss_is_worse_than_the_nominal_stop_distance():
    o = list(FLAT_O); o[3] = 80.0
    low = [99.0] * 10; low[3] = 79.0
    p = _panel(o, FLAT_H, low, FLAT_C)
    tr = ExecutionEngine().run(_rule(stop_pct=0.05), p, audit=False).trades[0]
    assert tr.gross_ret < -0.05, "a 5% stop produced a loss of 5% or less through a gap"


# -- the exit bar is not scannable -------------------------------------------

def test_stop_on_the_exit_bar_does_not_fire():
    """The scheduled exit fills at the exit bar's open, before its low prints."""
    p_low = [99.0] * 10
    normal = ExecutionEngine().run(_rule(hold=3), _panel(FLAT_O, FLAT_H, p_low, FLAT_C),
                                   audit=False).trades[0]
    low = [99.0] * 10
    low[normal.exit_idx] = 50.0         # deep breach, but only on the exit bar
    p = _panel(FLAT_O, FLAT_H, low, FLAT_C)
    tr = ExecutionEngine().run(_rule(hold=3, stop_pct=0.05), p, audit=False).trades[0]
    assert tr.terminated == "scheduled"
    assert tr.exit_idx == normal.exit_idx


# -- shorts ------------------------------------------------------------------

def test_short_stop_sits_above_entry_and_fires_on_a_rally():
    high = [101.0] * 10
    high[3] = 110.0
    p = _panel(FLAT_O, high, [99.0] * 10, FLAT_C)
    tr = ExecutionEngine().run(_rule(stop_pct=0.05, side=-1), p, audit=False).trades[0]
    assert tr.stop_px == pytest.approx(105.0)
    assert tr.terminated == "stopped"
    assert tr.exit_px == pytest.approx(105.0)


def test_short_gap_up_fills_worse_than_the_stop():
    o = list(FLAT_O); o[3] = 120.0
    high = [101.0] * 10; high[3] = 121.0
    p = _panel(o, high, [99.0] * 10, FLAT_C)
    tr = ExecutionEngine().run(_rule(stop_pct=0.05, side=-1), p, audit=False).trades[0]
    assert tr.gapped_through_stop is True
    assert tr.exit_px == pytest.approx(120.0)
    assert tr.exit_px > tr.stop_px


# -- ATR sizing --------------------------------------------------------------

def test_atr_stop_scales_with_volatility():
    rng = np.random.default_rng(0)
    n = 60
    c = 100 + np.cumsum(rng.normal(0, 0.5, n))
    o = c.copy(); h = c + 2.0; l = c - 2.0
    p = _panel(o, h, l, c)
    eng = ExecutionEngine()
    tight = eng.run(_rule(stop_atr=0.5, warmup=30), p, audit=False).trades
    wide = eng.run(_rule(stop_atr=8.0, warmup=30), p, audit=False).trades
    assert tight and wide
    # a wider ATR multiple must place the stop further from entry
    assert (wide[0].entry_px - wide[0].stop_px) > (tight[0].entry_px - tight[0].stop_px)


# -- the point of a stop -----------------------------------------------------

def test_stop_truncates_the_left_tail():
    """A stop should bound how bad the typical loss gets. It cannot bound the
    worst case -- that is what gap risk means."""
    rng = np.random.default_rng(3)
    n = 400
    c = 100 * np.exp(np.cumsum(rng.normal(0, 0.02, n)))
    o = c.copy()
    h = c * (1 + np.abs(rng.normal(0, 0.01, n)))
    l = c * (1 - np.abs(rng.normal(0, 0.01, n)))
    p = _panel(o, h, l, c)

    def every_bar(stop):
        def sig(window, params):
            return SignalDecision(side=1, hold_bars=10, tag="t", stop_pct=stop)
        return Rule("r", sig, 20, {}, "", 10)

    eng = ExecutionEngine()
    plain = eng.run(every_bar(None), p, audit=False).net_returns
    stopped = eng.run(every_bar(0.03), p, audit=False).net_returns
    assert plain.size > 10 and stopped.size > 10
    assert stopped.min() > plain.min(), "the stop did not improve the worst loss"
    assert np.percentile(stopped, 5) > np.percentile(plain, 5)


def test_unit_helper_excludes_exit_bar_and_handles_gaps():
    o = np.array([100.0, 100.0, 90.0, 100.0])
    h = np.array([101.0] * 4)
    l = np.array([99.0, 99.0, 89.0, 50.0])
    tr = np.ones(4, bool)
    assert _first_stop_hit(o, h, l, tr, 0, 3, 95.0, 1) == (2, 90.0, True)
    # bar 3 is the exit bar; its 50.0 low must be ignored
    assert _first_stop_hit(o, h, np.array([99.0, 99.0, 99.0, 50.0]), tr, 0, 3, 95.0, 1) is None


# -- slippage MAGNITUDE, not merely the fact of a gap -------------------------
# A boolean cannot separate a one-tick miss from a gap that doubled the loss.
# The size is surfaced so it stays visible instead of being buried inside
# net_ret.

def test_slippage_px_is_zero_when_the_stop_fills_at_the_stop():
    low = [99.0] * 10
    low[3] = 94.0                       # breaches 95.0 intrabar; opens above it
    p = _panel(FLAT_O, FLAT_H, low, FLAT_C)
    tr = ExecutionEngine().run(_rule(stop_pct=0.05), p, audit=False).trades[0]
    assert tr.terminated == "stopped"
    assert tr.gapped_through_stop is False
    assert tr.exit_px == pytest.approx(95.0)
    assert tr.slippage_px == pytest.approx(0.0)


def test_slippage_px_measures_the_size_of_the_gap():
    o = list(FLAT_O); o[3] = 88.0
    low = [99.0] * 10; low[3] = 87.0
    p = _panel(o, FLAT_H, low, FLAT_C)
    tr = ExecutionEngine().run(_rule(stop_pct=0.05), p, audit=False).trades[0]
    assert tr.gapped_through_stop is True
    assert tr.stop_px == pytest.approx(95.0)
    assert tr.exit_px == pytest.approx(88.0)
    assert tr.slippage_px == pytest.approx(7.0)


def test_slippage_px_is_zero_on_a_scheduled_exit():
    p = _panel(FLAT_O, FLAT_H, [99.0] * 10, FLAT_C)
    tr = ExecutionEngine().run(_rule(), p, audit=False).trades[0]
    assert tr.terminated == "scheduled"
    assert tr.slippage_px == pytest.approx(0.0)


def test_short_slippage_px_measures_the_size_of_the_gap():
    o = list(FLAT_O); o[3] = 120.0
    high = [101.0] * 10; high[3] = 121.0
    p = _panel(o, high, [99.0] * 10, FLAT_C)
    tr = ExecutionEngine().run(_rule(stop_pct=0.05, side=-1), p, audit=False).trades[0]
    assert tr.gapped_through_stop is True
    assert tr.stop_px == pytest.approx(105.0)
    assert tr.slippage_px == pytest.approx(15.0)


def test_slippage_px_is_the_difference_between_gross_loss_and_the_stop_distance():
    """The number must reconcile with the return, not float free of it."""
    o = list(FLAT_O); o[3] = 88.0
    low = [99.0] * 10; low[3] = 87.0
    p = _panel(o, FLAT_H, low, FLAT_C)
    tr = ExecutionEngine().run(_rule(stop_pct=0.05), p, audit=False).trades[0]
    nominal = (tr.stop_px - tr.entry_px) / tr.entry_px        # -5%
    assert tr.gross_ret == pytest.approx(nominal - tr.slippage_px / tr.entry_px)
