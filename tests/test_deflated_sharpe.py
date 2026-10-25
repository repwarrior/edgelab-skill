"""The deflated-Sharpe hurdle must be estimable when only one rule is live.

The trial-Sharpe variance `sr_var` is meant to answer: across trials that have
NO edge, how much does Sharpe move around by luck? Estimating it from the one
observed Sharpe (`sr_var = sr**2`) answers a different question and produces a
hurdle of ~2.2*|sr| -- above the observation by construction, for every sr.

That defect is invisible in the arithmetic and obvious the moment a dataset with
a known injected edge is pushed through the gate, so these tests push one
through. A guard is unverified until it has been seen to fire.
"""

from __future__ import annotations

import datetime as _dt
import json

import numpy as np
import pytest

from edgelab import build_panel
from edgelab import config as cfg
from edgelab import stats as st
from edgelab.types import PortfolioMode, RawSeries, RunResult, Trade
from edgelab.verdict import assess_grid

N_BARS = 800
SYMS = [f"S{i}" for i in range(6)]
HOLD = 10
N_TRADES = 150

# Pinned so the hurdle is fixed and the test never touches a real trial counter.
N_TRIALS = 40


# --------------------------------------------------------------------------
# Synthetic material
# --------------------------------------------------------------------------

def _synth(symbol: str, seed: int) -> RawSeries:
    rng = np.random.default_rng(seed)
    r = rng.normal(0.0003, 0.015, N_BARS)
    c = 100 * np.exp(np.cumsum(r))
    h = c * (1 + np.abs(rng.normal(0, 0.004, N_BARS)))
    l = c * (1 - np.abs(rng.normal(0, 0.004, N_BARS)))
    o = l + (h - l) * rng.random(N_BARS)
    dates = np.array([np.datetime64("2019-01-01", "D") + np.timedelta64(i, "D")
                      for i in range(N_BARS)])
    return RawSeries(symbol=symbol, dates=dates, o=o, h=h, l=l, c=c,
                     volume=np.full(N_BARS, 2e6), dividend=np.zeros(N_BARS),
                     split=np.zeros(N_BARS))


@pytest.fixture(scope="module")
def panel():
    series = [_synth(s, seed=7 + i) for i, s in enumerate(SYMS)]
    return build_panel(series, series[0].dates)


@pytest.fixture(autouse=True)
def pinned_trials(tmp_path, monkeypatch):
    """Pin n_trials without reading or writing the real counter."""
    p = tmp_path / "trials.json"
    p.write_text(json.dumps({"total_trials": N_TRIALS,
                             "untracked_count": N_TRIALS - 1,
                             "strategies": [], "history": []}))
    monkeypatch.setattr(cfg, "TRIALS_FILE", p)
    return p


def _result(panel, target_sharpe: float, rule_id: str, n_trades: int = N_TRADES,
            seed: int = 3, sigma: float = 0.02) -> RunResult:
    """A run whose per-trade Sharpe is EXACTLY `target_sharpe`.

    Standardising the draws puts the edge under test rather than the RNG: the
    assertions below are about what the hurdle does with a known Sharpe.
    """
    rng = np.random.default_rng(seed)
    z = rng.normal(0.0, 1.0, n_trades)
    z = (z - z.mean()) / z.std(ddof=1)
    net = z * sigma + target_sharpe * sigma

    entries = np.linspace(40, N_BARS - HOLD - 20, n_trades).astype(int)
    dates = panel.dates
    trades = []
    for k, (e, nr) in enumerate(zip(entries, net)):
        sym = SYMS[k % len(SYMS)]
        o = panel.col("o", sym)
        x = int(e) + HOLD
        trades.append(Trade(
            symbol=sym, side=1, tag="", signal_idx=int(e) - 1, entry_idx=int(e),
            exit_idx=x,
            signal_date=_d(dates[int(e) - 1]), entry_date=_d(dates[int(e)]),
            exit_date=_d(dates[x]),
            entry_px=float(o[int(e)]), exit_px=float(o[x]),
            dividends_cash=0.0, cost_cash=0.0006, borrow_cash=0.0,
            gross_ret=float(nr) + 0.0006, net_ret=float(nr),
            bars_held=HOLD, fill_delay_bars=1, terminated="scheduled",
            regime="bull/calm" if k % 2 else "bear/stressed",
        ))
    return RunResult(rule_id=rule_id, params={"hold": HOLD, "sr": target_sharpe},
                     mode=PortfolioMode.NON_OVERLAPPING, trades=trades,
                     symbols=tuple(SYMS), causality_verified=True)


def _d(dt64) -> _dt.date:
    return np.datetime64(dt64, "D").astype(_dt.date)


# --------------------------------------------------------------------------
# The defect, stated as arithmetic
# --------------------------------------------------------------------------

def test_own_sharpe_fallback_is_unclearable_by_construction():
    """sr_var = sr**2 puts the hurdle above the observation for every sr."""
    for sr in (0.05, 0.2, 0.5, 1.0, 3.0):
        hurdle = st.expected_max_sharpe(sr ** 2, N_TRIALS)
        assert hurdle > sr, (
            f"sr={sr}: hurdle {hurdle:.3f} -- if this ever fails the sr**2 "
            f"fallback is not, in fact, always unclearable"
        )


# --------------------------------------------------------------------------
# The defect, as the gate actually behaves
# --------------------------------------------------------------------------

def test_single_live_rule_with_injected_edge_clears_the_hurdle(panel):
    a = assess_grid([_result(panel, 0.60, "edge")], panel, count_trials=False)[0]
    thr = cfg.SETTINGS.stats.dsr_threshold
    assert a.dsr >= thr, (
        f"per-trade Sharpe {a.sharpe_net:.3f} injected by hand scored a deflated "
        f"Sharpe of {a.dsr:.3f} against hurdle SR {a.dsr_hurdle:.3f} from "
        f"{a.n_trials} trials -- below the {thr} PROVEN gate. A gate a known "
        f"edge cannot pass is not conservative, it is uninformative."
    )


def test_single_live_rule_on_noise_scores_near_zero(panel):
    a = assess_grid([_result(panel, 0.0, "noise")], panel, count_trials=False)[0]
    assert a.dsr < 0.05, (
        f"a zero-Sharpe run scored a deflated Sharpe of {a.dsr:.3f} against "
        f"hurdle SR {a.dsr_hurdle:.3f}; the fix has made the gate credulous"
    )
    assert a.dsr_hurdle > 0.0, "no hurdle at all was applied"


def test_one_live_rule_beside_a_starved_one_still_clears(panel):
    """The fallback fires whenever exactly one rule clears min_trades_hard,
    not only in a grid of one."""
    grid = [_result(panel, 0.60, "edge"),
            _result(panel, 5.0, "starved", n_trades=5, seed=9)]
    ass = assess_grid(grid, panel, count_trials=False)
    edge = next(a for a in ass if a.rule_id == "edge")
    assert edge.dsr >= cfg.SETTINGS.stats.dsr_threshold, (
        f"deflated Sharpe {edge.dsr:.3f} against hurdle {edge.dsr_hurdle:.3f}; "
        f"a starved companion rule must not resurrect the sr**2 fallback"
    )


# --------------------------------------------------------------------------
# Discrimination must not have been bought with credulity
# --------------------------------------------------------------------------

def test_single_real_rule_on_a_random_walk_is_not_proven():
    """The negative control, on the changed path: a genuine rule, genuine
    random-walk data, assessed alone. Loosening the single-live-rule hurdle
    must not let noise through it."""
    from edgelab import ExecutionEngine, Verdict
    from control_rules import control_catalogue

    series = [_synth(s, seed=7 + i) for i, s in enumerate(SYMS)]
    walk = build_panel(series, series[0].dates)
    rule = next(r for r in control_catalogue(hold=HOLD)
                if r.rule_id == "dip_below_mean")
    res = ExecutionEngine().run(rule, walk)
    assert res.net_returns.size >= cfg.SETTINGS.stats.min_trades_hard

    a = assess_grid([res], walk, count_trials=False)[0]
    assert a.verdict is not Verdict.PROVEN, (
        f"graded PROVEN on a random walk: SR {a.sharpe_net:+.3f} against hurdle "
        f"{a.dsr_hurdle:.3f}, dsr {a.dsr:.3f}"
    )
    assert a.dsr_hurdle > 0.0, (
        "a single-rule grid was assessed against no luck hurdle at all"
    )


# --------------------------------------------------------------------------
# The multi-rule path uses the cross-rule variance
# --------------------------------------------------------------------------

def test_multi_rule_hurdle_is_still_the_cross_rule_variance(panel):
    grid = [_result(panel, sr, f"r{i}", seed=20 + i)
            for i, sr in enumerate((0.1, 0.3, 0.6))]
    ass = assess_grid(grid, panel, count_trials=False)
    sr_var = float(np.var([0.1, 0.3, 0.6], ddof=1))
    expected = st.expected_max_sharpe(sr_var, ass[0].n_trials)
    for a in ass:
        assert a.dsr_hurdle == pytest.approx(expected, rel=1e-9), (
            f"{a.rule_id}: hurdle {a.dsr_hurdle:.6f} != {expected:.6f}; the "
            f"multi-rule estimate must remain the variance across the family"
        )


def test_a_missing_trial_variance_never_becomes_a_missing_hurdle():
    """trial_sharpe_variance returns NaN when the null cannot be estimated, so the
    caller must decide. Turning that into 0.0 would make deflated_sharpe report a
    PASS built out of absent data."""
    import numpy as np
    from edgelab import stats as st
    assert np.isnan(st.expected_max_sharpe(float("nan"), 50))
    assert np.isnan(st.expected_max_sharpe(-1.0, 50))
    dsr, hurdle = st.deflated_sharpe(np.random.default_rng(0).normal(0.001, 0.01, 4000), float("nan"), 50)
    assert np.isnan(dsr) and np.isnan(hurdle)          # unknown, not a pass
    assert not (dsr > 0.95)
    # the legitimate zero-hurdle cases still return 0.0
    assert st.expected_max_sharpe(0.0, 50) == 0.0      # no dispersion across trials
    assert st.expected_max_sharpe(0.25, 1) == 0.0      # one trial is no selection


def test_the_naive_search_bar_pins_family_wise_error_at_63_percent_whatever_n():
    """1 - 1/n reads as if it tightens with more trials. It does not."""
    import numpy as np
    from edgelab import stats as st
    for n in (5, 20, 50, 1000):
        b = (1.0 - 1.0 / n)
        fwer = 1.0 - b ** n
        assert 0.63 < fwer < 0.68, (n, b, fwer)          # flat in n, never near 0.05
    assert abs((1.0 - (1.0 - 1.0 / 10_000) ** 10_000) - (1 - 1 / np.e)) < 0.01


def test_the_calibrated_search_bar_actually_holds_the_error_rate():
    import numpy as np
    from edgelab import stats as st
    for n in (5, 20, 50, 1000):
        for fwer in (0.05, 0.01):
            b = st.search_bar(n, fwer)
            assert abs((1.0 - b ** n) - fwer) < 1e-9
            assert b > (1.0 - 1.0 / n)           # strictly stricter than the naive bar
    assert abs(st.search_bar(50) - 0.998975) < 1e-5
    # a Monte Carlo check that the bar does what the algebra says
    rng = np.random.default_rng(7)
    n, trials = 50, 20_000
    maxes = rng.random((trials, n)).max(axis=1)
    assert abs(float(np.mean(maxes > st.search_bar(n, 0.05))) - 0.05) < 0.01
    assert 0.60 < float(np.mean(maxes > (1.0 - 1.0 / n))) < 0.67
    for bad in (0, -1):
        with pytest.raises(ValueError):
            st.search_bar(bad)
