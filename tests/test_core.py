"""Acceptance tests for edgelab.

The load-bearing ones are the two controls:

  - a pure random walk must NEVER be graded PROVEN (no false positives)
  - a series with a known injected edge MUST be detected (not merely always-negative)

Either one alone is easy to pass with a broken harness. A harness that always
says NO_EDGE passes the first; a credulous one passes the second. Both together
constrain it to be actually calibrated.
"""

from __future__ import annotations

import numpy as np
import pytest

from edgelab import (
    ExecutionEngine,
    Verdict,
    assess_grid,
    build_panel,
    verify_causality,
)
from edgelab import indicators as ind
from edgelab import stats as st
from edgelab.types import Bars, RawSeries, Rule

import control_rules as cr
from control_rules import control_catalogue

N_BARS = 1500
N_SYMS = 8


# --------------------------------------------------------------------------
# Synthetic data
# --------------------------------------------------------------------------

def _synth(symbol: str, n: int = N_BARS, seed: int = 0, drift: float = 0.0003,
           edge: float = 0.0) -> RawSeries:
    """Geometric random walk. If `edge` is set, bars following a genuine
    oversold condition get a real positive bump -- a detectable edge."""
    rng = np.random.default_rng(seed)
    r = rng.normal(drift, 0.015, n)
    if edge:
        c0 = 100 * np.exp(np.cumsum(r))
        for t in range(30, n - 12):
            look = c0[t - 14: t + 1]
            if look[-1] < look.mean() * 0.96:
                r[t + 1: t + 11] += edge
    c = 100 * np.exp(np.cumsum(r))
    h = c * (1 + np.abs(rng.normal(0, 0.004, n)))
    l = c * (1 - np.abs(rng.normal(0, 0.004, n)))
    o = l + (h - l) * rng.random(n)
    v = rng.integers(1_000_000, 5_000_000, n).astype(float)
    dates = np.array([np.datetime64("2018-01-01", "D") + np.timedelta64(i, "D")
                      for i in range(n)])
    return RawSeries(symbol=symbol, dates=dates, o=o, h=h, l=l, c=c, volume=v,
                     dividend=np.zeros(n), split=np.zeros(n))


def _panel(edge: float = 0.0, seed0: int = 0, n_syms: int = N_SYMS, n: int = N_BARS):
    series = [_synth(f"S{i}", n=n, seed=seed0 + i, edge=edge) for i in range(n_syms)]
    cal = series[0].dates
    regime = np.array(["bull/calm"] * (n // 2) + ["bear/stressed"] * (n - n // 2),
                      dtype=object)
    return build_panel(series, cal, regime=regime)


@pytest.fixture(scope="module")
def noise_panel():
    return _panel(edge=0.0, seed0=0)


@pytest.fixture(scope="module")
def edge_panel():
    return _panel(edge=0.004, seed0=100)


@pytest.fixture(scope="module")
def noise_assessments(noise_panel):
    eng = ExecutionEngine()
    results = [eng.run(r, noise_panel) for r in control_catalogue(hold=10)]
    return assess_grid(results, noise_panel, count_trials=False)


@pytest.fixture(scope="module")
def edge_assessments(edge_panel):
    eng = ExecutionEngine()
    results = [eng.run(r, edge_panel) for r in control_catalogue(hold=10)]
    return assess_grid(results, edge_panel, count_trials=False)


# --------------------------------------------------------------------------
# 1. Negative control -- noise is never PROVEN
# --------------------------------------------------------------------------

def test_random_walk_is_never_proven(noise_assessments):
    proven = [a.rule_id for a in noise_assessments if a.verdict is Verdict.PROVEN]
    assert not proven, (
        f"graded PROVEN on pure random-walk data with no edge: {proven}. "
        f"The harness is manufacturing signal from noise."
    )


def test_random_walk_mostly_no_edge(noise_assessments):
    """Not just 'never PROVEN' -- the bulk should land on NO_EDGE, not LIKELY.
    A harness that hedges everything to LIKELY is not discriminating."""
    verdicts = [a.verdict for a in noise_assessments]
    no_edge = sum(v is Verdict.NO_EDGE for v in verdicts)
    assert no_edge >= 0.6 * len(verdicts), (
        f"only {no_edge}/{len(verdicts)} rules graded NO_EDGE on noise: {verdicts}"
    )


# --------------------------------------------------------------------------
# 2. Positive control -- a real edge is detected
# --------------------------------------------------------------------------

def test_injected_edge_is_detected(edge_assessments):
    """The harness must not be merely always-negative."""
    detected = [
        a for a in edge_assessments
        if a.verdict in (Verdict.PROVEN, Verdict.LIKELY)
    ]
    assert detected, (
        "no rule reached LIKELY or PROVEN on data containing a large, genuine, "
        "mechanically-injected edge. The harness cannot detect anything."
    )


def test_injected_edge_beats_random_entry(edge_assessments):
    """dip_below_mean is the rule the edge was injected for; it must clearly
    beat the randomized-entry null."""
    best = max(
        (a for a in edge_assessments if np.isfinite(a.perm_pctile)),
        key=lambda a: a.perm_pctile,
    )
    assert best.perm_pctile >= 0.95, (
        f"best rule ({best.rule_id}) beat only {best.perm_pctile:.1%} of "
        f"random-entry shuffles despite a real injected edge"
    )


# --------------------------------------------------------------------------
# 3. Lookahead is structurally impossible
# --------------------------------------------------------------------------

def test_window_cannot_reach_the_future(noise_panel):
    w = noise_panel.window("S0", 200)
    assert len(w.c) == 201
    with pytest.raises(IndexError):
        _ = w.c[201]


def test_window_arrays_are_immutable(noise_panel):
    w = noise_panel.window("S0", 200)
    with pytest.raises(ValueError):
        w.c[0] = 1.0


def test_peeking_rule_raises(noise_panel):
    """A rule that tries to read tomorrow cannot silently succeed."""
    rule = Rule("peek", cr.peeking_rule, 30, {"hold": 5}, "", 5)
    eng = ExecutionEngine()
    with pytest.raises(IndexError):
        eng.run(rule, noise_panel, symbols=["S0"], audit=False)


def test_all_registered_indicators_are_causal(noise_panel):
    o, h, l, c = (noise_panel.col(f, "S0") for f in ("o", "h", "l", "c"))
    bars = Bars(o, h, l, c, np.zeros_like(c))
    for name, fn in ind.INDICATOR_REGISTRY.items():
        ind.assert_causal(fn, bars, n_probes=40, seed=1)


def test_causality_prober_catches_a_broken_indicator(noise_panel):
    """The prober must be capable of failing, or its passes mean nothing."""
    o, h, l, c = (noise_panel.col(f, "S0") for f in ("o", "h", "l", "c"))
    bars = Bars(o, h, l, c, np.zeros_like(c))
    with pytest.raises(AssertionError):
        ind.assert_causal(cr.noncausal_zscore, bars, n_probes=40, seed=1)


def test_fills_are_always_after_signals(noise_assessments, noise_panel):
    eng = ExecutionEngine()
    res = eng.run(control_catalogue(hold=10)[0], noise_panel)
    for t in res.trades:
        assert t.entry_idx > t.signal_idx
        assert t.exit_idx > t.entry_idx


def test_causality_audit_passes_on_clean_run(noise_panel):
    eng = ExecutionEngine()
    res = eng.run(control_catalogue(hold=10)[0], noise_panel)
    assert res.causality_verified is True, res.causality_problems


# --------------------------------------------------------------------------
# 4. Costs
# --------------------------------------------------------------------------

def test_costs_reduce_returns(noise_panel):
    eng = ExecutionEngine()
    for rule in control_catalogue(hold=10):
        res = eng.run(rule, noise_panel, audit=False)
        if len(res.trades) < 20:
            continue
        assert res.gross_returns.mean() > res.net_returns.mean(), (
            f"{rule.rule_id}: costs did not reduce the mean return"
        )
        return
    pytest.fail("no rule produced enough trades to test costs")


def test_no_proven_without_causality_audit(noise_panel):
    """Skipping the audit must make PROVEN unreachable, not merely unverified."""
    eng = ExecutionEngine()
    results = [eng.run(r, noise_panel, audit=False) for r in control_catalogue(hold=10)]
    ass = assess_grid(results, noise_panel, count_trials=False)
    assert all(a.verdict is not Verdict.PROVEN for a in ass)


# --------------------------------------------------------------------------
# 5. Reproducibility
# --------------------------------------------------------------------------

def test_results_are_reproducible(noise_panel):
    eng = ExecutionEngine()
    rule = control_catalogue(hold=10)[0]
    a = assess_grid([eng.run(rule, noise_panel)], noise_panel, count_trials=False)[0]
    b = assess_grid([eng.run(rule, noise_panel)], noise_panel, count_trials=False)[0]
    assert a.mean_net == b.mean_net
    assert a.perm_pctile == b.perm_pctile
    assert a.dsr == b.dsr


def test_config_hash_is_stable(noise_panel):
    eng = ExecutionEngine()
    rule = control_catalogue(hold=10)[0]
    assert (eng.run(rule, noise_panel, audit=False).config_hash
            == eng.run(rule, noise_panel, audit=False).config_hash)


# --------------------------------------------------------------------------
# 6. Statistics unit checks
# --------------------------------------------------------------------------

def test_deflated_sharpe_hurdle_rises_with_trials():
    """More trials must make the hurdle harder, or the anti-fishing mechanism
    does nothing."""
    h10 = st.expected_max_sharpe(0.25, 10)
    h1000 = st.expected_max_sharpe(0.25, 1000)
    assert h1000 > h10 > 0


def test_deflated_sharpe_penalises_fishing():
    rng = np.random.default_rng(0)
    r = rng.normal(0.004, 0.02, 200)
    few, _ = st.deflated_sharpe(r, sr_variance=0.25, n_trials=2)
    many, _ = st.deflated_sharpe(r, sr_variance=0.25, n_trials=5000)
    assert few > many


def test_benjamini_hochberg_controls_family():
    p = np.array([0.001, 0.2, 0.5, 0.9, 0.04])
    rej = st.benjamini_hochberg(p, alpha=0.05)
    assert rej[0] and not rej[3]


def test_effective_sample_size_shrinks_with_correlation():
    assert st.effective_sample_size(100, 10, 0.0) == pytest.approx(100)
    assert st.effective_sample_size(100, 10, 0.9) < 15


def test_block_bootstrap_ci_contains_mean():
    rng = np.random.default_rng(3)
    r = rng.normal(0.001, 0.02, 400)
    b = st.block_bootstrap(r, n_boot=500, seed=1)
    assert b.mean_lo < b.mean < b.mean_hi


def test_purged_kfold_excludes_overlapping_labels():
    entry = np.arange(0, 200, 2)
    exit_ = entry + 10
    folds = st.purged_kfold(entry, exit_, n_splits=4)
    assert folds
    for train, test in folds:
        assert not set(train) & set(test)
        for te in test:
            for tr in train:
                overlap = not (exit_[tr] < entry[te] or entry[tr] > exit_[te])
                assert not overlap, "a training label overlaps a test label"
