"""Turning statistics into a verdict, and rendering it honestly.

The decision rules here are deliberately hard to pass. Every threshold is
surfaced in the report so the user can disagree with a specific number rather
than with a black box.

The one rule that is never relaxed: a result cannot be graded PROVEN unless the
causality audit passed. An unverified strategy is not a strategy.
"""

from __future__ import annotations

import datetime as _dt
import json
from dataclasses import asdict, dataclass, field
from pathlib import Path

import numpy as np

from . import config as cfg
from . import stats as st
from .costs import CostModel
from .engine import buy_and_hold_benchmark, random_entry_null
from .panel import Panel
from .types import RunResult, Verdict


# --------------------------------------------------------------------------
# Persistent trial counter
# --------------------------------------------------------------------------

def load_trials(path: Path | None = None) -> dict:
    p = Path(path or cfg.TRIALS_FILE)
    if not p.exists():
        return {"total_trials": 0, "history": []}
    try:
        return json.loads(p.read_text())
    except Exception:
        return {"total_trials": 0, "history": []}


def strategy_fingerprint(rule_id: str, params: dict) -> str:
    """Identity of a strategy for trial-counting: its id plus its parameters.

    Changing a threshold IS a new strategy -- searching RSI<30 and then RSI<20 is
    two attempts at the same idea and the hurdle must reflect that. Re-running
    an identical rule is not.
    """
    flat = ",".join(f"{k}={params[k]!r}" for k in sorted(params))
    return f"{rule_id}({flat})"


def record_trials(
    fingerprints: list[str] | int, note: str = "", path: Path | None = None
) -> dict:
    """Record which DISTINCT strategies have been tried, ever.

    This is the anti-fishing mechanism: the deflated Sharpe hurdle rises with the
    number of strategies attempted, so grinding through a hundred variants makes
    a PROVEN verdict progressively harder to reach. That is intended.

    What is NOT intended is counting re-runs: evaluating the same 13 rules five
    times is 13 strategies tried, not 65. Counting re-runs would inflate the
    hurdle for no epistemic reason and punish you for re-running a report, so
    trials are a SET keyed by rule id and parameters.
    """
    p = Path(path or cfg.TRIALS_FILE)
    p.parent.mkdir(parents=True, exist_ok=True)
    d = load_trials(p)

    known = set(d.get("strategies", []))
    if isinstance(fingerprints, int):
        # A bare count, for callers with no strategy identities: count it as-is.
        d["untracked_count"] = int(d.get("untracked_count", 0)) + fingerprints
        added = fingerprints
    else:
        before = len(known)
        known.update(fingerprints)
        added = len(known) - before
        d["strategies"] = sorted(known)

    d["total_trials"] = max(len(known) + int(d.get("untracked_count", 0)), 1)
    d.setdefault("history", []).append({
        "when": _dt.datetime.now().isoformat(timespec="seconds"),
        "n_new": int(added),
        "total": d["total_trials"],
        "note": note,
    })
    d["history"] = d["history"][-500:]
    p.write_text(json.dumps(d, indent=2))
    return d


# --------------------------------------------------------------------------
# Assessment
# --------------------------------------------------------------------------

@dataclass
class Assessment:
    rule_id: str
    verdict: Verdict
    n_trades: int
    n_symbols: int
    n_eff: float
    mean_net: float
    mean_gross: float
    median_net: float
    win_rate: float
    sharpe_net: float
    sharpe_ci: tuple[float, float]
    mean_ci: tuple[float, float]
    dsr: float
    dsr_hurdle: float
    n_trials: int
    perm_pctile: float
    perm_p: float
    perm_null_mean: float
    buy_hold_mean: float
    regime_means: dict[str, float]
    regime_counts: dict[str, int]
    cv_folds_positive: float
    n_cv_folds: int
    cost_drag: float
    causality_verified: bool | None
    n_censored: int
    n_unfilled: int
    fdr_reject: bool = False
    reasons: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)
    sr_var: float = float("nan")
    sr_var_source: str = ""


def _regime_trend(label: str) -> str:
    return str(label).split("/")[0] if label else "unknown"


def assess_grid(
    results: list[RunResult],
    panel: Panel,
    settings: cfg.Settings = cfg.SETTINGS,
    seed: int = cfg.MASTER_SEED,
    count_trials: bool = True,
    costs: CostModel | None = None,
) -> list[Assessment]:
    """Assess every rule in a grid jointly.

    Joint assessment is not a convenience -- it is required for correctness. The
    deflated Sharpe hurdle depends on how many rules were tried and how their
    Sharpes were distributed, and FDR control operates on the whole family of
    p-values at once. Assessing one rule in isolation would understate both.

    `costs` must be the same CostModel the engine ran with, so the random-entry
    null pays exactly what the strategy paid. A null charged less than the
    strategy flatters the null; charged more, it flatters the strategy.
    """
    s = settings.stats

    # Only rules with a usable sample contribute to the trial-Sharpe variance.
    # A Sharpe estimated from four trades is noise, and letting it into this
    # variance inflates the deflated-Sharpe hurdle enough to bury a genuine
    # edge -- which is how an over-conservative harness becomes a useless one.
    live = [r for r in results if r.net_returns.size >= s.min_trades_hard]
    sharpes = np.array([st.sharpe(r.net_returns) for r in live], dtype=float)
    sharpes = sharpes[np.isfinite(sharpes)]
    if sharpes.size > 1:
        sr_var = float(np.var(sharpes, ddof=1))
    elif sharpes.size == 1:
        # One live rule is no family, so there is no variance across the family
        # to take. The obvious fallback, sr**2, sets the hurdle to ~2.2*|sr| --
        # above the observed Sharpe BY CONSTRUCTION, whatever it is, so a
        # single-rule run could never be graded PROVEN for a reason that has
        # nothing to do with the rule. See tests/test_deflated_sharpe.py.
        #
        # None means "derive it from this rule's own permutation null", which
        # _assess_one does: every replicate there is a complete no-edge history
        # of the same length, which is what the deflation formula is asking for.
        sr_var = None
    else:
        sr_var = 0.0

    fps = [strategy_fingerprint(r.rule_id, dict(r.params)) for r in results]
    if count_trials:
        trials_state = record_trials(fps, note=f"grid of {len(results)} rules")
    else:
        # Still count this grid for THIS run's hurdle -- not counting the rules
        # you are currently testing would understate the search that produced
        # them. It just is not persisted.
        trials_state = load_trials()
        known = set(trials_state.get("strategies", [])) | set(fps)
        trials_state = {**trials_state,
                        "total_trials": len(known) + int(trials_state.get("untracked_count", 0))}
    n_trials = max(int(trials_state.get("total_trials", 0)), len(results), 1)

    assessments: list[Assessment] = []
    for r in results:
        assessments.append(
            _assess_one(r, panel, sr_var, n_trials, settings, seed, costs)
        )

    # FDR across the family of permutation p-values.
    pvals = np.array([a.perm_p if np.isfinite(a.perm_p) else 1.0 for a in assessments])
    rejects = st.benjamini_hochberg(pvals, s.fdr_alpha)
    for a, rej in zip(assessments, rejects):
        a.fdr_reject = bool(rej)
        if not rej and a.verdict is Verdict.PROVEN:
            a.verdict = Verdict.LIKELY
            a.reasons.append(
                f"downgraded: fails Benjamini-Hochberg FDR control at alpha={s.fdr_alpha} "
                f"across the {len(assessments)}-rule family"
            )
    return assessments


def _assess_one(
    r: RunResult,
    panel: Panel,
    sr_var: float | None,
    n_trials: int,
    settings: cfg.Settings,
    seed: int,
    costs: CostModel | None = None,
) -> Assessment:
    s = settings.stats
    net = r.net_returns
    gross = r.gross_returns
    n = net.size
    reasons: list[str] = []
    warnings = list(r.warnings)

    if r.n_censored:
        warnings.append(
            f"{r.n_censored} trades censored (exit fell past the end of the data) "
            f"and excluded -- they were NOT closed at the last price"
        )
    if r.n_unfilled:
        warnings.append(f"{r.n_unfilled} signals could not be filled within the delay limit")
    if r.causality_verified is False:
        warnings.append(
            "CAUSALITY AUDIT FAILED -- a lookahead path was found; every number "
            "below is untrustworthy: " + "; ".join(r.causality_problems[:3])
        )

    def empty(verdict: Verdict, why: str) -> Assessment:
        reasons.append(why)
        return Assessment(
            rule_id=r.rule_id, verdict=verdict, n_trades=n,
            n_symbols=len({t.symbol for t in r.trades}), n_eff=0.0,
            mean_net=float("nan"), mean_gross=float("nan"), median_net=float("nan"),
            win_rate=float("nan"), sharpe_net=float("nan"),
            sharpe_ci=(float("nan"), float("nan")), mean_ci=(float("nan"), float("nan")),
            dsr=float("nan"), dsr_hurdle=float("nan"), n_trials=n_trials,
            perm_pctile=float("nan"), perm_p=float("nan"), perm_null_mean=float("nan"),
            buy_hold_mean=float("nan"), regime_means={}, regime_counts={},
            cv_folds_positive=float("nan"), n_cv_folds=0, cost_drag=float("nan"),
            causality_verified=r.causality_verified, n_censored=r.n_censored, n_unfilled=r.n_unfilled,
            reasons=reasons, warnings=warnings,
        )

    if n < s.min_trades_hard:
        return empty(
            Verdict.INSUFFICIENT_SAMPLE,
            f"only {n} completed trades; {s.min_trades_hard} is the minimum at which "
            f"anything can be concluded",
        )

    boot = st.block_bootstrap(
        net, n_boot=s.n_bootstrap,
        block_len=max(1.0, float(np.mean([t.bars_held for t in r.trades]))),
        ci=s.bootstrap_ci, seed=seed,
    )

    want_paths = sr_var is None
    null_out = random_entry_null(
        r, panel, costs=costs or CostModel(settings.costs), n_perm=s.n_permutations, seed=seed + 1,
        return_paths=want_paths,
    )
    null, null_paths = null_out if want_paths else (null_out, None)
    if null.size:
        perm_pctile = float(np.mean(null < net.mean()))
        perm_p = float((np.sum(null >= net.mean()) + 1) / (null.size + 1))
        perm_null_mean = float(np.mean(null))
    else:
        perm_pctile = perm_p = perm_null_mean = float("nan")

    if not want_paths:
        sr_var_source = "variance across the live rules in this grid"
    else:
        sr_var = st.trial_sharpe_variance(null_paths)
        sr_var_source = f"variance across {s.n_permutations} random-entry replicates"
        if not np.isfinite(sr_var):
            # The null could not be built (no tradable candidates, degenerate
            # replicates). Fall back to the asymptotic no-edge variance of the
            # Sharpe estimator, 1/n -- which is what the null converges to. Not
            # sr**2: a hurdle derived from the observation cannot judge it.
            sr_var = 1.0 / max(n, 2)
            sr_var_source = f"asymptotic 1/n at n={n} (the null gave no estimate)"

    dsr, hurdle = st.deflated_sharpe(net, sr_var, n_trials)

    # Effective sample size: how correlated are the concurrent positions?
    syms = sorted({t.symbol for t in r.trades})
    rho = _mean_pairwise_corr(panel, syms)
    m_bar = _mean_concurrency(r)
    n_eff = st.effective_sample_size(n, m_bar, rho)

    regime_means: dict[str, float] = {}
    regime_counts: dict[str, int] = {}
    for trend in ("bull", "bear"):
        vals = [t.net_ret for t in r.trades if _regime_trend(t.regime) == trend]
        regime_counts[trend] = len(vals)
        regime_means[trend] = float(np.mean(vals)) if vals else float("nan")

    folds = st.purged_kfold(
        np.array([t.entry_idx for t in r.trades]),
        np.array([t.exit_idx for t in r.trades]),
        n_splits=5,
    )
    fold_pos = [float(np.mean(net[te])) > 0 for _, te in folds] if folds else []
    cv_positive = float(np.mean(fold_pos)) if fold_pos else float("nan")

    hold = int(np.median([t.bars_held for t in r.trades]))
    bh = buy_and_hold_benchmark(panel, syms, max(1, hold))

    a = Assessment(
        rule_id=r.rule_id,
        verdict=Verdict.NO_EDGE,
        n_trades=n,
        n_symbols=len(syms),
        n_eff=n_eff,
        mean_net=float(net.mean()),
        mean_gross=float(gross.mean()),
        median_net=float(np.median(net)),
        win_rate=float(np.mean(net > 0)),
        sharpe_net=boot.sharpe,
        sharpe_ci=(boot.sharpe_lo, boot.sharpe_hi),
        mean_ci=(boot.mean_lo, boot.mean_hi),
        dsr=dsr, dsr_hurdle=hurdle, n_trials=n_trials,
        perm_pctile=perm_pctile, perm_p=perm_p, perm_null_mean=perm_null_mean,
        buy_hold_mean=bh,
        regime_means=regime_means, regime_counts=regime_counts,
        cv_folds_positive=cv_positive, n_cv_folds=len(folds),
        cost_drag=float(gross.mean() - net.mean()),
        causality_verified=r.causality_verified,
        n_censored=r.n_censored, n_unfilled=r.n_unfilled,
        reasons=reasons, warnings=warnings,
        sr_var=float(sr_var), sr_var_source=sr_var_source,
    )

    _decide(a, settings)
    return a


def _decide(a: Assessment, settings: cfg.Settings) -> None:
    """Apply the verdict ladder. Each failed hurdle is recorded by name."""
    s = settings.stats
    fails: list[str] = []

    if a.mean_net <= 0:
        fails.append(f"mean net return is {a.mean_net:+.4%}, not positive")
    if not (a.mean_ci[0] > 0):
        fails.append(
            f"bootstrap {s.bootstrap_ci:.0%} CI on the mean includes zero "
            f"[{a.mean_ci[0]:+.4%}, {a.mean_ci[1]:+.4%}]"
        )
    if not (a.perm_pctile >= s.perm_percentile_proven):
        fails.append(
            f"beats only {a.perm_pctile:.1%} of random-entry runs "
            f"(need {s.perm_percentile_proven:.0%})"
        )
    if not (a.dsr >= s.dsr_threshold):
        fails.append(
            f"deflated Sharpe {a.dsr:.3f} below {s.dsr_threshold} "
            f"(hurdle SR {a.dsr_hurdle:.3f} from {a.n_trials} trials)"
        )
    if a.n_trades < s.min_trades_proven:
        fails.append(f"{a.n_trades} trades, below the {s.min_trades_proven} required")
    if a.n_symbols < s.min_symbols_proven:
        fails.append(f"only {a.n_symbols} symbols, below the {s.min_symbols_proven} required")
    if a.n_eff < s.min_eff_sample_proven:
        fails.append(
            f"effective sample size {a.n_eff:.0f} below {s.min_eff_sample_proven:.0f} "
            f"after correcting for correlated concurrent positions"
        )
    if s.require_positive_in_both_regimes:
        thin = [reg for reg in ("bull", "bear")
                if a.regime_counts.get(reg, 0) < s.min_trades_per_regime]
        if thin:
            # An unrun check is a failed check. Without this, a run with no
            # benchmark -- or one whose trades all fell in one regime -- would
            # pass a hurdle it never faced.
            fails.append(
                "regime check not applied: "
                + ", ".join(f"{a.regime_counts.get(r, 0)} {r} trades" for r in ("bull", "bear"))
                + f" (need {s.min_trades_per_regime} in each; for a CSV run, add --benchmark-csv)"
            )
        for reg in ("bull", "bear"):
            if reg not in thin and not (a.regime_means.get(reg, 0) > 0):
                fails.append(f"mean net return not positive in {reg} regime")
    if s.require_causality_audit and a.causality_verified is not True:
        fails.append(
            "causality audit did not pass (not run, or it found a lookahead path)"
        )

    a.reasons.extend(fails)

    if not fails:
        a.verdict = Verdict.PROVEN
        return

    likely = (
        a.mean_net > 0
        and np.isfinite(a.perm_pctile)
        and a.perm_pctile >= s.perm_percentile_likely
        and a.causality_verified is not False
    )
    a.verdict = Verdict.LIKELY if likely else Verdict.NO_EDGE


def _mean_pairwise_corr(panel: Panel, symbols: list[str]) -> float:
    if len(symbols) < 2:
        return 0.0
    cols = []
    for s in symbols[:40]:                 # cap for cost; 40 is plenty for an average
        c = panel.col("c", s)
        r = np.diff(np.log(np.where(c > 0, c, np.nan)))
        cols.append(r)
    m = np.vstack(cols)
    ok = np.isfinite(m).all(axis=0)
    if ok.sum() < 30:
        return 0.5                          # conservative default
    cm = np.corrcoef(m[:, ok])
    iu = np.triu_indices_from(cm, k=1)
    vals = cm[iu]
    vals = vals[np.isfinite(vals)]
    return float(np.clip(np.mean(vals), 0.0, 1.0)) if vals.size else 0.5


def _mean_concurrency(r: RunResult) -> float:
    if not r.trades:
        return 1.0
    lo = min(t.entry_idx for t in r.trades)
    hi = max(t.exit_idx for t in r.trades)
    if hi <= lo:
        return 1.0
    counts = np.zeros(hi - lo + 2, dtype=int)
    for t in r.trades:
        counts[t.entry_idx - lo] += 1
        counts[t.exit_idx - lo + 1] -= 1
    open_positions = np.cumsum(counts)[:-1]
    active = open_positions[open_positions > 0]
    return float(active.mean()) if active.size else 1.0


# --------------------------------------------------------------------------
# Rendering
# --------------------------------------------------------------------------

_ORDER = {Verdict.PROVEN: 0, Verdict.LIKELY: 1, Verdict.NO_EDGE: 2,
          Verdict.INSUFFICIENT_SAMPLE: 3}


def render_report(
    assessments: list[Assessment],
    universe: list[str],
    start: _dt.date,
    end: _dt.date,
    settings: cfg.Settings = cfg.SETTINGS,
) -> str:
    lines: list[str] = []
    A = lines.append

    A("=" * 78)
    A("EDGELAB -- SIGNAL EDGE VALIDATION REPORT")
    A("=" * 78)
    A(f"window      : {start} to {end}")
    A(f"universe    : {len(universe)} symbols ({', '.join(universe[:12])}"
      f"{', ...' if len(universe) > 12 else ''})")
    A(f"rules tested: {len(assessments)}")
    A(f"trials to date (feeds the deflated-Sharpe hurdle): "
      f"{assessments[0].n_trials if assessments else 0}")
    A("")

    A("VERDICTS")
    A("-" * 78)
    A(f"{'rule':<28}{'verdict':<22}{'n':>6}{'mean net':>11}{'vs random':>11}")
    A("-" * 78)
    for a in sorted(assessments, key=lambda x: (_ORDER[x.verdict], -(x.mean_net if np.isfinite(x.mean_net) else -9))):
        mean = f"{a.mean_net:+.3%}" if np.isfinite(a.mean_net) else "  --"
        pct = f"{a.perm_pctile:.1%}" if np.isfinite(a.perm_pctile) else "  --"
        A(f"{a.rule_id:<28}{a.verdict.value:<22}{a.n_trades:>6}{mean:>11}{pct:>11}")
    A("")

    A("DETAIL")
    A("-" * 78)
    for a in sorted(assessments, key=lambda x: (_ORDER[x.verdict], x.rule_id)):
        A(f"\n{a.rule_id}  ->  {a.verdict.value}")
        if a.n_trades == 0:
            A("  no trades generated")
            for w in a.reasons:
                A(f"  - {w}")
            continue
        A(f"  trades {a.n_trades} across {a.n_symbols} symbols   "
          f"effective sample {a.n_eff:.0f}")
        A(f"  mean net {a.mean_net:+.4%}   median {a.median_net:+.4%}   "
          f"win rate {a.win_rate:.1%}")
        A(f"  gross {a.mean_gross:+.4%} - costs {a.cost_drag:.4%} = net {a.mean_net:+.4%}")
        A(f"  Sharpe/trade {a.sharpe_net:+.3f}  CI [{a.sharpe_ci[0]:+.3f}, {a.sharpe_ci[1]:+.3f}]")
        A(f"  mean CI [{a.mean_ci[0]:+.4%}, {a.mean_ci[1]:+.4%}]")
        A(f"  random-entry null: strategy beats {a.perm_pctile:.1%} of shuffles, "
          f"p={a.perm_p:.4f}, null mean {a.perm_null_mean:+.4%}")
        A(f"  buy-and-hold same universe/horizon: {a.buy_hold_mean:+.4%}")
        A(f"  deflated Sharpe {a.dsr:.3f} (hurdle SR {a.dsr_hurdle:.3f} from {a.n_trials} trials)"
          f"   BH-FDR {'pass' if a.fdr_reject else 'fail'}")
        if a.sr_var_source:
            A(f"    trial-Sharpe sd {np.sqrt(a.sr_var):.3f}, {a.sr_var_source}")
        reg = "  ".join(
            f"{k}: {a.regime_means[k]:+.4%} (n={a.regime_counts[k]})"
            if np.isfinite(a.regime_means.get(k, float('nan'))) else f"{k}: --"
            for k in ("bull", "bear")
        )
        A(f"  by regime: {reg}")
        if a.n_cv_folds:
            A(f"  purged CV: {a.cv_folds_positive:.0%} of {a.n_cv_folds} test folds positive")
        A(f"  causality audit: "
          f"{'PASSED' if a.causality_verified else 'FAILED' if a.causality_verified is False else 'not run'}")
        if a.reasons:
            A("  why not PROVEN:")
            for w in a.reasons:
                A(f"    - {w}")
        for w in a.warnings:
            A(f"  ! {w}")

    A("")
    A("=" * 78)
    A("STANDING LIMITATIONS -- these apply to every number above")
    A("=" * 78)
    for w in _standing_warnings():
        A(f"  ! {w}")
    A("")
    A("A PROVEN verdict is evidence that a rule had an edge in this sample under")
    A("these cost assumptions. It is not a prediction, not a recommendation, and")
    A("not a guarantee that the edge persists. Edges decay as they are discovered.")
    return "\n".join(lines)


def _standing_warnings() -> list[str]:
    """Non-suppressible. There is no flag to turn these off, by design."""
    return [
        "SURVIVORSHIP BIAS: the universe is built from symbols that exist today. "
        "Companies that were delisted, acquired or went to zero are absent, which "
        "flatters every long strategy measured here.",
        "The universe was chosen by a human who already knows how these symbols "
        "performed. That is a selection effect no statistic below can correct.",
        "Costs are modelled, not incurred. Real fills are worse than modelled fills, "
        "especially in size and especially in stress.",
        "Daily bars only. Gaps through stops fill at the open, but intrabar path "
        "(whether a stop or a target was touched first inside one bar) is not modelled.",
        "Default costs assume liquid US large caps. Thinner markets need a larger "
        "half-spread (--half-spread-bps), or every result here is flattered.",
        "Past edge is not future edge. Published anomalies decay after publication.",
    ]
