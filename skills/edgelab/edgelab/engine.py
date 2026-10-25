"""The execution engine.

Two properties are enforced structurally rather than by convention:

1. A fill index is always strictly greater than its signal index. There is an
   assertion on every fill. Entry is at the OPEN of a later bar, never at the
   close of the decision bar -- the single most common way a backtest invents
   returns it could not have earned.

2. An unfinished trade is never closed early at the last available price.
   Trades whose exit falls past the end of the data are marked censored and
   excluded from headline statistics. Closing them at the final bar is a direct
   upward bias in any rising market, and it is a very easy mistake to make.
"""

from __future__ import annotations

import hashlib
import json
from typing import Mapping

import numpy as np

from . import config as cfg
from .costs import CostModel, corwin_schultz_spread
from .indicators import INDICATOR_REGISTRY, assert_causal, atr, realized_vol
from .panel import Panel
from .types import Bars, PortfolioMode, Rule, RunResult, SignalDecision, Trade


def _first_stop_hit(
    o: np.ndarray, h: np.ndarray, l: np.ndarray, tradable: np.ndarray,
    entry_idx: int, exit_idx: int, stop_px: float, side: int,
) -> tuple[int, float, bool] | None:
    """First bar whose range breaches the stop, and the price actually filled.

    Returns (bar_index, fill_price, gapped_through) or None if never hit.

    Two decisions here are where a stop-loss backtest usually starts lying:

    1. THE EXIT BAR IS EXCLUDED. The scheduled exit fills at exit_idx's OPEN,
       which happens before that bar's high and low print. A stop "hit" later in
       that same bar is therefore not reachable -- the position is already
       closed. Scanning through exit_idx would book stop exits that could not
       have occurred, and because a stop is only hit on adverse moves, doing so
       biases results DOWNWARD here rather than upward. It is still wrong.

    2. GAPS FILL AT THE OPEN, NOT AT THE STOP. If a bar opens beyond the stop,
       the market never traded at the stop level -- the fill is the open, which
       is worse. Filling at the stop price is the classic way a backtest
       pretends stops cap losses precisely. They do not; that is gap risk, and
       it is exactly the risk a daily-bar model must not assume away.
    """
    for i in range(entry_idx, exit_idx):
        if not tradable[i]:
            continue
        oi, hi, li = o[i], h[i], l[i]
        if side == 1:
            if np.isfinite(li) and li <= stop_px:
                gapped = bool(np.isfinite(oi) and oi < stop_px)
                return i, (float(oi) if gapped else float(stop_px)), gapped
        else:
            if np.isfinite(hi) and hi >= stop_px:
                gapped = bool(np.isfinite(oi) and oi > stop_px)
                return i, (float(oi) if gapped else float(stop_px)), gapped
    return None


def _kth_tradable_after(tradable_col: np.ndarray, start_idx: int, k: int) -> int | None:
    """Index of the k-th tradable bar strictly after start_idx, or None."""
    found = 0
    for i in range(start_idx + 1, len(tradable_col)):
        if tradable_col[i]:
            found += 1
            if found == k:
                return i
    return None


def _next_tradable(tradable_col: np.ndarray, after: int, max_delay: int) -> int | None:
    for step in range(1, max_delay + 1):
        i = after + step
        if i >= len(tradable_col):
            return None
        if tradable_col[i]:
            return i
    return None


class ExecutionEngine:
    def __init__(
        self,
        costs: CostModel | None = None,
        conf: cfg.EngineConfig | None = None,
    ) -> None:
        self.costs = costs or CostModel()
        self.conf = conf or cfg.SETTINGS.engine

    # ------------------------------------------------------------------
    def run(
        self,
        rule: Rule,
        panel: Panel,
        symbols: list[str] | None = None,
        mode: PortfolioMode = PortfolioMode.NON_OVERLAPPING,
        audit: bool = True,
    ) -> RunResult:
        symbols = sorted(symbols or panel.symbols)
        trades, n_censored, n_unfilled, n_delisted, warnings = self._simulate(
            rule, panel, symbols
        )

        causality_verified: bool | None = None
        problems: list[str] = []
        if audit:
            causality_verified, problems = verify_causality(panel, symbols, trades)
            if not causality_verified:
                warnings.append("CAUSALITY AUDIT FAILED: " + "; ".join(problems))

        max_conc = _max_concurrent(trades)
        result = RunResult(
            rule_id=rule.rule_id,
            params=dict(rule.params),
            mode=mode,
            trades=trades,
            n_censored=n_censored,
            n_unfilled=n_unfilled,
            n_delisted=n_delisted,
            symbols=tuple(symbols),
            max_daily_positions=max_conc,
            warnings=warnings,
            causality_verified=causality_verified,
            causality_problems=problems,
        )
        result.config_hash = _config_hash(rule, symbols, mode, self.conf, self.costs)
        return result

    # ------------------------------------------------------------------
    def _simulate(
        self, rule: Rule, panel: Panel, symbols: list[str]
    ) -> tuple[list[Trade], int, int, int, list[str]]:
        source = panel
        trades: list[Trade] = []
        n_censored = n_unfilled = n_delisted = 0
        warnings: list[str] = []

        for sym in symbols:
            tradable = panel.tradable[:, panel._idx[sym]]
            o = panel.col("o", sym)
            c = panel.col("c", sym)
            h = panel.col("h", sym)
            l = panel.col("l", sym)
            div = panel.col("dividend", sym)
            n = panel.n_dates

            spread = corwin_schultz_spread(h, l, self.costs.c.spread_lookback)
            bars_all = Bars(o, h, l, c, np.zeros_like(c))
            vol = realized_vol(bars_all, 20)
            atr_arr = atr(bars_all, self.conf.stop_atr_period)

            start = panel.first_valid[sym] + rule.warmup_bars
            if start >= n - 2:
                continue

            busy_until = -1
            for t in range(start, n - 1):
                if t <= busy_until:
                    continue
                if not np.isfinite(c[t]) or not tradable[t]:
                    continue

                decision = rule.signal(source.window(sym, t), rule.params)
                if decision is None:
                    continue
                if decision.side == -1 and not self.conf.allow_shorts:
                    continue

                entry_idx = _next_tradable(tradable, t, self.conf.max_fill_delay_bars)
                if entry_idx is None:
                    n_unfilled += 1
                    continue
                assert entry_idx > t, "fill index must be strictly after the signal bar"

                exit_idx = _kth_tradable_after(tradable, entry_idx, decision.hold_bars)
                if exit_idx is None:
                    n_censored += 1
                    busy_until = n
                    continue

                entry_px = o[entry_idx]
                if not np.isfinite(entry_px) or entry_px <= 0:
                    n_unfilled += 1
                    continue

                # -- stop resolution ---------------------------------------
                # The stop distance is fixed at the SIGNAL bar (ATR is known
                # then); the level is anchored to the actual fill. Scanning
                # starts at the entry bar itself, because a position opened at
                # that bar's open is exposed to the rest of that bar.
                stop_px = None
                terminated = "scheduled"
                gapped = False
                slippage = 0.0
                if decision.has_stop:
                    dist = (
                        decision.stop_pct * entry_px if decision.stop_pct is not None
                        else decision.stop_atr * atr_arr[t]
                    )
                    if np.isfinite(dist) and dist > 0:
                        stop_px = (entry_px - dist if decision.side == 1
                                   else entry_px + dist)
                        hit = _first_stop_hit(
                            o, h, l, tradable, entry_idx, exit_idx,
                            stop_px, decision.side,
                        )
                        if hit is not None:
                            exit_idx, fill_px, gapped = hit
                            terminated = "stopped"
                            # Zero unless the bar gapped through: an ungapped
                            # stop fills AT stop_px by construction.
                            slippage = abs(fill_px - stop_px)

                exit_px = o[exit_idx] if terminated == "scheduled" else fill_px
                if not np.isfinite(exit_px):
                    n_unfilled += 1
                    continue

                # Dividend interval is half-open: (entry_date, exit_date].
                # Entry fills at the entry bar's OPEN, so you did not hold at the
                # prior close and do NOT receive a dividend going ex that day.
                # Exit fills at the exit bar's OPEN, so you DID hold through the
                # prior close and DO receive one going ex that day.
                d_cash = float(np.nansum(div[entry_idx + 1: exit_idx + 1]))

                side = decision.side
                gross_ret = side * ((exit_px - entry_px) + d_cash) / entry_px

                # Each leg is charged at its own day's volatility -- exactly what
                # random_entry_null charges, so the null and the strategy pay
                # identical costs for identical fills.
                hs_sym = self.costs.half_spread_for(sym)
                dvol = vol / np.sqrt(252.0)
                rt_bps = max(
                    self.costs.per_side_bps(spread[entry_idx], entry_px,
                                            float(dvol[entry_idx]), half_spread_bps=hs_sym)
                    + self.costs.per_side_bps(spread[exit_idx], exit_px,
                                              float(dvol[exit_idx]), half_spread_bps=hs_sym),
                    self.costs.c.min_round_trip_bps,
                )
                bars_held = exit_idx - entry_idx
                borrow = (
                    self.costs.c.borrow_bps_per_day * bars_held / 1e4 if side == -1 else 0.0
                )
                net_ret = gross_ret - rt_bps / 1e4 - borrow

                trades.append(Trade(
                    symbol=sym, side=side, tag=decision.tag,
                    signal_idx=t, entry_idx=entry_idx, exit_idx=exit_idx,
                    signal_date=panel.dates[t].astype("datetime64[D]").astype(object),
                    entry_date=panel.dates[entry_idx].astype("datetime64[D]").astype(object),
                    exit_date=panel.dates[exit_idx].astype("datetime64[D]").astype(object),
                    entry_px=float(entry_px), exit_px=float(exit_px),
                    dividends_cash=d_cash,
                    cost_cash=rt_bps / 1e4, borrow_cash=borrow,
                    gross_ret=float(gross_ret), net_ret=float(net_ret),
                    bars_held=int(bars_held),
                    fill_delay_bars=int(entry_idx - t),
                    terminated=terminated,
                    regime=str(panel.regime[entry_idx]),
                    stop_px=None if stop_px is None else float(stop_px),
                    gapped_through_stop=gapped,
                    slippage_px=float(slippage),
                ))
                busy_until = exit_idx      # NON_OVERLAPPING within a symbol

        return trades, n_censored, n_unfilled, n_delisted, warnings


def random_entry_null(
    result: RunResult,
    panel: Panel,
    costs: CostModel | None = None,
    n_perm: int = 2000,
    seed: int = 0,
    return_paths: bool = False,
) -> np.ndarray | tuple[np.ndarray, np.ndarray]:
    """Distribution of mean net return under randomized entry timing.

    This is the benchmark that matters. It holds constant everything about the
    strategy EXCEPT the thing being tested: the same symbols, the same number of
    trades, the same holding periods, the same cost model, the same market
    window -- but entry dates drawn at random.

    A long-only rule in a rising market makes money. The question is whether it
    makes more than throwing darts at the same calendar would have. Comparing to
    zero, or even to buy-and-hold, does not answer that. This does.

    Returns an array of `n_perm` mean net returns under the null. With
    `return_paths`, also returns the (n_perm, n_trades) matrix those means were
    collapsed from: each ROW is a complete no-edge history of the same length as
    the real one, which is the material the deflated-Sharpe hurdle needs and the
    collapsed mean cannot supply. See `stats.trial_sharpe_variance`.
    """
    costs = costs or CostModel()
    rng = np.random.default_rng(seed)
    trades = result.trades
    if not trades:
        empty = np.empty(0, dtype=float)
        return (empty, np.empty((0, 0), dtype=float)) if return_paths else empty

    null_sums = np.zeros(n_perm, dtype=float)
    # Materialised only on request: an (n_perm, n_trades) array is large, and
    # every caller that only wants the mean should not pay for it.
    paths = np.zeros((n_perm, len(trades)), dtype=float) if return_paths else None
    cache: dict[tuple[str, int], tuple] = {}

    for k, tr in enumerate(trades):
        # Hold length in TRADABLE bars, as the engine counted it. bars_held is a
        # positional distance, which overstates the hold whenever an untradable
        # bar fell inside the trade.
        j_sym = panel._idx[tr.symbol]
        n_trad = max(1, int(panel.tradable[tr.entry_idx + 1: tr.exit_idx + 1, j_sym].sum()))
        key = (tr.symbol, n_trad)
        if key not in cache:
            j = panel._idx[tr.symbol]
            tradable = panel.tradable[:, j]
            o = panel.col("o", tr.symbol)
            h = panel.col("h", tr.symbol)
            l = panel.col("l", tr.symbol)
            c = panel.col("c", tr.symbol)
            div = np.nan_to_num(panel.col("dividend", tr.symbol), nan=0.0)
            n = panel.n_dates

            spread = corwin_schultz_spread(h, l, costs.c.spread_lookback)
            vol = realized_vol(Bars(o, h, l, c, np.zeros_like(c)), 20) / np.sqrt(252.0)

            # exit_map[i] = index of the n_trad-th tradable bar after i
            exit_map = np.full(n, -1, dtype=int)
            trad_idx = np.flatnonzero(tradable)
            posn = np.searchsorted(trad_idx, np.arange(n), side="right")
            tgt = posn + n_trad - 1
            valid_tgt = tgt < trad_idx.size
            exit_map[valid_tgt] = trad_idx[tgt[valid_tgt]]

            hs_sym = costs.half_spread_for(tr.symbol)
            per_side = np.array([
                costs.per_side_bps(
                    spread[i], o[i] if np.isfinite(o[i]) and o[i] > 0 else 1.0,
                    vol[i] if np.isfinite(vol[i]) else np.nan,
                    half_spread_bps=hs_sym,
                )
                for i in range(n)
            ])

            candidates = np.flatnonzero(
                tradable & (exit_map > 0) & np.isfinite(o) & (o > 0)
            )
            candidates = candidates[np.isfinite(o[exit_map[candidates]])]
            cumdiv = np.concatenate([[0.0], np.nancumsum(div)])
            cache[key] = (o, exit_map, per_side, candidates, cumdiv)

        o, exit_map, per_side, candidates, cumdiv = cache[key]
        if candidates.size == 0:
            continue

        ent = rng.choice(candidates, size=n_perm, replace=True)
        ext = exit_map[ent]
        entry_px, exit_px = o[ent], o[ext]
        d_cash = cumdiv[ext + 1] - cumdiv[ent + 1]

        gross = tr.side * ((exit_px - entry_px) + d_cash) / entry_px
        rt = np.maximum(per_side[ent] + per_side[ext], costs.c.min_round_trip_bps)
        borrow = (
            costs.c.borrow_bps_per_day * (ext - ent) / 1e4 if tr.side == -1 else 0.0
        )
        leg = gross - rt / 1e4 - borrow
        null_sums += leg
        if paths is not None:
            paths[:, k] = leg

    means = null_sums / len(trades)
    return (means, paths) if return_paths else means


def buy_and_hold_benchmark(panel: Panel, symbols: list[str], hold_bars: int) -> float:
    """Mean net return of holding each symbol for `hold_bars` from every bar.

    Reported alongside the strategy so a rule that merely reproduces the drift
    of its own universe is visible as such.
    """
    rets: list[float] = []
    for sym in symbols:
        o = panel.col("o", sym)
        n = len(o)
        for i in range(0, n - hold_bars, max(1, hold_bars)):
            a, b = o[i], o[i + hold_bars]
            if np.isfinite(a) and np.isfinite(b) and a > 0:
                rets.append((b - a) / a)
    return float(np.mean(rets)) if rets else float("nan")


def verify_causality(
    panel: Panel, symbols: list[str], trades: list[Trade], n_probes: int = 40, seed: int = 0
) -> tuple[bool, list[str]]:
    """Audit that nothing in this run could have seen the future.

    The obvious check -- re-run the strategy on a panel whose future bars are
    NaN and compare ledgers -- is vacuous here: a rule only ever receives a
    window truncated at the decision bar, so the poisoned and normal windows are
    byte-identical by construction and the comparison can only produce false
    alarms.

    What is actually worth verifying, and is verified here:

      1. STRUCTURAL -- the window really is truncated, really does raise on an
         attempt to read past the cursor, and really is immutable.
      2. INDICATORS -- every registered indicator is causal on THIS panel's real
         data, not merely on a synthetic series. This is the check that catches
         a centered window, a full-sample normalization, or a wrong-direction
         shift.
      3. COST PRECOMPUTATION -- the engine computes spread and volatility over
         the whole series once, before the bar loop, for speed. That is only
         safe if those functions are causal, so they are probed the same way.
      4. LEDGER -- every fill is strictly after its signal, and every exit is
         strictly after its entry.
    """
    problems: list[str] = []

    # 1. Structural guarantees.
    sym = symbols[0] if symbols else panel.symbols[0]
    t = min(panel.first_valid[sym] + 50, panel.n_dates - 2)
    if t > 1:
        w = panel.window(sym, t)
        if len(w.c) != t + 1:
            problems.append(f"window length {len(w.c)} != t+1 ({t + 1})")
        try:
            _ = w.c[t + 1]
            problems.append("window exposed a bar beyond the decision cursor")
        except IndexError:
            pass
        try:
            w.c[0] = -1.0
            problems.append("window arrays are writeable (a rule could corrupt shared state)")
        except ValueError:
            pass

    # 2 & 3. Causality of every indicator, on this panel's actual prices.
    o, h, l, c = (panel.col(f, sym) for f in ("o", "h", "l", "c"))
    ok = np.isfinite(c)
    if ok.sum() > 300:
        real = Bars(o[ok], h[ok], l[ok], c[ok], np.zeros(int(ok.sum())))
        for name, fn in sorted(INDICATOR_REGISTRY.items()):
            try:
                assert_causal(fn, real, n_probes=n_probes, seed=seed)
            except AssertionError as exc:
                problems.append(f"indicator {name} not causal: {exc}")
        try:
            _assert_spread_causal(real, n_probes=min(12, n_probes), seed=seed)
        except AssertionError as exc:
            problems.append(str(exc))

    # 4. Ledger invariants.
    for tr in trades:
        if tr.entry_idx <= tr.signal_idx:
            problems.append(
                f"{tr.symbol}: fill at {tr.entry_idx} is not after signal at {tr.signal_idx}"
            )
            break
        if tr.exit_idx <= tr.entry_idx:
            problems.append(f"{tr.symbol}: exit {tr.exit_idx} not after entry {tr.entry_idx}")
            break

    return (not problems), problems


def _assert_spread_causal(bars: Bars, n_probes: int = 12, seed: int = 0) -> None:
    """The Corwin-Schultz estimator is precomputed over the full series, so it
    must not look forward. Probed the same way as the indicators."""
    full = corwin_schultz_spread(bars.h, bars.l)
    rng = np.random.default_rng(seed)
    n = len(bars)
    for t in rng.choice(np.arange(max(80, n // 3), n), size=min(n_probes, n // 3), replace=False):
        t = int(t)
        trunc = corwin_schultz_spread(bars.h[: t + 1], bars.l[: t + 1])[-1]
        ref = full[t]
        if np.isnan(trunc) and np.isnan(ref):
            continue
        if np.isnan(trunc) != np.isnan(ref) or abs(trunc - ref) > 1e-12:
            raise AssertionError(
                f"corwin_schultz_spread not causal at t={t}: {trunc!r} != {ref!r}"
            )


def _max_concurrent(trades: list[Trade]) -> int:
    """Peak simultaneous open positions across the whole run.

    Feeds the effective-sample-size calculation: trades that overlap in time are
    correlated through the market factor, so N overlapping trades are worth far
    fewer than N independent observations.
    """
    if not trades:
        return 0
    events: list[tuple[int, int]] = []
    for t in trades:
        events.append((t.entry_idx, 1))
        events.append((t.exit_idx, -1))
    events.sort()
    cur = peak = 0
    for _, delta in events:
        cur += delta
        peak = max(peak, cur)
    return peak


def _config_hash(rule, symbols, mode, engine_conf, costs) -> str:
    payload = {
        "rule": rule.rule_id,
        "params": {k: str(v) for k, v in sorted(dict(rule.params).items())},
        "warmup": rule.warmup_bars,
        "symbols": sorted(symbols),
        "mode": mode.value,
        "engine": {
            "max_fill_delay_bars": engine_conf.max_fill_delay_bars,
            "allow_shorts": engine_conf.allow_shorts,
            "min_price": engine_conf.min_price,
        },
        "costs": {
            "commission_bps": costs.c.commission_bps,
            "half_spread_by_symbol": sorted(costs.by_symbol.items()),
            "half_spread_bps": costs.c.half_spread_bps,
            "slippage_eta": costs.c.slippage_eta,
            "min_round_trip_bps": costs.c.min_round_trip_bps,
        },
    }
    return hashlib.sha256(
        json.dumps(payload, sort_keys=True, default=str).encode()
    ).hexdigest()[:16]
