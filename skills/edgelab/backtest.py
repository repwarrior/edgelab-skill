#!/usr/bin/env python3
"""edgelab command line: does this trading rule actually have an edge?

    python3 backtest.py run --rules my_rules.py --symbols AAPL MSFT JPM ... --years 12
    python3 backtest.py run --rules my_rules.py --csv data/BTC.csv --half-spread-bps 5
    python3 backtest.py selftest        # prove the harness on synthetic data first
    python3 backtest.py trials          # how many strategies you have tried, ever
    python3 backtest.py probe           # check every built-in indicator is causal

A rules file is ordinary Python that defines RULES, a list of edgelab.Rule.
See rule_template.py next to this script.

Nothing here places an order, connects to a broker, or recommends a position.
"""

from __future__ import annotations

import argparse
import csv
import dataclasses
import datetime as _dt
import importlib.util
import json
import math
import sys
from pathlib import Path

import warnings

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

# yfinance on recent pandas prints pages of deprecation notices. They say
# nothing about the backtest and bury the report.
for _cat in (DeprecationWarning, FutureWarning, PendingDeprecationWarning):
    warnings.filterwarnings("ignore", category=_cat)

try:
    import numpy as np
    import edgelab as el
    from edgelab import config as cfg
    from edgelab import data as dat
    from edgelab import indicators as ind
except ModuleNotFoundError as exc:          # pragma: no cover - environment problem
    print(f"missing dependency: {exc.name}. Install with:\n"
          f"  python3 -m pip install -r {HERE / 'requirements.txt'}", file=sys.stderr)
    sys.exit(3)


# --------------------------------------------------------------------------
# Loading rules
# --------------------------------------------------------------------------

def load_rules(path: str) -> list[el.Rule]:
    p = Path(path).expanduser().resolve()
    if not p.exists():
        raise SystemExit(f"rules file not found: {p}")
    spec = importlib.util.spec_from_file_location(f"edgelab_user_rules_{p.stem}", p)
    mod = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(mod)
    rules = getattr(mod, "RULES", None)
    if callable(rules):
        rules = rules()
    if not rules:
        raise SystemExit(f"{p.name} must define RULES = [Rule(...), ...]")
    rules = list(rules)
    for r in rules:
        if not isinstance(r, el.Rule):
            raise SystemExit(f"{p.name}: RULES contains {type(r).__name__}, not edgelab.Rule")
    ids = [r.rule_id for r in rules]
    dupes = sorted({i for i in ids if ids.count(i) > 1})
    if dupes:
        raise SystemExit(f"{p.name}: duplicate rule ids {dupes}")
    return rules


# --------------------------------------------------------------------------
# Building the panel
# --------------------------------------------------------------------------

def _parse_date(s: str) -> _dt.date:
    return _dt.datetime.strptime(s, "%Y-%m-%d").date()


def _symbols(args) -> list[str]:
    syms = list(args.symbols or [])
    if args.symbols_file:
        for line in Path(args.symbols_file).expanduser().read_text().splitlines():
            line = line.split("#", 1)[0].strip()
            if line:
                syms.extend(line.replace(",", " ").split())
    return list(dict.fromkeys(s.upper() for s in syms))


def build_panel_from_args(args, quality: cfg.QualityThresholds):
    """Returns (panel, test_symbols, start, end, notes)."""
    notes: list[str] = []
    end = args.end or (_dt.date.today() - _dt.timedelta(days=1))
    start = args.start or (end - _dt.timedelta(days=int(365.25 * args.years)))

    if args.csv:
        series: dict[str, el.RawSeries] = {}
        masks: dict[str, np.ndarray] = {}
        for path in args.csv:
            try:
                s = dat.load_csv(path)
            except el.types.AdapterFetchError as exc:
                raise SystemExit(f"cannot read CSV: {exc}")
            if s.symbol in series:
                raise SystemExit(f"two CSV files are both named {s.symbol}: rename one "
                                 f"(the file name becomes the symbol)")
            keep = (s.dates >= np.datetime64(start, "D")) & (s.dates <= np.datetime64(end, "D"))
            s = dat._subset(s, keep)
            mask, rep = dat.quality_mask(s, quality)
            if rep.get("rejected") or len(s) == 0:
                print(f"  [drop] {s.symbol}: {rep.get('rejected') or 'no bars in window'}")
                continue
            _print_gate(rep)
            series[s.symbol] = s
            masks[s.symbol] = mask
        if not series:
            raise SystemExit("no usable CSV series")
        test_syms = sorted(series)
        calendar = dat.build_calendar(series)
        try:
            bench = dat.load_csv(args.benchmark_csv, "BENCHMARK") if args.benchmark_csv else None
        except el.types.AdapterFetchError as exc:
            raise SystemExit(f"cannot read benchmark CSV: {exc}")
        if bench is None:
            notes.append("no --benchmark-csv given: trades carry no market regime, so the "
                         "bull/bear check could not run and PROVEN is unreachable")
    else:
        test_syms = _symbols(args)
        if not test_syms:
            raise SystemExit("give --symbols, --symbols-file, or --csv")
        print(f"loading {len(test_syms)} symbols {start} -> {end} (yfinance, cached in "
              f"{cfg.OHLCV_DIR}) ...", flush=True)
        series, masks, reports = dat.load_universe(
            test_syms + [args.benchmark.upper()], start, end,
            force_refresh=args.refresh, quality=quality, verbose=True,
        )
        for rep in reports:
            if not rep.get("rejected") and rep.get("symbol") in test_syms:
                _print_gate(rep)
        dropped = [s for s in test_syms if s not in series]
        if dropped:
            notes.append(f"dropped by data checks: {', '.join(dropped)}")
        test_syms = [s for s in test_syms if s in series]
        if not test_syms:
            raise SystemExit("no usable symbols after data checks")
        calendar = dat.build_calendar(series)
        bench = series.get(args.benchmark.upper())
        if bench is None:
            notes.append(f"benchmark {args.benchmark} unavailable: bull/bear check could not "
                         f"run and PROVEN is unreachable")

    regime = (dat.compute_regime(bench, calendar) if bench is not None
              else np.full(len(calendar), "unknown", dtype=object))
    panel = el.build_panel(series.values(), calendar, masks, regime=regime)
    first = str(panel.dates[0])[:10] if panel.n_dates else str(start)
    last = str(panel.dates[-1])[:10] if panel.n_dates else str(end)
    return panel, test_syms, first, last, notes


def _print_gate(rep) -> None:
    flagged = {k: rep.get(k, 0) for k in ("n_malformed", "n_suspect_print", "n_stale", "n_penny")}
    if any(flagged.values()):
        bits = ", ".join(f"{v} {k[2:].replace('_', ' ')}" for k, v in flagged.items() if v)
        print(f"  [gate] {rep['symbol']}: {bits} bar(s) marked untradable")


# --------------------------------------------------------------------------
# Commands
# --------------------------------------------------------------------------

def _settings(args) -> cfg.Settings:
    base = cfg.SETTINGS
    costs = dataclasses.replace(
        base.costs,
        half_spread_bps=args.half_spread_bps,
        commission_bps=args.commission_bps,
    )
    engine = dataclasses.replace(
        base.engine, allow_shorts=not args.long_only, min_price=args.min_price,
    )
    quality = dataclasses.replace(base.quality, min_price=args.min_price)
    stats = base.stats
    if args.fast:
        stats = dataclasses.replace(stats, n_bootstrap=500, n_permutations=500)
    return dataclasses.replace(base, costs=costs, engine=engine, quality=quality, stats=stats)


def cmd_run(args) -> int:
    cfg.ensure_dirs()
    rules = load_rules(args.rules)
    if args.rule:
        wanted = set(args.rule)
        rules = [r for r in rules if r.rule_id in wanted]
        if not rules:
            raise SystemExit(f"no rules matched {sorted(wanted)}")

    settings = _settings(args)
    panel, syms, first, last, notes = build_panel_from_args(args, settings.quality)
    print(f"panel: {panel.n_dates} sessions x {len(syms)} test symbols, {first} -> {last}")

    costs = el.CostModel(settings.costs)
    engine = el.ExecutionEngine(costs=costs, conf=settings.engine)
    results = []
    for r in rules:
        print(f"  running {r.rule_id} ...", flush=True)
        results.append(engine.run(r, panel, symbols=syms, audit=not args.no_audit))

    print("assessing (bootstrap, random-entry null, deflated Sharpe) ...", flush=True)
    assessments = el.assess_grid(
        results, panel, settings=settings, seed=args.seed,
        count_trials=not args.no_count_trials, costs=costs,
    )
    for a in assessments:
        a.warnings.extend(notes)
    if args.no_count_trials:
        for a in assessments:
            a.warnings.append("run with --no-count-trials: this search was NOT added to the "
                              "permanent trial counter")

    report = el.render_report(assessments, syms, first, last, settings)
    cost_line = (f"costs: {settings.costs.commission_bps} bps commission + "
                 f"{settings.costs.half_spread_bps} bps half-spread per side, "
                 f"round-trip floor {settings.costs.min_round_trip_bps} bps"
                 f"{'; long-only' if args.long_only else ''}")
    report = report.replace("\nVERDICTS", f"{cost_line}\n\nVERDICTS", 1)
    print()
    print(report)

    if args.out:
        Path(args.out).write_text(report + "\n")
        print(f"\nreport written to {args.out}")
    if args.json:
        Path(args.json).write_text(json.dumps(
            [_jsonable(dataclasses.asdict(a)) for a in assessments], indent=2))
        print(f"assessments written to {args.json}")
    if args.trades:
        _write_trades(results, args.trades)
        print(f"trades written to {args.trades}")
    return 0


def _jsonable(x):
    if isinstance(x, dict):
        return {k: _jsonable(v) for k, v in x.items()}
    if isinstance(x, (list, tuple)):
        return [_jsonable(v) for v in x]
    if isinstance(x, el.Verdict):
        return x.value
    if isinstance(x, (np.floating, float)):
        f = float(x)
        return f if math.isfinite(f) else None
    if isinstance(x, np.integer):
        return int(x)
    if isinstance(x, np.bool_):
        return bool(x)
    return x


def _write_trades(results, path: str) -> None:
    fields = [f.name for f in dataclasses.fields(el.Trade)]
    with open(path, "w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["rule_id"] + fields)
        for r in results:
            for t in r.trades:
                w.writerow([r.rule_id] + [getattr(t, f) for f in fields])


def cmd_trials(args) -> int:
    d = el.load_trials()
    print(f"distinct strategies tried, ever: {d.get('total_trials', 0)}   ({cfg.TRIALS_FILE})")
    print("Every one of them raises the deflated-Sharpe hurdle for the next rule you test.")
    for h in d.get("history", [])[-15:]:
        print(f"  {h['when']}  +{h.get('n_new', 0):<4} total {h.get('total', '?'):<5} {h.get('note', '')}")
    return 0


def cmd_probe(args) -> int:
    rng = np.random.default_rng(args.seed)
    n = 800
    c = 100 * np.exp(np.cumsum(rng.normal(0, 0.012, n)))
    h = c * (1 + np.abs(rng.normal(0, 0.006, n)))
    l = c * (1 - np.abs(rng.normal(0, 0.006, n)))
    o = l + (h - l) * rng.random(n)
    res = ind.probe_all(el.Bars(o, h, l, c, np.ones(n)), seed=args.seed)
    bad = 0
    for name, status in res.items():
        print(f"  {name:<16} {status}")
        bad += status != "causal"
    print(f"\n{len(res)} indicators, {bad} non-causal")
    return 1 if bad else 0


def cmd_selftest(args) -> int:
    """The two controls that keep the harness honest, run on synthetic data:
    pure noise must never be PROVEN, and a mechanically injected edge must be
    found. Either alone is trivially passed by a broken harness."""
    from edgelab.types import SignalDecision

    def synth(sym, seed, edge):
        rng = np.random.default_rng(seed)
        n = 1500
        r = rng.normal(0.0003, 0.015, n)
        if edge:
            c0 = 100 * np.exp(np.cumsum(r))
            for t in range(30, n - 12):
                look = c0[t - 14: t + 1]
                if look[-1] < look.mean() * 0.96:
                    r[t + 1: t + 11] += edge
        c = 100 * np.exp(np.cumsum(r))
        hi = c * (1 + np.abs(rng.normal(0, 0.004, n)))
        lo = c * (1 - np.abs(rng.normal(0, 0.004, n)))
        o = lo + (hi - lo) * rng.random(n)
        d = np.datetime64("2010-01-01", "D") + np.arange(n).astype("timedelta64[D]")
        return el.RawSeries(sym, d, o, hi, lo, c, np.full(n, 2e6), np.zeros(n), np.zeros(n))

    def dip(w, p):
        if len(w.c) < 15:
            return None
        look = w.c[-15:]
        return SignalDecision(1, p["hold"]) if look[-1] < look.mean() * 0.96 else None

    rule = el.Rule("dip_below_15bar_mean", dip, 30, {"hold": 10}, hold_bars=10)
    ok = True
    for label, edge, want in (("pure noise", 0.0, "not PROVEN"), ("injected edge", 0.004, "PROVEN or LIKELY")):
        series = [synth(f"S{i}", 100 * (edge > 0) + i, edge) for i in range(8)]
        n = len(series[0].dates)
        regime = np.array(["bull/calm"] * (n // 2) + ["bear/stressed"] * (n - n // 2), dtype=object)
        panel = el.build_panel(series, series[0].dates, regime=regime)
        res = el.ExecutionEngine().run(rule, panel)
        a = el.assess_grid([res], panel, count_trials=False)[0]
        passed = (a.verdict is not el.Verdict.PROVEN) if edge == 0 else (
            a.verdict in (el.Verdict.PROVEN, el.Verdict.LIKELY))
        ok &= passed
        print(f"  {label:<14} -> {a.verdict.value:<20} (want {want}; beats "
              f"{a.perm_pctile:.1%} of random entries, {a.n_trades} trades)  "
              f"{'ok' if passed else 'FAIL'}")
    print("\nselftest passed" if ok else "\nSELFTEST FAILED -- do not trust any verdict from this install")
    return 0 if ok else 1


# --------------------------------------------------------------------------

def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(prog="edgelab", description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = p.add_subparsers(dest="cmd", required=True)

    r = sub.add_parser("run", help="backtest the rules in a rules file")
    r.add_argument("--rules", required=True, help="Python file defining RULES")
    r.add_argument("--rule", nargs="+", help="only these rule ids from the file")
    src = r.add_argument_group("data (pick one)")
    src.add_argument("--symbols", nargs="+", help="tickers, fetched from Yahoo Finance")
    src.add_argument("--symbols-file", help="file of tickers, one per line or comma-separated")
    src.add_argument("--csv", nargs="+", help="your own daily-bar CSV file(s), one symbol each")
    r.add_argument("--benchmark", default="SPY", help="regime benchmark ticker (default SPY)")
    r.add_argument("--benchmark-csv", help="regime benchmark as a CSV, for --csv runs")
    r.add_argument("--start", type=_parse_date, help="YYYY-MM-DD")
    r.add_argument("--end", type=_parse_date, help="YYYY-MM-DD (default yesterday)")
    r.add_argument("--years", type=float, default=12.0, help="history length if no --start")
    r.add_argument("--half-spread-bps", type=float, default=1.0,
                   help="per-side half spread in bps (default 1.0: liquid US large caps)")
    r.add_argument("--commission-bps", type=float, default=0.5, help="per side (default 0.5)")
    r.add_argument("--min-price", type=float, default=5.0,
                   help="bars below this price are untradable (default 5; lower for FX/crypto)")
    r.add_argument("--long-only", action="store_true", help="ignore short signals")
    r.add_argument("--no-count-trials", action="store_true",
                   help="do not add this run to the permanent trial counter")
    r.add_argument("--no-audit", action="store_true", help="skip the causality audit "
                   "(makes PROVEN unreachable)")
    r.add_argument("--fast", action="store_true", help="500 instead of 2000 resamples")
    r.add_argument("--refresh", action="store_true", help="re-download cached prices")
    r.add_argument("--seed", type=int, default=cfg.MASTER_SEED)
    r.add_argument("--out", help="also write the report to this file")
    r.add_argument("--json", help="write assessments as JSON")
    r.add_argument("--trades", help="write every trade as CSV")
    r.set_defaults(fn=cmd_run)

    t = sub.add_parser("trials", help="show the permanent trial counter")
    t.set_defaults(fn=cmd_trials)
    pr = sub.add_parser("probe", help="check built-in indicators are causal")
    pr.add_argument("--seed", type=int, default=0)
    pr.set_defaults(fn=cmd_probe)
    st = sub.add_parser("selftest", help="noise must fail, a real edge must pass")
    st.set_defaults(fn=cmd_selftest)

    args = p.parse_args(argv)
    if getattr(args, "csv", None) and (args.symbols or args.symbols_file):
        p.error("use --csv or --symbols/--symbols-file, not both")
    return args.fn(args)


if __name__ == "__main__":
    sys.exit(main())
