"""The command line and the bring-your-own-data path."""
from __future__ import annotations

import importlib.util
import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from edgelab import data as dat
from edgelab.types import AdapterFetchError

SKILL = Path(__file__).resolve().parent.parent / "skills" / "edgelab"


def _cli():
    spec = importlib.util.spec_from_file_location("edgelab_cli", SKILL / "backtest.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def _write_csv(path: Path, n: int = 900, seed: int = 0, volume: bool = True, price: float = 100.0):
    rng = np.random.default_rng(seed)
    c = price * np.exp(np.cumsum(rng.normal(0.0002, 0.012, n)))
    h = c * (1 + np.abs(rng.normal(0, 0.004, n)))
    l = c * (1 - np.abs(rng.normal(0, 0.004, n)))
    o = l + (h - l) * rng.random(n)
    df = pd.DataFrame({"Date": pd.bdate_range("2019-01-01", periods=n),
                       "Open": o, "High": h, "Low": l, "Close": c,
                       "Adj Close": c * 0.9})
    if volume:
        df["Volume"] = 1_000_000
    df.to_csv(path, index=False)
    return path


def test_template_rules_load():
    rules = _cli().load_rules(str(SKILL / "rule_template.py"))
    assert rules and all(r.rule_id for r in rules)


def test_rules_file_without_rules_is_refused(tmp_path):
    f = tmp_path / "empty.py"
    f.write_text("x = 1\n")
    with pytest.raises(SystemExit):
        _cli().load_rules(str(f))


def test_csv_loader_reads_ohlc_and_ignores_adjusted_close(tmp_path):
    s = dat.load_csv(_write_csv(tmp_path / "abc.csv"))
    raw = pd.read_csv(tmp_path / "abc.csv")
    assert s.symbol == "ABC"
    np.testing.assert_allclose(s.c, raw["Close"].to_numpy())


def test_csv_without_volume_is_not_rejected_as_zero_volume(tmp_path):
    s = dat.load_csv(_write_csv(tmp_path / "fx.csv", volume=False, price=1.1))
    mask, rep = dat.quality_mask(s, dat.cfg.QualityThresholds(min_price=0.0))
    assert not rep.get("rejected")
    assert mask.mean() > 0.95


def test_intraday_csv_is_refused(tmp_path):
    p = tmp_path / "m1.csv"
    pd.DataFrame({"time": pd.date_range("2024-01-02 09:30", periods=50, freq="min"),
                  "open": 1.0, "high": 1.0, "low": 1.0, "close": 1.0}).to_csv(p, index=False)
    with pytest.raises(AdapterFetchError):
        dat.load_csv(p)


def test_run_on_csv_end_to_end(tmp_path, capsys):
    files = [str(_write_csv(tmp_path / f"s{i}.csv", seed=i)) for i in range(5)]
    out = tmp_path / "a.json"
    trades = tmp_path / "t.csv"
    rc = _cli().main(["run", "--rules", str(SKILL / "rule_template.py"), "--csv", *files,
                      "--start", "2019-01-01", "--end", "2023-12-31", "--fast",
                      "--no-count-trials", "--json", str(out), "--trades", str(trades)])
    assert rc == 0
    text = capsys.readouterr().out
    assert "VERDICTS" in text and "STANDING LIMITATIONS" in text
    assert "bull/bear check could not run" in text
    assert "regime check not applied" in text
    a = json.loads(out.read_text())
    assert a[0]["rule_id"] == "sma50_cross_up"
    assert a[0]["verdict"] != "PROVEN"           # random walks, no benchmark
    assert trades.read_text().startswith("rule_id,")


def test_selftest_passes():
    assert _cli().main(["selftest"]) == 0


def _edge_csvs(tmp_path, n_files=6, n=1500):
    """Random walks with an edge injected after dips below the 15-bar mean."""
    paths = []
    for i in range(n_files):
        rng = np.random.default_rng(200 + i)
        r = rng.normal(0.0003, 0.015, n)
        c0 = 100 * np.exp(np.cumsum(r))
        for t in range(30, n - 12):
            look = c0[t - 14: t + 1]
            if look[-1] < look.mean() * 0.96:
                r[t + 1: t + 11] += 0.004
        c = 100 * np.exp(np.cumsum(r))
        h = c * 1.004; l = c * 0.996; o = (h + l) / 2
        p = tmp_path / f"e{i}.csv"
        pd.DataFrame({"Date": pd.bdate_range("2012-01-02", periods=n), "Open": o,
                      "High": h, "Low": l, "Close": c, "Volume": 1e6}).to_csv(p, index=False)
        paths.append(str(p))
    rules = tmp_path / "dip.py"
    rules.write_text(
        "from edgelab import Rule, SignalDecision\n"
        "def dip(w, p):\n"
        "    if len(w.c) < 15: return None\n"
        "    look = w.c[-15:]\n"
        "    return SignalDecision(1, 10) if look[-1] < look.mean() * 0.96 else None\n"
        "RULES = [Rule('dip', dip, 30, {}, hold_bars=10)]\n")
    return paths, str(rules)


def test_no_benchmark_means_no_proven_even_with_a_real_edge(tmp_path):
    """An unrun regime check is a failed check, not a skipped hurdle."""
    files, rules = _edge_csvs(tmp_path)
    out = tmp_path / "a.json"
    _cli().main(["run", "--rules", rules, "--csv", *files, "--start", "2012-01-01",
                 "--end", "2018-12-31", "--fast", "--no-count-trials", "--json", str(out)])
    a = json.loads(out.read_text())[0]
    assert a["verdict"] != "PROVEN"
    assert any("regime check not applied" in r for r in a["reasons"])


def test_yfinance_style_offsets_across_dst_parse_to_local_dates(tmp_path):
    p = tmp_path / "yf.csv"
    p.write_text("Date,Open,High,Low,Close,Volume\n"
                 "2019-03-08 00:00:00-05:00,10,11,9,10.5,100\n"
                 "2019-03-11 00:00:00-04:00,10.5,11,10,10.8,100\n"
                 "2019-03-12 00:00:00+09:00,10.8,11,10,10.9,100\n")
    s = dat.load_csv(p)
    assert [str(d) for d in s.dates] == ["2019-03-08", "2019-03-11", "2019-03-12"]


def test_blank_date_is_reported_as_a_blank_date(tmp_path):
    p = tmp_path / "gap.csv"
    p.write_text("Date,Open,High,Low,Close\n2020-01-02,1,1,1,1\n,1,1,1,1\n")
    with pytest.raises(AdapterFetchError, match="blank or unreadable date"):
        dat.load_csv(p)


def test_all_zero_volume_is_treated_as_no_volume(tmp_path):
    """Yahoo reports FX volume as 0, not NaN."""
    s = dat.load_csv(_write_csv(tmp_path / "eurusd.csv", price=1.1))
    s = dataclasses_replace_volume(s, 0.0)
    _, rep = dat.quality_mask(s, dat.cfg.QualityThresholds(min_price=0.0))
    assert not rep.get("rejected")


def dataclasses_replace_volume(s, value):
    import dataclasses
    return dataclasses.replace(s, volume=np.full(len(s), value))


def test_duplicate_csv_names_are_refused(tmp_path):
    a = tmp_path / "a"; b = tmp_path / "b"; a.mkdir(); b.mkdir()
    f1 = _write_csv(a / "X.csv"); f2 = _write_csv(b / "X.csv", seed=1)
    with pytest.raises(SystemExit, match="both named X"):
        _cli().main(["run", "--rules", str(SKILL / "rule_template.py"), "--csv", str(f1), str(f2),
                     "--fast", "--no-count-trials", "--start", "2019-01-01", "--end", "2022-12-31"])
