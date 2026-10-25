"""Data acquisition, caching, calendar construction and quality gates.

yfinance is the source of executable prices: deep daily history, free, and the
prices the fill logic uses are ones that actually traded. Prices from any other
source should be checked against it before use -- a few bps of disagreement on
the open is the same order of magnitude as the edges this tool measures.

This is the ONLY module permitted to import yfinance. A rule or stats routine
that calls yfinance directly bypasses the cache, the quality gates, the calendar
and the lookahead firewall all at once.

Data you bring yourself (a CSV export from your broker or data vendor) enters
through `load_csv`, and goes through the same quality gates.
"""

from __future__ import annotations

import datetime as _dt
import json
import time
import warnings
import zoneinfo
from pathlib import Path
from typing import Iterable, Mapping, Sequence

import numpy as np
import pandas as pd

from . import config as cfg
from .types import (
    AdapterCaps,
    AdapterFetchError,
    CacheIntegrityError,
    RawSeries,
)

_ET = zoneinfo.ZoneInfo("America/New_York")


# --------------------------------------------------------------------------
# Adapter
# --------------------------------------------------------------------------

class YFinanceAdapter:
    """Per-symbol fetch. Deliberately NOT `yf.download` on a symbol list.

    Batch download turns a dead or misspelled ticker into a silent all-NaN
    column. That produces zero trades, which produces an INSUFFICIENT_SAMPLE
    verdict that looks like a legitimate statistical finding rather than the
    plumbing failure it actually is. Per-symbol fetch with raise_errors=True
    turns the same failure into a loud exception.
    """

    name = "yfinance"

    def capabilities(self) -> AdapterCaps:
        return AdapterCaps(
            executable=True,
            max_history_days=None,
            has_reliable_volume=True,
            is_point_in_time=False,
        )

    def fetch(self, symbol: str, start: _dt.date, end: _dt.date) -> RawSeries:
        import yfinance as yf  # imported here so the ban is easy to enforce

        last_exc: Exception | None = None
        for attempt in range(3):
            try:
                # yfinance on recent pandas emits pages of deprecation notices
                # that say nothing about the data; keep them out of the report.
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore")
                    df = yf.Ticker(symbol).history(
                        start=start.isoformat(),
                        end=(end + _dt.timedelta(days=1)).isoformat(),
                        auto_adjust=False,      # raw-ish prices; see module docstring
                        actions=True,
                        raise_errors=True,
                    )
                break
            except Exception as exc:       # noqa: BLE001 - re-raised as our type
                last_exc = exc
                time.sleep(0.4 * (attempt + 1))
        else:
            raise AdapterFetchError(f"{symbol}: yfinance fetch failed: {last_exc}") from last_exc

        if df is None or df.empty:
            raise AdapterFetchError(f"{symbol}: yfinance returned no rows")

        # tz-aware midnight America/New_York -> naive dates, or DST produces
        # off-by-one joins against the calendar.
        idx = pd.DatetimeIndex(df.index)
        if idx.tz is not None:
            idx = idx.tz_convert(_ET).tz_localize(None)
        dates = idx.normalize().to_numpy().astype("datetime64[D]")

        def col(name: str, default: float = np.nan) -> np.ndarray:
            if name not in df.columns:
                return np.full(len(df), default, dtype=float)
            return df[name].to_numpy(dtype=float, copy=True)

        series = RawSeries(
            symbol=symbol,
            dates=dates,
            o=col("Open"), h=col("High"), l=col("Low"), c=col("Close"),
            volume=col("Volume"),
            dividend=np.nan_to_num(col("Dividends"), nan=0.0),
            split=np.nan_to_num(col("Stock Splits"), nan=0.0),
            source="yfinance",
        )
        return _drop_incomplete_tail(series)


def _drop_incomplete_tail(s: RawSeries) -> RawSeries:
    """Never store a bar for a session that has not finished.

    A partial bar cached at midday poisons every downstream rerun, and worse,
    it does so invisibly -- the cache looks complete.
    """
    if len(s) == 0:
        return s
    today_et = _dt.datetime.now(_ET).date()
    keep = s.dates < np.datetime64(today_et, "D")
    if keep.all():
        return s
    return _subset(s, keep)


def _subset(s: RawSeries, keep: np.ndarray) -> RawSeries:
    return RawSeries(
        symbol=s.symbol, dates=s.dates[keep], o=s.o[keep], h=s.h[keep],
        l=s.l[keep], c=s.c[keep], volume=s.volume[keep],
        dividend=s.dividend[keep], split=s.split[keep], source=s.source,
        volume_quarantined=s.volume_quarantined,
    )


# --------------------------------------------------------------------------
# Cache
# --------------------------------------------------------------------------

def _safe_name(symbol: str) -> str:
    return symbol.replace(".", "_DOT_").replace("^", "_CARET_").replace("/", "_SLASH_")


class NpzCache:
    """Raw-layer-only cache. Adjusted prices are a derived view, never stored."""

    def __init__(self, directory: Path | None = None) -> None:
        self.dir = Path(directory or cfg.OHLCV_DIR)
        self.dir.mkdir(parents=True, exist_ok=True)

    def _path(self, symbol: str) -> Path:
        return self.dir / f"{_safe_name(symbol)}.npz"

    def load(self, symbol: str) -> RawSeries | None:
        p = self._path(symbol)
        if not p.exists():
            return None
        with np.load(p, allow_pickle=False) as z:
            meta = json.loads(str(z["meta"]))
            return RawSeries(
                symbol=meta["symbol"],
                dates=z["dates"].astype("datetime64[D]"),
                o=z["o"], h=z["h"], l=z["l"], c=z["c"],
                volume=z["volume"], dividend=z["dividend"], split=z["split"],
                source=meta.get("source", "yfinance"),
            )

    def save(self, s: RawSeries) -> None:
        meta = json.dumps({
            "symbol": s.symbol, "source": s.source,
            "first": str(s.dates[0]) if len(s) else None,
            "last": str(s.dates[-1]) if len(s) else None,
            "n": len(s),
        })
        np.savez_compressed(
            self._path(s.symbol),
            dates=s.dates.astype("datetime64[D]"), o=s.o, h=s.h, l=s.l, c=s.c,
            volume=s.volume, dividend=s.dividend, split=s.split,
            meta=np.array(meta),
        )

    def ensure(
        self,
        symbol: str,
        start: _dt.date,
        end: _dt.date,
        adapter: YFinanceAdapter,
        force_refresh: bool = False,
    ) -> RawSeries:
        cached = None if force_refresh else self.load(symbol)
        if cached is None:
            fresh = adapter.fetch(symbol, start, end)
            self.save(fresh)
            return _clip(fresh, start, end)

        have_first = cached.dates[0].astype("datetime64[D]").astype(_dt.date)
        have_last = cached.dates[-1].astype("datetime64[D]").astype(_dt.date)
        if have_first <= start and have_last >= end - _dt.timedelta(days=5):
            return _clip(cached, start, end)

        fresh = adapter.fetch(symbol, min(start, have_first), max(end, have_last))
        _check_overlap_integrity(cached, fresh, symbol)
        self.save(fresh)
        return _clip(fresh, start, end)


def _clip(s: RawSeries, start: _dt.date, end: _dt.date) -> RawSeries:
    keep = (s.dates >= np.datetime64(start, "D")) & (s.dates <= np.datetime64(end, "D"))
    return _subset(s, keep)


def _check_overlap_integrity(cached: RawSeries, fresh: RawSeries, symbol: str) -> None:
    """Cached and fresh history must agree on their overlap.

    If they do not, one of two things happened: a split retroactively rebased
    the whole series (legitimate, and detectable), or the vendor restated
    history (not legitimate to absorb silently).
    """
    common, ci, fi = np.intersect1d(cached.dates, fresh.dates, return_indices=True)
    if common.size == 0:
        return
    a, b = cached.c[ci], fresh.c[fi]
    ok = np.isfinite(a) & np.isfinite(b) & (a != 0)
    if not ok.any():
        return
    rel = np.abs(a[ok] - b[ok]) / np.abs(a[ok])
    if rel.max() < 1e-6:
        return
    if np.any(np.asarray(fresh.split)[fi][ok] > 0):
        return  # split rebase: expected, caller overwrites
    raise CacheIntegrityError(
        f"{symbol}: cached history disagrees with fresh fetch on "
        f"{int((rel >= 1e-6).sum())}/{ok.sum()} overlapping bars "
        f"(max rel diff {rel.max():.3e}) with no split to explain it. "
        f"Re-run with force_refresh=True only if you accept the vendor's restatement."
    )


# --------------------------------------------------------------------------
# Calendar
# --------------------------------------------------------------------------

def build_calendar(
    series_by_symbol: Mapping[str, RawSeries],
    refs: Sequence[str] = cfg.CALENDAR_REFS,
    min_refs: int = cfg.MIN_CALENDAR_REFS,
) -> np.ndarray:
    """A date is a session iff at least `min_refs` reference ETFs printed a bar.

    Two-of-four survives one bad reference feed while still excluding genuine
    market closures that a single-reference calendar would misclassify.
    """
    present = [series_by_symbol[r].dates for r in refs if r in series_by_symbol]
    if not present:
        present = [s.dates for s in series_by_symbol.values()]
        min_refs = 1
    all_dates, counts = np.unique(np.concatenate(present), return_counts=True)
    return np.sort(all_dates[counts >= min(min_refs, len(present))]).astype("datetime64[D]")


# --------------------------------------------------------------------------
# Quality gates
# --------------------------------------------------------------------------

class QualityReport(dict):
    """Per-symbol gate outcomes; truthy `rejected` means drop the symbol."""


def quality_mask(
    s: RawSeries,
    q: cfg.QualityThresholds = cfg.SETTINGS.quality,
) -> tuple[np.ndarray, QualityReport]:
    """Return a per-bar tradable mask plus a report.

    The gate that matters most is SUSPECT_PRINT. A 50%+ single-bar move with no
    split and no large dividend is the signature of an unadjusted corporate
    action or a bad print -- and it is exactly the bar a mean-reversion rule
    will preferentially trade into, manufacturing a spectacular fake edge.
    """
    n = len(s)
    mask = np.ones(n, dtype=bool)
    rep = QualityReport(symbol=s.symbol, n_bars=n, rejected=None)

    finite = np.isfinite(s.o) & np.isfinite(s.h) & np.isfinite(s.l) & np.isfinite(s.c)
    positive = (s.o > 0) & (s.h > 0) & (s.l > 0) & (s.c > 0)
    ordered = (s.h >= s.l) & (s.c <= s.h) & (s.c >= s.l) & (s.o <= s.h) & (s.o >= s.l)
    malformed = ~(finite & positive & ordered)
    mask &= ~malformed
    rep["n_malformed"] = int(malformed.sum())

    with np.errstate(invalid="ignore", divide="ignore"):
        logret = np.diff(np.log(np.where(s.c > 0, s.c, np.nan)), prepend=np.nan)
    prev_c = np.roll(s.c, 1)
    prev_c[0] = np.nan
    with np.errstate(invalid="ignore", divide="ignore"):
        div_frac = np.where(prev_c > 0, s.dividend / prev_c, 0.0)
    suspect = (
        (np.abs(logret) > q.suspect_move)
        & (s.split <= 0)
        & (np.nan_to_num(div_frac) < q.suspect_div_frac)
    )
    mask &= ~suspect
    rep["n_suspect_print"] = int(suspect.sum())

    no_vol = np.isnan(s.volume) | (s.volume == 0)
    stale = (s.o == s.h) & (s.h == s.l) & (s.l == s.c) & no_vol
    mask &= ~stale
    rep["n_stale"] = int(stale.sum())

    penny = np.nan_to_num(s.c, nan=0.0) < q.min_price
    mask &= ~penny
    rep["n_penny"] = int(penny.sum())

    # A source with no volume at all -- NaN in a CSV, or all zeros as Yahoo
    # reports FX -- is not a zero-volume symbol. The gate only applies when
    # some real volume was supplied.
    if n and np.nan_to_num(s.volume).any():
        zero_vol_frac = float(np.mean(np.nan_to_num(s.volume) == 0))
        rep["zero_volume_frac"] = zero_vol_frac
        if zero_vol_frac > q.max_zero_volume_frac:
            rep["rejected"] = f"ZERO_VOLUME {zero_vol_frac:.1%}"
    rep["n_tradable"] = int(mask.sum())
    return mask, rep


# --------------------------------------------------------------------------
# Regime tagging (causal)
# --------------------------------------------------------------------------

def compute_regime(
    benchmark: RawSeries,
    calendar: np.ndarray,
    trend_window: int = 200,
    vol_window: int = 20,
) -> np.ndarray:
    """Label each session by market regime, using only past data.

    Regime is used to check whether an apparent edge is just beta wearing a
    costume. The volatility split uses an EXPANDING median rather than a
    full-sample one -- a full-sample quantile would leak the future into the
    label, which matters because the label slices the results.
    """
    pos = {d: i for i, d in enumerate(calendar)}
    c = np.full(len(calendar), np.nan)
    for d, px in zip(benchmark.dates, benchmark.c):
        i = pos.get(np.datetime64(d, "D"))
        if i is not None:
            c[i] = px
    s = pd.Series(c).ffill()
    sma = s.rolling(trend_window, min_periods=trend_window).mean()
    trend = np.where(s.to_numpy(copy=True) > sma.to_numpy(copy=True), "bull", "bear")

    r = np.log(s).diff()
    rv = r.rolling(vol_window, min_periods=vol_window).std(ddof=0)
    expanding_med = rv.expanding(min_periods=vol_window * 3).median()
    vol = np.where(
        rv.to_numpy(copy=True) > expanding_med.to_numpy(copy=True), "stressed", "calm"
    )

    out = np.array([f"{t}/{v}" for t, v in zip(trend, vol)], dtype=object)
    unknown = ~np.isfinite(sma.to_numpy(copy=True)) | ~np.isfinite(
        expanding_med.to_numpy(copy=True)
    )
    out[unknown] = "unknown"
    return out


# --------------------------------------------------------------------------
# Top-level loader
# --------------------------------------------------------------------------

def load_universe(
    symbols: Iterable[str],
    start: _dt.date,
    end: _dt.date,
    cache: NpzCache | None = None,
    adapter: YFinanceAdapter | None = None,
    include_refs: bool = True,
    verbose: bool = True,
    force_refresh: bool = False,
    quality: cfg.QualityThresholds | None = None,
) -> tuple[dict[str, RawSeries], dict[str, np.ndarray], list[QualityReport]]:
    """Fetch/caches a universe and apply quality gates.

    Returns (series_by_symbol, tradable_masks, quality_reports). Symbols that
    fail a symbol-level gate are dropped and reported -- never silently kept.
    """
    cache = cache or NpzCache()
    adapter = adapter or YFinanceAdapter()
    wanted = list(dict.fromkeys(list(symbols) + (list(cfg.CALENDAR_REFS) if include_refs else [])))

    series: dict[str, RawSeries] = {}
    masks: dict[str, np.ndarray] = {}
    reports: list[QualityReport] = []
    for sym in wanted:
        try:
            s = cache.ensure(sym, start, end, adapter, force_refresh=force_refresh)
        except (AdapterFetchError, CacheIntegrityError) as exc:
            if verbose:
                print(f"  [drop] {sym}: {exc}")
            reports.append(QualityReport(symbol=sym, rejected=str(exc), n_bars=0))
            continue
        if len(s) == 0:
            reports.append(QualityReport(symbol=sym, rejected="EMPTY", n_bars=0))
            continue
        mask, rep = quality_mask(s, quality) if quality is not None else quality_mask(s)
        reports.append(rep)
        if rep.get("rejected"):
            if verbose:
                print(f"  [drop] {sym}: {rep['rejected']}")
            continue
        series[sym] = s
        masks[sym] = mask
    return series, masks, reports


# --------------------------------------------------------------------------
# Bring-your-own data
# --------------------------------------------------------------------------

_CSV_ALIASES = {
    "date": ("date", "datetime", "time", "timestamp", "day"),
    "o": ("open", "o"),
    "h": ("high", "h"),
    "l": ("low", "l"),
    "c": ("close", "c", "last"),
    "v": ("volume", "vol", "v"),
    "dividend": ("dividend", "dividends"),
    "split": ("split", "splits", "stock splits"),
}


def load_csv(path: str | Path, symbol: str | None = None) -> RawSeries:
    """One symbol's DAILY bars from a CSV file.

    Needs a date column and open/high/low/close (names matched case-insensitively,
    common aliases accepted). Volume, dividend and split columns are optional.

    Prices must be split-adjusted and NOT dividend-adjusted -- the levels you
    could actually have traded at. A dividend-adjusted "Adj Close" column is
    deliberately ignored: it rescales history every time a dividend lands.

    Intraday rows are rejected rather than silently resampled: the engine's
    holding periods, costs and regime labels are all daily-bar quantities.
    """
    p = Path(path)
    df = pd.read_csv(p)
    lower = {c.strip().lower(): c for c in df.columns}

    def pick(key: str, required: bool) -> str | None:
        for alias in _CSV_ALIASES[key]:
            if alias in lower:
                return lower[alias]
        if required:
            raise AdapterFetchError(
                f"{p.name}: no {key!r} column (looked for {', '.join(_CSV_ALIASES[key])}); "
                f"columns are {list(df.columns)}"
            )
        return None

    cols = {k: pick(k, k in ("date", "o", "h", "l", "c")) for k in _CSV_ALIASES}
    # Keep the LOCAL wall-clock date and drop any UTC offset. yfinance exports
    # write "2019-01-02 00:00:00-05:00" and the offset flips at DST, which
    # pandas refuses to parse as one column; converting to UTC instead would
    # move east-of-UTC sessions onto the previous day.
    raw = df[cols["date"]].astype(str).str.strip()
    raw = raw.str.replace(r"(?:Z|[+-]\d{2}:?\d{2})$", "", regex=True)
    stamps = pd.to_datetime(raw, errors="coerce")
    if stamps.isna().any():
        bad = int(stamps.isna().sum())
        raise AdapterFetchError(f"{p.name}: {bad} row(s) with a blank or unreadable date")
    if (stamps != stamps.dt.normalize()).any():
        raise AdapterFetchError(
            f"{p.name}: rows carry a time of day. edgelab tests DAILY bars only; "
            f"resample to one bar per session first."
        )
    df = df.assign(_d=stamps.dt.normalize()).sort_values("_d")
    if df["_d"].duplicated().any():
        raise AdapterFetchError(f"{p.name}: duplicate dates")

    n = len(df)

    def arr(key: str, default: float) -> np.ndarray:
        c = cols[key]
        if c is None:
            return np.full(n, default, dtype=float)
        return pd.to_numeric(df[c], errors="coerce").to_numpy(dtype=float, copy=True)

    return RawSeries(
        symbol=symbol or p.stem.upper(),
        dates=df["_d"].to_numpy().astype("datetime64[D]"),
        o=arr("o", np.nan), h=arr("h", np.nan), l=arr("l", np.nan), c=arr("c", np.nan),
        volume=arr("v", np.nan),
        dividend=np.nan_to_num(arr("dividend", 0.0), nan=0.0),
        split=np.nan_to_num(arr("split", 0.0), nan=0.0),
        source=f"csv:{p.name}",
    )
