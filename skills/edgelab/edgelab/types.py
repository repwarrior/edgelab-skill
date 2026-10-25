"""Core datatypes for edgelab.

Everything that crosses a module boundary is defined here so the boundaries stay
narrow and the shapes stay honest. Most types are frozen: a backtest that can
mutate its own inputs mid-run is a backtest whose results cannot be reproduced.
"""

from __future__ import annotations

import datetime as _dt
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Callable, Literal, Mapping, NamedTuple

import numpy as np

Side = Literal[1, -1]


# --------------------------------------------------------------------------
# Errors
# --------------------------------------------------------------------------

class EdgelabError(Exception):
    """Base for every error this package raises deliberately."""


class AdapterFetchError(EdgelabError):
    """A data source failed. Raised loudly instead of yielding a NaN column.

    This matters more than it looks: a silently-empty symbol produces zero
    trades, which produces an INSUFFICIENT_SAMPLE verdict that looks like a
    legitimate statistical result rather than a plumbing failure.
    """


class CacheIntegrityError(EdgelabError):
    """Cached history disagrees with freshly fetched history, with no split to
    explain it. A vendor restating history is a human decision, not something
    to paper over automatically."""


class CalendarMismatchError(EdgelabError):
    """Symbol does not trade on the reference (US) calendar."""


class LookaheadError(EdgelabError):
    """A signal or indicator touched data at or beyond the decision bar."""


class InsufficientDataError(EdgelabError):
    """Not enough bars to evaluate the rule at all."""


# --------------------------------------------------------------------------
# Price containers
# --------------------------------------------------------------------------

class Bars(NamedTuple):
    """Aligned OHLCV arrays. The uniform argument type for every indicator.

    Prices are split-adjusted but dividend-UNadjusted, i.e. actually tradable
    price levels. Volume may be all-NaN when the source's volume is quarantined.
    """

    o: np.ndarray
    h: np.ndarray
    l: np.ndarray
    c: np.ndarray
    v: np.ndarray

    def __len__(self) -> int:
        return len(self.c)

    def head(self, n: int) -> "Bars":
        """First n bars. Used by the causality prober."""
        return Bars(self.o[:n], self.h[:n], self.l[:n], self.c[:n], self.v[:n])


@dataclass(frozen=True)
class AdapterCaps:
    """What a data source is permitted to do.

    `executable` is load-bearing: a source whose prices are not what actually
    traded (a snapshot feed, a delayed quote) must set it False. An open that is
    off by a few bps is the same order as the edges being measured.
    """

    executable: bool
    max_history_days: int | None
    has_reliable_volume: bool
    is_point_in_time: bool


@dataclass(frozen=True)
class RawSeries:
    """One symbol's raw history, as cached. Never stores adjusted prices --
    adjusted prices are a derived view that silently rescales on every
    ex-dividend, so caching them makes reruns non-reproducible."""

    symbol: str
    dates: np.ndarray            # datetime64[D], naive, sorted, unique
    o: np.ndarray
    h: np.ndarray
    l: np.ndarray
    c: np.ndarray
    volume: np.ndarray           # NaN where the source is untrustworthy
    dividend: np.ndarray         # cash per share on the ex-date
    split: np.ndarray            # ratio; 0.0 where no split
    source: str = "yfinance"
    volume_quarantined: bool = False

    def __len__(self) -> int:
        return len(self.dates)


# --------------------------------------------------------------------------
# Signals and trades
# --------------------------------------------------------------------------

@dataclass(frozen=True)
class SignalDecision:
    """What a rule emits on bar t. The entry it schedules fills at t+1 open.

    A stop is optional and specified as a DISTANCE, not a price, because the
    entry price is not known when the decision is made -- it is next bar's open.
    Exactly one of stop_atr / stop_pct may be given; both None means no stop and
    the engine takes the plain scheduled-exit path.
    """

    side: Side
    hold_bars: int                  # tradable bars, not calendar days
    size: float = 1.0
    tag: str = ""
    stop_atr: float | None = None   # stop distance in ATR(period) multiples
    stop_pct: float | None = None   # stop distance as a fraction of entry price

    def __post_init__(self) -> None:
        if self.side not in (1, -1):
            raise ValueError(f"side must be 1 or -1, got {self.side!r}")
        if self.hold_bars < 1:
            raise ValueError(f"hold_bars must be >= 1, got {self.hold_bars}")
        if self.stop_atr is not None and self.stop_pct is not None:
            raise ValueError("specify stop_atr or stop_pct, not both")
        for name, v in (("stop_atr", self.stop_atr), ("stop_pct", self.stop_pct)):
            if v is not None and not (v > 0):
                raise ValueError(f"{name} must be positive, got {v!r}")

    @property
    def has_stop(self) -> bool:
        return self.stop_atr is not None or self.stop_pct is not None


TerminationReason = Literal["scheduled", "stopped", "delisted", "censored"]


@dataclass(frozen=True)
class Trade:
    """A completed round trip at fixed $1 notional.

    Sizing is deliberately non-compounding. Compounding makes each trade's
    return depend on the wealth path before it, which inflates late-sample
    Sharpe whenever the equity curve rises.
    """

    symbol: str
    side: Side
    tag: str
    signal_idx: int                 # bar the decision was made on
    entry_idx: int                  # bar the entry filled on (> signal_idx)
    exit_idx: int
    signal_date: _dt.date
    entry_date: _dt.date
    exit_date: _dt.date
    entry_px: float
    exit_px: float
    dividends_cash: float
    cost_cash: float
    borrow_cash: float
    gross_ret: float
    net_ret: float
    bars_held: int
    fill_delay_bars: int
    terminated: TerminationReason
    regime: str = "unknown"
    stop_px: float | None = None      # the stop level, if one was set
    gapped_through_stop: bool = False  # filled worse than the stop -- see engine
    slippage_px: float = 0.0          # |fill - stop|; 0.0 when not stopped.
                                      # The SIZE is surfaced, not just the fact:
                                      # a boolean cannot tell a one-tick miss
                                      # from a gap that doubled the loss.


class Verdict(str, Enum):
    PROVEN = "PROVEN"
    LIKELY = "LIKELY"
    NO_EDGE = "NO_EDGE"
    INSUFFICIENT_SAMPLE = "INSUFFICIENT_SAMPLE"


class PortfolioMode(str, Enum):
    NON_OVERLAPPING = "non_overlapping"
    PYRAMID = "pyramid"


# --------------------------------------------------------------------------
# Results
# --------------------------------------------------------------------------

@dataclass
class RunResult:
    """Output of one rule over one universe."""

    rule_id: str
    params: Mapping[str, Any]
    mode: PortfolioMode
    trades: list[Trade]
    n_censored: int = 0
    n_unfilled: int = 0
    n_delisted: int = 0
    symbols: tuple[str, ...] = ()
    max_daily_positions: int = 1
    warnings: list[str] = field(default_factory=list)
    causality_verified: bool | None = None   # None = the audit was not run
    causality_problems: list[str] = field(default_factory=list)
    config_hash: str = ""

    @property
    def net_returns(self) -> np.ndarray:
        if not self.trades:
            return np.empty(0, dtype=float)
        return np.array([t.net_ret for t in self.trades], dtype=float)

    @property
    def gross_returns(self) -> np.ndarray:
        if not self.trades:
            return np.empty(0, dtype=float)
        return np.array([t.gross_ret for t in self.trades], dtype=float)


SignalFn = Callable[[Any, Mapping[str, Any]], SignalDecision | None]


@dataclass(frozen=True)
class Rule:
    """A candidate signal. `warmup_bars` is declared, not inferred -- the engine
    refuses to evaluate before it, so an indicator cannot quietly emit a value
    computed from a half-filled window."""

    rule_id: str
    signal: SignalFn
    warmup_bars: int
    params: Mapping[str, Any] = field(default_factory=dict)
    description: str = ""
    hold_bars: int = 10
