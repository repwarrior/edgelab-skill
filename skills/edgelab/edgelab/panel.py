"""The lookahead firewall.

Lookahead bias is not prevented by discipline or code review. It is prevented by
never handing the signal function an object that physically contains the future.

A rule receives a `SymbolWindow`, not the panel. Its arrays are truncated at the
decision bar, so `window.c[-1]` is today's close and there is no bar t+1 inside
the object at all -- reaching for one raises IndexError rather than returning
tomorrow's price. The arrays are also marked non-writeable, so one symbol's rule
cannot corrupt another's data.

This is a structural guarantee, not a runtime check, which is the important
part: the class of bug is unwritable rather than merely detectable. The
complementary risk -- an indicator that looks forward internally, or an engine
precomputation that spans the whole series -- is covered by
`engine.verify_causality`.
"""

from __future__ import annotations

from typing import Iterable, Mapping, Sequence

import numpy as np

from .types import Bars, RawSeries

_FIELDS = ("o", "h", "l", "c", "v", "dividend", "split")


class SymbolWindow:
    """A read-only view of one symbol, as of bar `t`.

    len(window) == t + 1. window.c[-1] is the decision bar's close.
    """

    __slots__ = ("_t", "_symbol", "_dates", "o", "h", "l", "c", "v")

    def __init__(
        self,
        symbol: str,
        t: int,
        dates: np.ndarray,
        o: np.ndarray,
        h: np.ndarray,
        l: np.ndarray,
        c: np.ndarray,
        v: np.ndarray,
    ) -> None:
        self._symbol = symbol
        self._t = t
        self._dates = dates
        self.o, self.h, self.l, self.c, self.v = o, h, l, c, v

    @property
    def symbol(self) -> str:
        return self._symbol

    @property
    def t(self) -> int:
        return self._t

    @property
    def dates(self) -> np.ndarray:
        return self._dates

    @property
    def bars(self) -> Bars:
        return Bars(self.o, self.h, self.l, self.c, self.v)

    def __len__(self) -> int:
        return self._t + 1

    def __repr__(self) -> str:
        return f"<SymbolWindow {self._symbol} t={self._t} len={len(self)}>"


class Panel:
    """Aligned multi-symbol price panel, shape (n_dates, n_symbols)."""

    def __init__(
        self,
        dates: np.ndarray,
        symbols: Sequence[str],
        arrays: Mapping[str, np.ndarray],
        tradable: np.ndarray,
        regime: np.ndarray | None = None,
    ) -> None:
        self.dates = np.asarray(dates)
        self.symbols = tuple(symbols)
        self._idx = {s: i for i, s in enumerate(self.symbols)}
        self._a = {k: np.asarray(arrays[k], dtype=float) for k in _FIELDS}
        self.tradable = np.asarray(tradable, dtype=bool)
        self.regime = (
            np.asarray(regime) if regime is not None
            else np.full(len(self.dates), "unknown", dtype=object)
        )
        n_d, n_s = len(self.dates), len(self.symbols)
        for k, arr in self._a.items():
            if arr.shape != (n_d, n_s):
                raise ValueError(f"array {k!r} has shape {arr.shape}, expected {(n_d, n_s)}")
            arr.flags.writeable = False
        self.tradable.flags.writeable = False

        # First bar at which each symbol has any data. Signals before this are
        # not evaluated -- otherwise a rule "trades" a symbol before it listed.
        self.first_valid: dict[str, int] = {}
        for s, j in self._idx.items():
            ok = np.flatnonzero(np.isfinite(self._a["c"][:, j]))
            self.first_valid[s] = int(ok[0]) if ok.size else len(self.dates)

    # -- accessors ---------------------------------------------------------

    @property
    def n_dates(self) -> int:
        return len(self.dates)

    def col(self, field: str, symbol: str) -> np.ndarray:
        return self._a[field][:, self._idx[symbol]]

    def is_tradable(self, symbol: str, t: int) -> bool:
        return bool(self.tradable[t, self._idx[symbol]])

    def window(self, symbol: str, t: int) -> SymbolWindow:
        """Truncated, immutable view. Indexing past t raises IndexError."""
        j = self._idx[symbol]
        sl = slice(0, t + 1)
        return SymbolWindow(
            symbol, t, self.dates[sl],
            self._a["o"][sl, j], self._a["h"][sl, j], self._a["l"][sl, j],
            self._a["c"][sl, j], self._a["v"][sl, j],
        )

    def total_return_close(self, symbol: str) -> np.ndarray:
        """Dividend-adjusted close, for indicator input only -- never for fills.

        Computed locally rather than taken from the vendor so it is
        deterministic: a vendor's adjusted series silently rescales its entire
        history every time a new dividend lands, which makes cached results
        irreproducible.
        """
        j = self._idx[symbol]
        c = self._a["c"][:, j]
        d = self._a["dividend"][:, j]
        factor = np.ones(len(c), dtype=float)
        run = 1.0
        for i in range(len(c) - 1, 0, -1):
            if np.isfinite(d[i]) and d[i] > 0 and np.isfinite(c[i - 1]) and c[i - 1] > 0:
                run *= 1.0 - d[i] / c[i - 1]
            factor[i - 1] = run
        return c * factor


def build_panel(
    series: Iterable[RawSeries],
    calendar: np.ndarray,
    tradable_masks: Mapping[str, np.ndarray] | None = None,
    regime: np.ndarray | None = None,
) -> Panel:
    """Reindex raw series onto a common calendar.

    Prices are NEVER forward-filled for execution. Forward-filling a halted
    symbol invents a fill at a price that did not exist, which is how a backtest
    "trades" through a halt and books a fake gap.
    """
    series = sorted(series, key=lambda s: s.symbol)
    symbols = [s.symbol for s in series]
    n_d, n_s = len(calendar), len(symbols)
    arrays = {f: np.full((n_d, n_s), np.nan, dtype=float) for f in _FIELDS}
    tradable = np.zeros((n_d, n_s), dtype=bool)

    cal_pos = {np.datetime64(d, "D"): i for i, d in enumerate(calendar)}
    for j, s in enumerate(series):
        rows = np.array(
            [cal_pos.get(np.datetime64(d, "D"), -1) for d in s.dates], dtype=int
        )
        keep = rows >= 0
        r = rows[keep]
        arrays["o"][r, j] = s.o[keep]
        arrays["h"][r, j] = s.h[keep]
        arrays["l"][r, j] = s.l[keep]
        arrays["c"][r, j] = s.c[keep]
        arrays["v"][r, j] = s.volume[keep]
        arrays["dividend"][r, j] = np.nan_to_num(s.dividend[keep], nan=0.0)
        arrays["split"][r, j] = np.nan_to_num(s.split[keep], nan=0.0)

        mask = (
            tradable_masks[s.symbol][keep]
            if tradable_masks and s.symbol in tradable_masks
            else np.ones(keep.sum(), dtype=bool)
        )
        tradable[r, j] = mask

    return Panel(calendar, symbols, arrays, tradable, regime=regime)
