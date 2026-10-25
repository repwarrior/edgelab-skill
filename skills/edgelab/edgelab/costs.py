"""Transaction cost model.

Costs are applied to every trade, on both sides, on actual notional:
commission + half-spread + a volatility-scaled slippage term, with a floor on
the round trip. The half-spread is an explicit, deliberately pessimistic number
by default (see `config.CostConfig` for why it is not estimated). Set it for
your market: the right figure for a mega-cap is wrong for a small cap, and the
choice quietly decides which rules survive.

All headline numbers in this package are net. `Trade.gross_ret` is retained only
so the report can show the cost drag explicitly: a rule earning 6bps gross that
dies against a 5bps round trip is a finding worth stating plainly, not a
detail to bury.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from . import config as cfg


def corwin_schultz_spread(
    high: np.ndarray, low: np.ndarray, lookback: int = 60
) -> np.ndarray:
    """Corwin-Schultz (2012) high/low bid-ask spread estimator, as a fraction.

    Uses the insight that the high/low range over two days reflects both
    volatility and the spread, but volatility scales with time while the spread
    does not -- so the two can be separated.

    DO NOT USE THIS ON LIQUID EQUITIES. Validated against a known spread
    (tests/test_costs.py), it overstates by 40-175x when the true spread is
    small relative to daily volatility, and its output tracks VOLATILITY rather
    than spread -- 66 bps at 1% daily vol, 175 bps at 2.5%, almost regardless of
    the true spread. The estimator cannot resolve a sub-basis-point spread
    inside 150 bps of range; the noise dwarfs the signal it is looking for.

    It remains here because it is genuinely useful where it was designed to be
    used -- illiquid names or historical data with no quotes, where the spread
    IS a meaningful fraction of the range. `CostConfig.half_spread_bps` is an
    explicit number by default, and this function is off unless asked for.
    """
    h = np.asarray(high, dtype=float)
    l = np.asarray(low, dtype=float)
    with np.errstate(invalid="ignore", divide="ignore"):
        hl = np.log(np.where((h > 0) & (l > 0), h / l, np.nan)) ** 2
        h2 = np.maximum(h[1:], h[:-1])
        l2 = np.minimum(l[1:], l[:-1])
        gamma = np.log(np.where((h2 > 0) & (l2 > 0), h2 / l2, np.nan)) ** 2
        beta = hl[1:] + hl[:-1]

        k = 3.0 - 2.0 * np.sqrt(2.0)
        alpha = (np.sqrt(2.0 * beta) - np.sqrt(beta)) / k - np.sqrt(gamma / k)
        s = 2.0 * (np.exp(alpha) - 1.0) / (1.0 + np.exp(alpha))

    s = np.where(np.isfinite(s) & (s > 0), s, np.nan)
    out = np.full(len(h), np.nan)
    out[2:] = s[:-1]                       # strictly causal: value at t uses t-1,t-2
    med = pd.Series(out).rolling(lookback, min_periods=max(5, lookback // 4)).median()
    return med.to_numpy(copy=True)


class CostModel:
    """Per-side cost in basis points, resolved per fill."""

    def __init__(self, conf: cfg.CostConfig | None = None,
                 half_spread_by_symbol: dict[str, float] | None = None) -> None:
        self.c = conf or cfg.SETTINGS.costs
        # Optional per-symbol half-spread in bps, for instruments whose spread
        # you actually know (a futures tick, a thin small cap). Overrides the
        # configured constant for that symbol only.
        self.by_symbol = dict(half_spread_by_symbol or {})

    def half_spread_for(self, symbol: str) -> float | None:
        return self.by_symbol.get(symbol)

    def half_spread_bps(self, spread_frac: float, price: float) -> float:
        if self.c.half_spread_bps is not None:
            return float(self.c.half_spread_bps)
        if not np.isfinite(spread_frac) or spread_frac <= 0:
            spread_frac = 0.0020            # 20bps fallback, deliberately pessimistic
        half = 0.5 * spread_frac * 1e4
        tick_floor = 0.5 * (0.01 / max(price, 1e-9)) * 1e4
        return float(max(half, tick_floor))

    def per_side_bps(self, spread_frac: float, price: float, daily_vol: float,
                     half_spread_bps: float | None = None) -> float:
        slippage = self.c.slippage_eta * (daily_vol * 1e4) * np.sqrt(self.c.participation)
        if not np.isfinite(slippage):
            slippage = 0.0
        # A per-symbol override names the half-spread explicitly; otherwise the
        # configured constant (or Corwin-Schultz, if enabled) applies.
        half = (float(half_spread_bps) if half_spread_bps is not None
                else self.half_spread_bps(spread_frac, price))
        return float(self.c.commission_bps + half + slippage)

    def round_trip_bps(
        self,
        spread_in: float, price_in: float,
        spread_out: float, price_out: float,
        daily_vol: float,
        half_spread_bps: float | None = None,
    ) -> float:
        rt = (
            self.per_side_bps(spread_in, price_in, daily_vol, half_spread_bps)
            + self.per_side_bps(spread_out, price_out, daily_vol, half_spread_bps)
        )
        return float(max(rt, self.c.min_round_trip_bps))
