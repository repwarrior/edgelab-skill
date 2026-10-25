"""Every tunable in one place, with deliberately conservative defaults.

Where a choice biases results, the default is set in the direction that makes a
strategy look WORSE. A harness whose defaults flatter strategies is a harness
that will eventually cost its user money.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path

# Everything edgelab writes lives here: the price cache and the persistent
# trial counter. Outside the skill folder on purpose -- a plugin update must not
# wipe the trial count, because the trial count is what keeps you honest.
DATA_DIR = Path(os.environ.get("EDGELAB_HOME", Path.home() / ".edgelab")).expanduser()
CACHE_DIR = DATA_DIR / "cache"
OHLCV_DIR = CACHE_DIR / "ohlcv"
TRIALS_FILE = DATA_DIR / "trials.json"

MASTER_SEED = 1234567

# Reference symbols used to derive the trading calendar. A date is a session
# iff at least MIN_CALENDAR_REFS of these print a bar -- surviving one bad feed.
CALENDAR_REFS = ("SPY", "QQQ", "IWM", "DIA")
MIN_CALENDAR_REFS = 2

# Regime tagging references.
REGIME_BENCHMARK = "SPY"


@dataclass(frozen=True)
class QualityThresholds:
    """Bar- and symbol-level data gates. See `data.quality_mask`."""

    min_price: float = 5.0              # below this, spreads are unmodelable
    suspect_move: float = 0.405465      # log(1.5); an unadjusted split's signature
    suspect_div_frac: float = 0.20      # a >20% special dividend can legitimately gap
    max_zero_volume_frac: float = 0.05
    max_missing_session_frac: float = 0.02
    min_bars_base: int = 252


@dataclass(frozen=True)
class StatsConfig:
    """Thresholds that decide verdicts. Each is a deliberate, arguable choice --
    they are surfaced in the report so the user can disagree with them."""

    n_bootstrap: int = 2000
    n_permutations: int = 2000
    bootstrap_ci: float = 0.95
    fdr_alpha: float = 0.05

    # Sample-size gates. Below min_trades_hard nothing can be concluded at all.
    min_trades_hard: int = 30
    min_trades_proven: int = 100
    min_symbols_proven: int = 5
    min_eff_sample_proven: float = 50.0

    # Hurdles for PROVEN.
    dsr_threshold: float = 0.95         # deflated Sharpe prob. that true SR > 0
    perm_percentile_proven: float = 0.99
    perm_percentile_likely: float = 0.95
    require_positive_in_both_regimes: bool = True
    min_trades_per_regime: int = 10     # below this in either regime, the check counts as failed
    require_causality_audit: bool = True   # no PROVEN without a passed causality audit


@dataclass(frozen=True)
class CostConfig:
    """Round-trip cost model.

    Why the half-spread is an explicit number rather than estimated: the
    Corwin-Schultz high/low estimator, validated against a known spread,
    overstates by 40-175x on liquid names -- it cannot resolve a sub-basis-point
    spread inside 150 bps of daily range, so it returns noise scaled by
    volatility. See
    `costs.corwin_schultz_spread` and tests/test_costs.py.

    Defaults are explicit and conservative for liquid US large caps:
    a penny spread on a $100-500 stock is 0.2-1.0 bps, so a 1.0 bps HALF-spread
    is roughly 2-5x pessimistic. Participation is retail-sized rather than an
    institutional block. Total round trip lands near 6 bps.

    These are ASSUMPTIONS, not measurements. Any result sensitive to them should
    say so, and a less liquid universe needs different numbers.
    """

    commission_bps: float = 0.5         # per side; 0 at many retail brokers
    half_spread_bps: float | None = 1.0   # explicit; None re-enables Corwin-Schultz
    slippage_eta: float = 1.0
    participation: float = 0.0001       # retail size, not a 1% institutional block
    borrow_bps_per_day: float = 1.5
    min_round_trip_bps: float = 2.0     # floor
    spread_lookback: int = 60


@dataclass(frozen=True)
class EngineConfig:
    max_fill_delay_bars: int = 3
    allow_shorts: bool = True
    stop_atr_period: int = 14      # ATR window backing stop_atr distances
    min_price: float = 5.0
    seed: int = MASTER_SEED


@dataclass(frozen=True)
class Settings:
    quality: QualityThresholds = field(default_factory=QualityThresholds)
    stats: StatsConfig = field(default_factory=StatsConfig)
    costs: CostConfig = field(default_factory=CostConfig)
    engine: EngineConfig = field(default_factory=EngineConfig)
    seed: int = MASTER_SEED


SETTINGS = Settings()


def ensure_dirs() -> None:
    for d in (DATA_DIR, CACHE_DIR, OHLCV_DIR):
        d.mkdir(parents=True, exist_ok=True)
