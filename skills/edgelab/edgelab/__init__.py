"""edgelab -- measure whether a trading rule actually has an edge.

This package does NOT produce buy/sell recommendations and has no order path.
It takes a candidate rule and reports, with the supporting numbers, whether that
rule had a real, cost-adjusted edge over a historical sample -- or whether it is
indistinguishable from picking entry dates at random.

The four verdicts:
    PROVEN              survived costs, multiple testing, a randomized-entry null,
                        both market regimes, and a causality audit
    LIKELY              positive and suggestive, but failed at least one hurdle
    NO_EDGE             indistinguishable from, or worse than, the null
    INSUFFICIENT_SAMPLE cannot be determined at this sample size
"""

from .config import SETTINGS, Settings, ensure_dirs
from .costs import CostModel, corwin_schultz_spread
from .data import (
    NpzCache,
    YFinanceAdapter,
    build_calendar,
    compute_regime,
    load_csv,
    load_universe,
    quality_mask,
)
from .engine import (
    ExecutionEngine,
    buy_and_hold_benchmark,
    random_entry_null,
    verify_causality,
)
from .indicators import INDICATOR_REGISTRY, assert_causal, probe_all
from .panel import Panel, SymbolWindow, build_panel
from .stats import (
    benjamini_hochberg,
    block_bootstrap,
    deflated_sharpe,
    effective_sample_size,
    probabilistic_sharpe,
    purged_kfold,
    sharpe,
)
from .types import (
    Bars,
    PortfolioMode,
    RawSeries,
    Rule,
    RunResult,
    SignalDecision,
    Trade,
    Verdict,
)
from .verdict import Assessment, assess_grid, load_trials, record_trials, render_report

__version__ = "1.0.0"

__all__ = [
    "SETTINGS", "Settings", "ensure_dirs",
    "CostModel", "corwin_schultz_spread",
    "NpzCache", "YFinanceAdapter", "build_calendar", "compute_regime",
    "load_csv", "load_universe", "quality_mask",
    "ExecutionEngine", "buy_and_hold_benchmark", "random_entry_null", "verify_causality",
    "INDICATOR_REGISTRY", "assert_causal", "probe_all",
    "Panel", "SymbolWindow", "build_panel",
    "benjamini_hochberg", "block_bootstrap", "deflated_sharpe",
    "effective_sample_size", "probabilistic_sharpe", "purged_kfold", "sharpe",
    "Bars", "PortfolioMode", "RawSeries", "Rule", "RunResult", "SignalDecision",
    "Trade", "Verdict",
    "Assessment", "assess_grid", "load_trials", "record_trials", "render_report",
]
