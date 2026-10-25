"""Statistical machinery for deciding whether an edge is real.

The guiding principle: every routine here is designed to make it HARDER to
declare an edge, because the failure mode that costs money is a false positive,
not a false negative.

Four defenses, each closing a different hole:

  block bootstrap        -- trades overlap in time, so plain iid resampling
                            wildly overstates significance
  deflated Sharpe        -- testing many rules guarantees a winner by luck; the
                            hurdle rises with the number of trials attempted
  random-entry null      -- a long-only rule in a bull market makes money for
                            reasons that have nothing to do with the signal
  effective sample size  -- 100 correlated positions are not 100 observations
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np
from scipy import stats as sps

EULER_GAMMA = 0.5772156649015329


# --------------------------------------------------------------------------
# Basic moments
# --------------------------------------------------------------------------

def sharpe(returns: np.ndarray) -> float:
    """Per-observation Sharpe. NOT annualized -- annualizing a per-trade Sharpe
    requires a trades-per-year assumption that is usually wrong and always
    flattering."""
    r = np.asarray(returns, dtype=float)
    r = r[np.isfinite(r)]
    if r.size < 2:
        return float("nan")
    sd = r.std(ddof=1)
    if sd == 0:
        return float("nan")
    return float(r.mean() / sd)


def moments(returns: np.ndarray) -> tuple[float, float]:
    """(skewness, raw kurtosis). Raw, not excess -- the PSR formula wants raw."""
    r = np.asarray(returns, dtype=float)
    r = r[np.isfinite(r)]
    if r.size < 4:
        return 0.0, 3.0
    return float(sps.skew(r)), float(sps.kurtosis(r, fisher=False))


# --------------------------------------------------------------------------
# Block bootstrap
# --------------------------------------------------------------------------

def stationary_bootstrap_indices(
    n: int, size: int, block_len: float, rng: np.random.Generator
) -> np.ndarray:
    """Politis-Romano stationary bootstrap indices (geometric block lengths).

    Preserves serial dependence up to roughly `block_len`. Using iid resampling
    on overlapping trades is the single most common way a backtest's confidence
    interval ends up several times too narrow.
    """
    if n <= 0:
        return np.empty(0, dtype=int)
    p = 1.0 / max(block_len, 1.0)
    idx = np.empty(size, dtype=int)
    i = int(rng.integers(n))
    for k in range(size):
        idx[k] = i
        if rng.random() < p:
            i = int(rng.integers(n))
        else:
            i = (i + 1) % n
    return idx


@dataclass(frozen=True)
class BootstrapResult:
    mean: float
    mean_lo: float
    mean_hi: float
    sharpe: float
    sharpe_lo: float
    sharpe_hi: float
    p_mean_le_zero: float
    n_boot: int
    block_len: float


def block_bootstrap(
    returns: np.ndarray,
    n_boot: int = 2000,
    block_len: float | None = None,
    ci: float = 0.95,
    seed: int = 0,
) -> BootstrapResult:
    r = np.asarray(returns, dtype=float)
    r = r[np.isfinite(r)]
    n = r.size
    if n < 3:
        nan = float("nan")
        return BootstrapResult(nan, nan, nan, nan, nan, nan, nan, 0, 0.0)

    if block_len is None:
        block_len = max(1.0, float(np.ceil(n ** (1.0 / 3.0))))

    rng = np.random.default_rng(seed)
    means = np.empty(n_boot)
    sharpes = np.empty(n_boot)
    for b in range(n_boot):
        sample = r[stationary_bootstrap_indices(n, n, block_len, rng)]
        means[b] = sample.mean()
        sd = sample.std(ddof=1)
        sharpes[b] = sample.mean() / sd if sd > 0 else np.nan

    lo_q, hi_q = (1 - ci) / 2 * 100, (1 + ci) / 2 * 100
    return BootstrapResult(
        mean=float(r.mean()),
        mean_lo=float(np.percentile(means, lo_q)),
        mean_hi=float(np.percentile(means, hi_q)),
        sharpe=sharpe(r),
        sharpe_lo=float(np.nanpercentile(sharpes, lo_q)),
        sharpe_hi=float(np.nanpercentile(sharpes, hi_q)),
        p_mean_le_zero=float(np.mean(means <= 0.0)),
        n_boot=n_boot,
        block_len=float(block_len),
    )


# --------------------------------------------------------------------------
# Probabilistic / Deflated Sharpe (Bailey & Lopez de Prado)
# --------------------------------------------------------------------------

def probabilistic_sharpe(
    sr: float, sr_benchmark: float, n: int, skew: float, kurt: float
) -> float:
    """P(true Sharpe > sr_benchmark), correcting for non-normal returns.

    Skew and kurtosis matter: a strategy that sells tail risk shows a lovely
    Sharpe with negative skew and fat tails, and the naive t-test badly
    overstates its significance.
    """
    if not np.isfinite(sr) or n < 3:
        return float("nan")
    denom_sq = 1.0 - skew * sr + (kurt - 1.0) / 4.0 * sr**2
    if denom_sq <= 0:
        return float("nan")
    z = (sr - sr_benchmark) * math.sqrt(n - 1) / math.sqrt(denom_sq)
    return float(sps.norm.cdf(z))


def search_bar(n_trials: int, fwer: float = 0.05) -> float:
    """Percentile a best-of-n result must clear to hold family-wise error at `fwer`.

    Under the null a trial's percentile against its own permutation null is
    uniform, so the best of n independent trials exceeds b with probability
    1 - b**n. Setting that to `fwer` gives b = (1 - fwer) ** (1 / n).

    The tempting alternative, `1 - 1/n`, controls nothing. It reads as if it
    tightens with more trials -- 98% at n=50 looks strict -- but
    1 - (1 - 1/n)**n converges to 1 - 1/e, so it pins the family-wise
    false-positive rate at about 63% for every n. The correct bar at n=50 is
    99.90%.
    """
    if n_trials < 1:
        raise ValueError("n_trials must be >= 1")
    if not 0.0 < fwer < 1.0:
        raise ValueError("fwer must be in (0, 1)")
    return float((1.0 - fwer) ** (1.0 / n_trials))


def expected_max_sharpe(sr_variance: float, n_trials: int) -> float:
    """Expected maximum Sharpe achievable by chance across n_trials.

    This is the hurdle a strategy must clear simply for having been selected as
    the best of many. Try enough rules and something will look excellent; this
    quantifies how excellent 'excellent by luck' looks.
    """
    if not np.isfinite(sr_variance) or sr_variance < 0:
        # A missing variance must NOT become a missing hurdle. `trial_sharpe_variance`
        # returns NaN precisely so the caller decides; returning 0.0 here would hand
        # them a silent zero, and `deflated_sharpe` would then report DSR = 1.0, a
        # PASS manufactured by the absence of data.
        return float("nan")
    if n_trials <= 1 or sr_variance == 0:
        return 0.0
    z1 = sps.norm.ppf(1.0 - 1.0 / n_trials)
    z2 = sps.norm.ppf(1.0 - 1.0 / (n_trials * math.e))
    return float(math.sqrt(sr_variance) * ((1.0 - EULER_GAMMA) * z1 + EULER_GAMMA * z2))


def trial_sharpe_variance(null_paths: np.ndarray) -> float:
    """Variance of Sharpe across trials that have NO edge, measured from a null.

    This is the `sr_variance` the deflation formula asks for, and it is easy to
    supply something else by accident. Estimating it from the single observed
    Sharpe -- `sr_variance = sr**2` -- makes the hurdle
    `expected_max_sharpe(sr**2, n)`, i.e. roughly `2.2 * |sr|` at n=40: above the
    observation for every possible value of it. A gate that scales with the
    thing it is judging is not conservative, it is uninformative, and it reads
    like a real test in the report beside real ones.

    `null_paths` is (n_replicates, n_observations) from a permutation null. Each
    ROW is a complete no-edge history of the same length as the real one, so the
    spread of their Sharpes is exactly the quantity wanted -- measured rather
    than assumed. Returns NaN if fewer than two replicates yield a finite
    Sharpe; callers must decide what to do rather than receive a silent zero.
    """
    m = np.asarray(null_paths, dtype=float)
    if m.ndim != 2 or m.shape[0] < 2 or m.shape[1] < 2:
        return float("nan")
    sd = m.std(axis=1, ddof=1)
    sr = np.where(sd > 0, m.mean(axis=1) / np.where(sd > 0, sd, 1.0), np.nan)
    sr = sr[np.isfinite(sr)]
    if sr.size < 2:
        return float("nan")
    return float(np.var(sr, ddof=1))


def deflated_sharpe(
    returns: np.ndarray, sr_variance: float, n_trials: int
) -> tuple[float, float]:
    """Returns (DSR, hurdle). DSR is P(true Sharpe > the luck hurdle)."""
    r = np.asarray(returns, dtype=float)
    r = r[np.isfinite(r)]
    sr = sharpe(r)
    sk, ku = moments(r)
    hurdle = expected_max_sharpe(sr_variance, n_trials)
    return probabilistic_sharpe(sr, hurdle, r.size, sk, ku), hurdle


# --------------------------------------------------------------------------
# Multiple testing
# --------------------------------------------------------------------------

def benjamini_hochberg(pvals: np.ndarray, alpha: float = 0.05) -> np.ndarray:
    """BH step-up FDR control. Returns a boolean 'reject null' mask."""
    p = np.asarray(pvals, dtype=float)
    n = p.size
    if n == 0:
        return np.zeros(0, dtype=bool)
    order = np.argsort(p)
    ranked = p[order]
    thresh = alpha * (np.arange(1, n + 1) / n)
    passed = ranked <= thresh
    out = np.zeros(n, dtype=bool)
    if passed.any():
        cutoff = np.max(np.flatnonzero(passed))
        out[order[: cutoff + 1]] = True
    return out


# --------------------------------------------------------------------------
# Effective sample size
# --------------------------------------------------------------------------

def effective_sample_size(n: int, mean_cluster: float, mean_corr: float) -> float:
    """Kish design effect: n_eff = n / (1 + (m - 1) * rho).

    Trades open at the same moment share the market factor. Ten simultaneous
    long positions in correlated names behave much more like one observation
    than like ten, and treating them as ten is how a backtest manufactures
    significance out of nothing.
    """
    if n <= 0:
        return 0.0
    m = max(1.0, float(mean_cluster))
    rho = float(np.clip(mean_corr, 0.0, 1.0))
    deff = 1.0 + (m - 1.0) * rho
    return float(np.clip(n / max(deff, 1e-9), 1.0, n))


# --------------------------------------------------------------------------
# Purged, embargoed cross-validation
# --------------------------------------------------------------------------

def purged_kfold(
    entry_idx: np.ndarray,
    exit_idx: np.ndarray,
    n_splits: int = 5,
    embargo_frac: float = 0.01,
) -> list[tuple[np.ndarray, np.ndarray]]:
    """K-fold splits that purge label-overlapping observations and embargo after.

    A trade whose holding period straddles the train/test boundary leaks
    information across it. Purging removes those; the embargo additionally drops
    observations immediately following the test fold, where serial correlation
    would otherwise carry information backwards.
    """
    entry_idx = np.asarray(entry_idx)
    exit_idx = np.asarray(exit_idx)
    n = entry_idx.size
    if n < n_splits or n_splits < 2:
        return []

    order = np.argsort(entry_idx)
    folds = np.array_split(order, n_splits)
    span = int(max(exit_idx.max() - entry_idx.min(), 1))
    embargo = int(np.ceil(embargo_frac * span))

    out: list[tuple[np.ndarray, np.ndarray]] = []
    for f in folds:
        if f.size == 0:
            continue
        test = np.sort(f)
        t_start = int(entry_idx[test].min())
        t_end = int(exit_idx[test].max())
        overlaps = (exit_idx >= t_start) & (entry_idx <= t_end + embargo)
        train = np.setdiff1d(np.arange(n), np.flatnonzero(overlaps), assume_unique=False)
        if train.size and test.size:
            out.append((train, test))
    return out
