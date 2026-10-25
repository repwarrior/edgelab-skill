# edgelab

**A Claude skill that tells you when your trading strategy has no edge.**

Describe a rule ("buy when RSI(14) closes below 30, hold 10 days") and Claude
writes it, backtests it on real daily data, and tells you whether it beat random
entry dates after costs. You get one of four verdicts, with the numbers behind
it.

The answer you'll get most often is `NO_EDGE`. That's the point. Most homemade
backtests are wrong in the same few ways, and every one of those mistakes
pushes toward finding an edge that isn't there. edgelab is built so those
mistakes can't happen.

> Not financial advice. edgelab never places trades, never connects to a broker
> and never recommends a position. It measures the past, and the past is not
> the future.

---

## Install

In Claude Code:

```
/plugin marketplace add repwarrior/edgelab-skill
/plugin install edgelab@edgelab-skill
```

Or copy `skills/edgelab/` into `~/.claude/skills/`.

Then ask Claude something like:

```
Backtest this: buy when a stock closes below its lower Bollinger band, hold 5
days, on the 20 largest US stocks over the last 12 years.
```

On first use the skill installs its Python dependencies (numpy, pandas, scipy,
yfinance) into `~/.edgelab/venv` and runs a self-test.

## What it guards against

| Failure | What it does to a backtest | What edgelab does |
|---|---|---|
| Lookahead | The signal quietly uses tomorrow's data | Rules get a read-only window that ends at today's bar. Reading past it raises `IndexError`, so the bug can't be written at all |
| Entry at signal close | Fills at a price you couldn't have got | Fills at the **next bar's open**, asserted on every trade |
| Overlapping trades | Confidence intervals several times too narrow | Stationary block bootstrap, with block length ≥ holding period |
| Multiple testing | Try 50 rules and one looks great by luck | Deflated Sharpe ratio with a **permanent** trial counter, plus Benjamini-Hochberg FDR across each run |
| No real benchmark | A long-only rule looks good in a bull market | Randomized-entry null: same symbols, trade count, holding periods and costs, with entry dates drawn at random |
| Beta passing as alpha | The "edge" is just market exposure | Results split by bull/bear regime; must be positive in both |
| Censored trades | Unfinished trades closed at the last price | Marked censored and excluded, never closed early |
| Stops that never gap | Stops "cap" losses exactly | A bar that opens through the stop fills at the open |
| Bad prints | A 50% one-bar jump with no split becomes a fake trade | Suspect bars are marked untradable |
| Correlated positions | 100 overlapping trades counted as 100 observations | Effective sample size corrected for concurrency and correlation |

## The verdicts

| Verdict | Meaning |
|---|---|
| `PROVEN` | Survived costs, the random-entry null (99th percentile), the deflated-Sharpe hurdle, FDR control, both market regimes (at least 10 trades in each) and the causality audit |
| `LIKELY` | Positive and beats 95% of random entries, but failed at least one hurdle. The report names which |
| `NO_EDGE` | No better than, or worse than, random entries after costs |
| `INSUFFICIENT_SAMPLE` | Fewer than 30 completed trades. A real answer, not an error |

Read **`random-entry null: strategy beats X% of shuffles`** first. A rule with a
nice return that beats only 60% of shuffles has no timing edge. It just held
stocks that went up.

Read the cost line second: **`gross − costs = net`**. Plenty of real signals
have a genuine gross edge that costs eat completely.

Every threshold lives in `skills/edgelab/edgelab/config.py`. They're
deliberately strict. You can disagree with a specific number, but you can't
call the whole thing a black box.

## The trial counter

Every distinct rule-plus-parameters you ever test is recorded in
`~/.edgelab/trials.json`. The deflated-Sharpe hurdle rises with that count, so
the more variants you try, the harder it gets for any one of them to reach
`PROVEN`.

That's intended. If you test a hundred variants and the hundred-and-first looks
brilliant, you searched a hundred times, and the tool holds you to it.

## Using it without Claude

```bash
cd skills/edgelab
python3 -m pip install -r requirements.txt
python3 backtest.py selftest
cp rule_template.py my_rules.py        # edit the rule
python3 backtest.py run --rules my_rules.py --symbols AAPL MSFT JPM XOM JNJ PG --years 12
python3 backtest.py run --rules my_rules.py --csv btc.csv eth.csv --half-spread-bps 10 --min-price 0
python3 backtest.py trials
```

A rule is a plain Python function:

```python
from edgelab import Rule, SignalDecision
from edgelab import indicators as ind

def my_signal(window, p):
    # window.c[-1] is today's close; nothing after today exists in `window`.
    rsi = ind.rsi(window.bars, p["period"])
    if rsi[-1] < p["level"]:
        return SignalDecision(side=1, hold_bars=p["hold"])   # filled at tomorrow's open
    return None

RULES = [Rule("my_signal", my_signal, warmup_bars=50,
              params={"period": 14, "level": 30, "hold": 10})]
```

`rule_template.py` documents the full contract: shorts, ATR and percentage
stops, warmup, and the one thing you must never do inside a signal function.

Your own data: one CSV per symbol, daily bars, with columns Date, Open, High,
Low, Close and optionally Volume, Dividends and Stock Splits. Prices must be
split-adjusted and not dividend-adjusted. An `Adj Close` column is ignored on
purpose.

## Tests

```bash
pip install -r skills/edgelab/requirements.txt pytest
pytest -q
```

Two tests matter more than the rest, because they constrain the harness from
opposite sides:

- **`test_random_walk_is_never_proven`**: pure noise must never be graded
  `PROVEN`. This catches a gullible harness.
- **`test_injected_edge_is_detected`**: data with a large, mechanically
  injected edge must be detected. This catches a harness that just says no to
  everything, which would pass the first test trivially.

The suite also checks that:
- the causality prober actually catches a deliberately broken indicator;
- reading past the decision bar raises an error instead of returning a price;
- costs reduce returns;
- stops gap correctly;
- results reproduce exactly under a fixed seed.

## Limitations

Read these before trusting any number.

- **Survivorship bias.** The symbols you test exist today. Companies that were
  delisted, acquired or went to zero are missing. This flatters every long
  strategy, and nothing in the output corrects for it.
- **Universe selection.** You pick the symbols, and you already know how they
  did. The math can't see that.
- **Costs are modelled, not incurred.** The defaults (0.5 bps commission + 1 bps
  half-spread per side) suit liquid US large caps. Use a larger
  `--half-spread-bps` for anything thinner.
- **Daily bars only.** There's no intrabar path: if one bar touches both a stop
  and a target, the order is unknown. No intraday strategies.
- **Yahoo Finance data is unofficial.** It can be wrong, rate-limited or
  restated. edgelab caches raw prices and refuses to silently absorb a vendor
  rewriting history.
- **A `PROVEN` verdict is evidence, not a guarantee.** Published anomalies decay.
  An edge that survived 2014-2026 may not survive 2027.

## License

MIT
