"""Rules file template for edgelab.

Copy this file, replace the example with the rule you want to test, and run:

    python3 backtest.py run --rules my_rules.py --symbols AAPL MSFT ... --years 12

THE CONTRACT
------------
A signal function receives `(window, params)` on every bar and returns either
None (do nothing) or a SignalDecision.

  window.o / .h / .l / .c / .v   numpy arrays, oldest first, ENDING AT TODAY.
                                  window.c[-1] is today's close. There is no
                                  tomorrow inside the window: reaching for it
                                  raises IndexError. The arrays are read-only.
  window.bars                     the same arrays as an edgelab.Bars, which is
                                  what every function in edgelab.indicators takes
  window.dates                    numpy datetime64[D] array, same length
  window.symbol                   the ticker

A decision made on today's bar is filled at the NEXT bar's open, never at
today's close. `hold_bars` is counted in trading sessions. Costs are charged on
both sides. You cannot opt out of any of that.

  SignalDecision(side=1, hold_bars=10)                 long, exit at the open 10 bars later
  SignalDecision(side=-1, hold_bars=5)                 short (borrow cost is charged)
  SignalDecision(side=1, hold_bars=20, stop_atr=2.0)   stop 2 x ATR(14) below the fill
  SignalDecision(side=1, hold_bars=20, stop_pct=0.05)  stop 5% below the fill

With a stop, hold_bars becomes a maximum. A bar that opens beyond the stop fills
at that open, not at the stop -- gaps are real.

Rule(rule_id, signal_fn, warmup_bars, params, description, hold_bars)
  warmup_bars   bars to skip at the start of each symbol so indicators are
                fully formed. Set it to at least your longest lookback.
  params        passed to your function. Changing a param makes it a NEW
                strategy for the trial counter -- that is the point.

DO NOT read anything from outside the window inside a signal function: no files,
no network, no global arrays of prices. That is the one way to smuggle the future
past the firewall, and nothing can detect it for you.
"""

from edgelab import Rule, SignalDecision
from edgelab import indicators as ind


# ---- EXAMPLE ONLY: replace with the rule you want to test --------------------

def close_crosses_above_sma(window, p):
    """Long when today's close crosses above its N-day simple moving average."""
    if len(window.c) < p["period"] + 2:
        return None
    sma = ind.sma(window.bars, p["period"])
    crossed = window.c[-2] <= sma[-2] and window.c[-1] > sma[-1]
    return SignalDecision(side=1, hold_bars=p["hold"]) if crossed else None


RULES = [
    Rule(
        rule_id="sma50_cross_up",
        signal=close_crosses_above_sma,
        warmup_bars=60,
        params={"period": 50, "hold": 20},
        description="close crosses above 50-day SMA, hold 20 sessions",
        hold_bars=20,
    ),
]

# Speed tip: indicators are recomputed on the whole window every bar. For long
# histories you can pass only the recent tail:
#     from edgelab import Bars
#     recent = Bars(*(a[-300:] for a in window.bars))
# Only do this for indicators whose value does not depend on where the series
# starts (SMA, rolling max/min, Bollinger). EMA, RSI and MACD do depend on it.
