---
name: edgelab
description: >-
  Honest backtester. Tests whether a trading rule actually had an edge after
  costs: next-bar-open fills, a lookahead firewall, a randomized-entry null,
  block bootstrap, deflated Sharpe with a permanent trial counter,
  Benjamini-Hochberg FDR, bull/bear regime split and a causality audit. Returns
  PROVEN / LIKELY / NO_EDGE / INSUFFICIENT_SAMPLE with the numbers behind it.
  Daily bars from Yahoo Finance or the user's own CSV. Use when the user says
  "backtest this", "does this strategy work", "test my trading rule", "is this
  edge real", "validate this signal", "check this indicator strategy", or
  pastes a rule like "buy when RSI < 30 and hold 10 days". Never places
  trades and never recommends a position.
argument-hint: "[describe the rule] [on SYMBOLS] [over N years]"
---

# edgelab

edgelab answers one question: **did this rule beat throwing darts at the same
calendar, after costs?** The most valuable answer it gives is a credible
`NO_EDGE`. A backtester that flatters strategies turns money into confidence.

`${CLAUDE_SKILL_DIR}` below is this skill's folder (where this SKILL.md lives).
Everything edgelab writes (price cache, trial counter) goes to `~/.edgelab/`, or
`$EDGELAB_HOME` if set.

## 0. One-time setup

```bash
python3 -m venv ~/.edgelab/venv
~/.edgelab/venv/bin/pip install -q -r "${CLAUDE_SKILL_DIR}/requirements.txt"
~/.edgelab/venv/bin/python "${CLAUDE_SKILL_DIR}/backtest.py" selftest
```

Use `~/.edgelab/venv/bin/python` for every command below. If that venv already
exists, skip the install. `selftest` checks the harness from both sides: pure
noise must not be graded PROVEN, and a mechanically injected edge must be found.
If it fails, stop and tell the user. No verdict from that install can be trusted.

## 1. Pin the rule down

A rule that can be backtested has all of these:

| Part | Example | Default if the user didn't say |
|---|---|---|
| Entry condition, on today's completed bar | RSI(14) closes below 30 | none: ask |
| Side | long | long |
| Exit | hold 10 sessions, or a stop | ask; offer "hold N sessions" |
| Stop (optional) | 2 x ATR(14), or 5% | none |
| Universe | 20 large caps, or a sector | ask; see section 2 |
| Period | 2014 to yesterday | 12 years |

If the entry is vague ("buy dips", "when momentum turns"), ask for a definition
before writing code. Don't invent one silently: the definition *is* the
strategy. Say back the exact rule in one sentence before running it.

## 2. Pick the universe, and say what's wrong with it

- At least **5 symbols** are needed for PROVEN. Prefer 20+ so the sample is
  real. One symbol is a story, not evidence.
- If the user names the stocks, use them, but say plainly that hand-picked
  symbols carry selection bias: they were picked knowing how they did.
- If they don't, propose a broad list fixed *before* looking at results, such
  as the largest 20-30 US stocks across sectors, or a sector's members.
- Survivorship bias always applies: delisted companies are missing. The report
  says so on every run. Don't hide it.

## 3. Write the rules file

Copy `${CLAUDE_SKILL_DIR}/rule_template.py` to the user's working folder (for
example `./edgelab_rules/<short_name>.py`). Replace the example with their rule.
Read the contract at the top of the template. The key points:

- The signal function gets `(window, params)`. `window.c[-1]` is **today's
  close**, and nothing after today exists in the window.
- Return `SignalDecision(side, hold_bars, stop_atr=..., stop_pct=...)` or `None`.
- The fill is the **next bar's open**, never today's close. Don't try to model
  "buy at the close".
- Use `edgelab.indicators` (sma, ema, rsi, atr, macd_hist, bollinger_z,
  williams_r, ultimate_osc, dist_sma, pct_52w, vix_fix, realized_vol). Each
  takes `window.bars`. Or compute your own from the window arrays.
- `warmup_bars` must cover the longest lookback.
- **Never** read files, the network, or any price array from outside the window
  inside a signal function. That is the one way to get future data past the
  firewall, and nothing can catch it.
- Several variants go in one `RULES` list so they're judged as one family. That
  is how the multiple-testing correction is meant to work.

## 4. Run it

```bash
~/.edgelab/venv/bin/python "${CLAUDE_SKILL_DIR}/backtest.py" run \
  --rules ./edgelab_rules/<name>.py \
  --symbols AAPL MSFT NVDA JPM XOM JNJ PG KO WMT HD \
  --years 12 --out ./edgelab_rules/<name>_report.txt
```

Options that matter:

| Flag | Use |
|---|---|
| `--csv a.csv b.csv` | the user's own daily bars, one symbol per file (Date, Open, High, Low, Close, optional Volume). The file name becomes the symbol. The data is still trimmed to `--start/--end` (default: the last 12 years up to yesterday), so set those to cover the file. **Without `--benchmark-csv` (an index or broad-market CSV), the bull/bear check can't run, and PROVEN is unreachable** |
| `--half-spread-bps N` | per-side spread. The default 1.0 is for liquid US large caps. Use 3-8 for mid/small caps and 5-20 for crypto. Thin markets need more |
| `--commission-bps N` | per side, default 0.5 |
| `--min-price N` | default 5. Set it to 0 for FX, or for crypto quoted in small units |
| `--long-only` | ignore short signals |
| `--json f`, `--trades f` | machine-readable output, every trade as CSV |
| `--fast` | 500 resamples instead of 2000, for a quick look. Re-run without it before reporting a verdict |

Notes:
- Yahoo symbols: `BRK-B`, `BTC-USD`, `EURUSD=X` (with `--min-price 0`), `^GSPC`.
  With `--symbols`, the calendar is the US equity session calendar, and SPY is
  the regime benchmark (`--benchmark` to change it).
- `[gate] ... marked untradable` lines are bad bars (high below close, 50%
  jumps with no split, and so on) that the data checks excluded. Mention them
  if there are many.
- A 12-year, 20-symbol run takes from under a minute to a few minutes,
  depending on how heavy the signal function is.
- **The trial counter is permanent.** Every distinct rule+params ever run raises
  the bar for the next one (deflated Sharpe). That's intended. If the user
  tweaks a threshold after seeing a result, that's a new trial. Tell them so.
  Use `--no-count-trials` only when the user explicitly says they're just
  exploring, and mention it in the summary. Never delete `~/.edgelab/trials.json`.
  `backtest.py trials` shows the count.

## 5. Report the result

Lead with the verdict, then translate the numbers. Read them in this order:

1. **`random-entry null: strategy beats X% of shuffles`.** This is the honest
   test of timing. Under 95% means the entries added nothing over random dates
   in the same stocks and period, *even if the return was positive*.
2. **`gross - costs = net`.** A real gross edge that costs eat is a finding.
   Say it.
3. **`why not PROVEN`.** List the failed hurdles in plain words.
4. **Bull/bear split.** Positive in only one regime usually means it's market
   exposure, not an edge. Fewer than 10 trades in either regime counts as a
   failed check ("regime check not applied"), not a skipped one.
5. **Warnings**: censored trades, dropped symbols, skipped regime check.

| Verdict | What to tell the user |
|---|---|
| `PROVEN` | It had an edge in this sample under these costs and passed every hurdle. That's evidence, not a guarantee. Edges decay once found. Suggest testing it out of sample: other years, other symbols |
| `LIKELY` | Suggestive, but it failed something specific. Name it. Don't call it profitable |
| `NO_EDGE` | No better than random entries after costs. Say this clearly, without softening it |
| `INSUFFICIENT_SAMPLE` | Too few trades to say anything. Suggest more symbols or years, not a looser rule |

Hard rules for this step:
- Never turn a verdict into "buy X" or "this will make money". This tool doesn't
  recommend positions.
- Don't loosen thresholds in `edgelab/config.py`, drop hurdles, or shrink the
  universe to get a better verdict. If the user asks, explain that this is the
  overfitting the tool exists to catch. If they insist, they can edit the
  config themselves, but report that a non-default config was used.
- Don't grid-search parameters to find a winner and present the winner. If the
  user wants variants, put them all in one `RULES` list and report the whole
  family, losers included.

## What it does not model

Intrabar path (whether a stop or a target was touched first inside one bar),
intraday strategies, options, leverage, taxes, and liquidity in size. Costs are
assumptions, not fills. If any of these decide the user's strategy, say that
edgelab can't answer it.
