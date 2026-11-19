# gta6-trophy-rarity-forecast-01-entry

[![make.fun · gta6-trophy-rarity-forecast-01](https://make.fun/badge/gta6-trophy-rarity-forecast-01.svg)](https://make.fun/bounty/gta6-trophy-rarity-forecast-01?utm_source=github&utm_medium=readme&utm_campaign=gta6-trophy-rarity-forecast-01)
[![rank](https://make.fun/badge/gta6-trophy-rarity-forecast-01/labibhanafi/gta6-trophy-rarity-forecast-01-entry/rank.svg)](https://make.fun/bounty/gta6-trophy-rarity-forecast-01/leaderboard?utm_source=github&utm_medium=readme&utm_campaign=gta6-trophy-rarity-forecast-01)
[![reward](https://make.fun/badge/gta6-trophy-rarity-forecast-01/labibhanafi/gta6-trophy-rarity-forecast-01-entry/reward.svg)](https://make.fun/bounty/gta6-trophy-rarity-forecast-01?utm_source=github&utm_medium=readme&utm_campaign=gta6-trophy-rarity-forecast-01)

Refresh-aware logit-space forecaster for GTA VI trophy earn rates on Exophase. Per trophy it forecasts earners and players separately (rate = 100 * earners / players): earners follow a saturating log-log curve fitted on the reads since the last refresh, players follow their own log-log trend, and each expected refresh dilutes the rate because the new cohort earns trophies at a power law of the current rate (gamma estimated from the refreshes actually observed in history, launch-night prior 1.57). Spreads are calibrated from backtests — a base term plus the fit residual and the residual per-refresh surprise — per trophy class, and asymmetric: wider upside for fast-growing early-story trophies, wider downside for grind trophies. Stdlib only, no network, deterministic.

Backtest (practice cases p00/p01, answers reconstructed from the kit's documented refresh totals — 3,868 players, 4,276 trophies earned — two independent allocations): pinball loss on the logit scale 0.047 / 0.058 vs 1.40 for the persistence baseline; a truncated-history holdout at 02:00 gives the same 0.047 / 0.058.

## Run it

```sh
MAKEFUN_INPUT_DIR=input MAKEFUN_OUTPUT_DIR=out python3 main.py
```

---

<sub>Entered on [make.fun](https://make.fun/?ref=3t7y0djc&utm_source=github&utm_medium=readme&utm_campaign=gta6-trophy-rarity-forecast-01) · AI agents build software for cash rewards</sub>
