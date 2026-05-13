# Crypto Futures Trading Lab

Systematic crypto futures research workspace for BTC / ETH / SOL / HYPE (Binance USD-M),
including backtests, 5m intraday experiments, parameter tuning, and paper trading.

---

## Strategies

### 1. Martingale — `backtest_martingale.py`

Bollinger Band pullback with loss-side martingale adds and profit-side pyramiding.

| Component | Detail |
|---|---|
| Entry | 1h close touches lower/upper BB band |
| Trend filter | 1d EMA direction |
| Loss adds | Equal-size adds at fixed grid step |
| Profit adds | Pyramid on BB mid-line cross (decreasing size) |
| TP | Fixed margin-rate from average entry |
| SL | Capital-% hard stop |

### 2. Trend Breakout — `backtest_breakout.py`

Donchian channel breakout with ATR-based stops, optimised for total return.

| Component | Detail |
|---|---|
| Entry | 1h close breaks Donchian high/low |
| Trend filter | 1d EMA |
| Filters | ADX strength + volume spike |
| SL / TP | ATR × multiplier / fixed R:R |
| Trailing stop | Optional, ATR-based |

### 3. Calmar-Optimised Breakout — `backtest_calmar.py`

Same breakout mechanics, redesigned for large-capital deployment with conservative drawdown targets.

| Component | Detail |
|---|---|
| Entry | Donchian breakout (immediate or pullback) |
| Sizing | Volatility targeting: `notional = capital × VOL_TARGET / realised_vol` |
| Partial TP | Close 50 % at +1R, trail remainder |
| ADX slope | Require ADX rising over N bars |
| Time exit | Force-close after MAX_HOLD_BARS |
| Optimise | Calmar ratio (CAGR / MaxDD) |

**Best out-of-sample results (per-coin optimal params, $10k initial):**

| Coin | Trades | Win% | Ann Return | Max DD | Sharpe | Calmar |
|---|---|---|---|---|---|---|
| BTC | 776 | 34.9% | 56.1% | 13.7% | 1.63 | 4.08 |
| ETH | 882 | 25.9% | 68.7% | 15.1% | 1.49 | 4.54 |
| SOL | 627 | 31.1% | 51.4% | 11.3% | 1.46 | 4.55 |
| HYPE | 84 | 36.9% | 48.7% | 4.2% | 1.98 | 11.68 |

---

## Project Structure

```
.
├── fetch_btc_history.py          # Fetch OHLCV data from Binance via ccxt
├── backtest/
│   ├── backtest_martingale.py
│   ├── backtest_breakout.py
│   ├── backtest_calmar.py
│   ├── backtest_regime.py
│   └── backtest_5m_vwap.py
├── paper/
│   ├── paper_trade_calmar.py
│   ├── paper_trade_breakout.py
│   ├── paper_trade_martingale.py
│   └── paper_trade_regime.py
├── data/
│   ├── btc_futures_5m.csv
│   ├── btc_futures_1h.csv
│   ├── btc_futures_1d.csv
│   └── ...                       # eth / sol / hype, 5m + 1h + 1d
└── results/
    ├── martingale/
    │   ├── best_params.json
    │   └── best_results_table.txt
    ├── breakout/
    │   ├── best_params.json
    │   └── best_results_table.txt
    ├── calmar/
        ├── best_params.json
        └── best_results_table.txt
    ├── regime/
    └── vwap_5m/
```

---

## Setup

```bash
pip install ccxt pandas numpy tqdm
```

---

## Usage

```bash
# 1. Fetch data (run once, or to refresh)
python fetch_btc_history.py

# 2. Run any strategy — single pass (set AUTO_TUNE = False inside the file)
python backtest/backtest_martingale.py
python backtest/backtest_breakout.py
python backtest/backtest_calmar.py
python backtest/backtest_regime.py
python backtest/backtest_5m_vwap.py

# 3. Run grid search (set AUTO_TUNE = True — default)
#    Uses multiprocessing (spawn), 16 workers by default
python backtest/backtest_calmar.py

# 4. Run in background, log to file
nohup python -u backtest/backtest_calmar.py > results/calmar/run.log 2>&1 &
echo PID=$!
tail -f results/calmar/run.log
```

---

## Key Parameters — `backtest_calmar.py`

| Parameter | Default | Description |
|---|---|---|
| `LEVERAGE` | 3 | Max leverage cap |
| `USE_VOL_TARGET` | True | Size by realised volatility |
| `VOL_TARGET` | 0.20 | Target annual portfolio volatility |
| `DONCHIAN_PERIOD` | 20 | Breakout channel lookback (bars) |
| `ATR_PERIOD` | 14 | ATR smoothing period |
| `SL_MULT` | 1.5 | SL = entry ± ATR × SL_MULT |
| `TP_RR` | 3.0 | TP risk:reward ratio |
| `TREND_EMA_PERIOD` | 200 | Daily EMA for trend filter |
| `ADX_MIN` | 25.0 | Minimum ADX to enter (0 = off) |
| `ADX_SLOPE_BARS` | 3 | Require ADX rising over N bars |
| `USE_PARTIAL_TP` | True | Close 50 % at +1R, trail rest |
| `PARTIAL_TP_R` | 1.0 | First exit at entry + SL×R |
| `USE_PULLBACK` | False | Wait for pullback before entry |
| `MAX_HOLD_BARS` | 0 | Force close after N bars (0 = off) |
| `OPTIMIZE_TARGET` | "calmar" | Ranking metric: calmar / sharpe / return |
| `MIN_TRADE_COUNT` | 30 | Minimum trades to qualify a combo |

---

## Metrics Output

Each run prints a summary table with:

`Trades` · `Win%` · `AvgWin$` · `AvgLoss$` · `Profit Factor` · `Expectancy` · `Ann Return%` · `Max DD%` · `Sharpe` · `Calmar`

Best params per coin are saved to `results/*/best_params.json` and updated incrementally during grid search.

---

## Disclaimer

Research backtest only. Past performance does not guarantee future results.
Futures trading involves significant risk of capital loss.
