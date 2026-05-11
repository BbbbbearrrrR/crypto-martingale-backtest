# Crypto Martingale Backtest

Perpetual futures martingale backtest on BTC/ETH/SOL/HYPE using Binance USD-M data.

## Strategy

- **Entry**: Bollinger Bands pullback confirmation (1h bars)
- **Trend filter**: 1d EMA direction filter
- **Loss-side adds**: Equal-size martingale adds at fixed grid step (`GRID_STEP_RATE`)
- **Profit-side adds**: Pyramid adds on BB mid-line cross while in profit (decreasing size)
- **Take profit**: Fixed margin-rate TP from average entry price
- **Stop loss**: Capital-% hard SL (analytically back-solved)

## Files

| File | Description |
|------|-------------|
| `fetch_btc_history.py` | Fetch OHLCV data from Binance via ccxt |
| `backtest_martingale.py` | Full backtest with auto-tune grid search |
| `results/best_params.json` | Best hyperparameters per coin (auto-updated) |

## Setup

```bash
pip install ccxt pandas numpy tqdm
```

## Usage

```bash
# Fetch data first
python fetch_btc_history.py

# Run backtest (single run)
# Set AUTO_TUNE = False in backtest_martingale.py
python backtest_martingale.py

# Run grid search
# Set AUTO_TUNE = True in backtest_martingale.py
python backtest_martingale.py
```

## Key Parameters

| Parameter | Default | Description |
|-----------|---------|-------------|
| `LEVERAGE` | 50 | Futures leverage |
| `BASE_RISK` | 0.05 | Capital fraction per level |
| `MAX_LEVELS` | 10 | Max loss-side martingale adds |
| `MAX_PYRAMID_LEVELS` | 10 | Max profit-side pyramid adds |
| `GRID_STEP_RATE` | 0.02 | Price drop % to trigger next add |
| `TP_MARGIN_RATE` | 1.00 | TP at N× total margin profit |
| `SL_CAPITAL_RATE` | 0.50 | SL when loss = N× total capital |
| `BOLL_PERIOD` | 20 | Bollinger Bands period |
| `BOLL_STD` | 2.0 | Bollinger Bands std multiplier |
| `TREND_EMA_PERIOD` | 20 | Daily EMA period for trend filter |

## Disclaimer

This is a research backtest. Past performance does not guarantee future results. Martingale strategies carry significant risk of total capital loss.
