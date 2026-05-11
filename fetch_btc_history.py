"""
Fetch perpetual futures OHLCV history from Binance for BTC, ETH, SOL.
Timeframes:
  1d / 1h  — past 5 years  (trend filter data)
  5m       — past 2 years  (scalping backtest data)
Output: data/{coin}_futures_{tf}.csv
"""

import ccxt
import pandas as pd
import time
from datetime import datetime, timezone, timedelta
from pathlib import Path

DATA_DIR = Path("data")
DATA_DIR.mkdir(exist_ok=True)

EXCHANGE_ID = "binance"
LIMIT = 1000                       # candles per request (Binance max)

# symbol -> coin label for filenames
SYMBOLS = {
    # "BTC/USDT:USDT": "btc",
    # "ETH/USDT:USDT": "eth",
    # "SOL/USDT:USDT": "sol",
    "HYPE/USDT:USDT": "hype",
}

# timeframe -> how many years back to fetch  (0 = skip)
TIMEFRAME_YEARS = {
    "1d": 0,
    "1h": 0,
    "5m": 2,
}

exchange = ccxt.binanceusdm({
    "enableRateLimit": True,
})

now_utc = datetime.now(timezone.utc)


def since_ms_for(years: int) -> int:
    return int((now_utc - timedelta(days=365 * years)).timestamp() * 1000)


def fetch_ohlcv_full(symbol: str, timeframe: str, since_ms: int) -> pd.DataFrame:
    """Paginate through all candles from since_ms to now."""
    since_date = datetime.fromtimestamp(since_ms / 1000, tz=timezone.utc).date()
    print(f"\n[{symbol}] Fetching {timeframe} candles from {since_date} ...")

    all_candles = []
    current_since = since_ms
    page = 0

    while True:
        page += 1
        candles = exchange.fetch_ohlcv(symbol, timeframe=timeframe, since=current_since, limit=LIMIT)
        if not candles:
            break

        all_candles.extend(candles)
        last_ts = candles[-1][0]
        fetched_dt = datetime.fromtimestamp(last_ts / 1000, tz=timezone.utc)
        print(f"  page {page:3d} | {len(candles):4d} candles | up to {fetched_dt.strftime('%Y-%m-%d %H:%M')}", end="\r")

        if len(candles) < LIMIT:
            break

        current_since = last_ts + 1
        time.sleep(exchange.rateLimit / 1000)

    print()

    df = pd.DataFrame(all_candles, columns=["timestamp", "open", "high", "low", "close", "volume"])
    df["datetime"] = pd.to_datetime(df["timestamp"], unit="ms", utc=True)
    df = df.drop(columns=["timestamp"]).set_index("datetime")
    df = df[~df.index.duplicated(keep="last")].sort_index()
    return df


def main():
    print(f"Exchange : {EXCHANGE_ID}")
    print(f"Today    : {now_utc.date()}")

    # fetch only timeframes with TIMEFRAME_YEARS > 0
    fetch_timeframes = [tf for tf, yr in TIMEFRAME_YEARS.items() if yr > 0]

    for symbol, coin in SYMBOLS.items():
        print(f"\n{'='*50}")
        print(f"Symbol: {symbol}")
        for timeframe in fetch_timeframes:
            years = TIMEFRAME_YEARS[timeframe]
            if years == 0:
                print(f"  [{timeframe}] skipped (TIMEFRAME_YEARS=0)")
                continue
            df = fetch_ohlcv_full(symbol, timeframe, since_ms_for(years))
            fname = f"{coin}_futures_{timeframe}.csv"
            out_path = DATA_DIR / fname
            df.to_csv(out_path)
            print(f"  Saved {len(df):,} rows → {out_path}  ({df.index[0].date()} ~ {df.index[-1].date()})")

    print("\nDone.")


if __name__ == "__main__":
    main()


if __name__ == "__main__":
    main()
