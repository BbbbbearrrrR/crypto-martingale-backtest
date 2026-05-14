"""
Fetch perpetual futures 5m OHLCV history from Binance.
Coins    : BTC, ETH, SOL, HYPE, SUI
Range    : past 3 months
Output   : data/{coin}_futures_5m.csv
"""

import ccxt
import pandas as pd
import time
from datetime import datetime, timezone, timedelta
from pathlib import Path

DATA_DIR = Path("data")
DATA_DIR.mkdir(exist_ok=True)

EXCHANGE_ID = "binance"
LIMIT = 1000  # candles per request (Binance max)

SYMBOLS = {
    "BTC/USDT:USDT": "btc",
    "ETH/USDT:USDT": "eth",
    "SOL/USDT:USDT": "sol",
    "HYPE/USDT:USDT": "hype",
    "SUI/USDT:USDT": "sui",
}

MONTHS = 3

exchange = ccxt.binanceusdm({"enableRateLimit": True})

now_utc = datetime.now(timezone.utc)
since_ms = int((now_utc - timedelta(days=30 * MONTHS)).timestamp() * 1000)


def fetch_ohlcv_full(symbol: str) -> pd.DataFrame:
    since_date = datetime.fromtimestamp(since_ms / 1000, tz=timezone.utc).date()
    print(f"\n[{symbol}] Fetching 5m candles from {since_date} ...")

    all_candles = []
    current_since = since_ms
    page = 0

    while True:
        page += 1
        candles = exchange.fetch_ohlcv(symbol, timeframe="5m", since=current_since, limit=LIMIT)
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
    print(f"Range    : last {MONTHS} months")

    for symbol, coin in SYMBOLS.items():
        print(f"\n{'='*50}")
        print(f"Symbol: {symbol}")
        df = fetch_ohlcv_full(symbol)
        out_path = DATA_DIR / f"{coin}_futures_5m.csv"
        df.to_csv(out_path)
        print(f"  Saved {len(df):,} rows → {out_path}  ({df.index[0].date()} ~ {df.index[-1].date()})")

    print("\nDone.")


if __name__ == "__main__":
    main()
