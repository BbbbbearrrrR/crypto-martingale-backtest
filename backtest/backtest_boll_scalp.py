"""
Bollinger Band Scalping Backtest (5m)
======================================
Entry  : 5m close crosses below lower band → long  (only if close > EMA trend)
         5m close crosses above upper band → short (only if close < EMA trend)
SL     : entry ± SL_TP_RATIO × |bb_mid - entry| (fixed R:R)
TP     : USE_PARTIAL_TP=True  → TP1=bb_mid (50%), TP2=opposite band (50%)
         USE_PARTIAL_TP=False → TP=bb_mid (full)
Size   : risk_amount = capital × BASE_RISK
         notional = risk_amount / sl_pct, capped at capital × LEVERAGE
Fee    : 0.05% per side (taker)
Data   : data/{coin}_futures_5m.csv  (3 months)
"""

import os
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")

import json
import sys
import itertools
import multiprocessing as mp
import pandas as pd
import numpy as np
from pathlib import Path
from tqdm import tqdm

_ROOT       = Path(__file__).resolve().parent.parent
DATA_DIR    = _ROOT / "data"
RESULTS_DIR = _ROOT / "results" / "boll_scalp"
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

BEST_PARAMS_FILE = RESULTS_DIR / "best_params.json"

# ── Parameters ────────────────────────────────────────────────────────────────
INITIAL_CAPITAL = 10_000
FEE_RATE        = 0.0005   # 0.05% per side (taker)
LEVERAGE        = 10

BASE_RISK       = 0.01     # fraction of capital risked per trade

BB_PERIOD       = 20       # Bollinger Band rolling window
BB_STD          = 2.0      # standard deviation multiplier

ATR_PERIOD      = 14
SL_TP_RATIO     = 0.5      # SL distance = SL_TP_RATIO × TP1 distance (bb_mid - entry)

TREND_EMA_PERIOD = 200     # 5m EMA for trend direction filter

USE_PARTIAL_TP  = True     # True: TP1=bb_mid(50%) + TP2=opp band(50%); False: TP=bb_mid(full)
MAX_HOLD_BARS   = 48       # max bars to hold before forced exit (48 × 5m = 4h)

# ── Auto-tuning ───────────────────────────────────────────────────────────────
AUTO_TUNE = True

TUNE_SPACE = {
    "LEVERAGE":         [5, 10],
    "BASE_RISK":        [0.01, 0.02],
    "BB_PERIOD":        [10, 20, 40],
    "BB_STD":           [1.5, 2.0, 2.5],
    "SL_TP_RATIO":      [0.3, 0.5, 0.75, 1.0, 1.5],
    "TREND_EMA_PERIOD": [50, 100, 200],
    "USE_PARTIAL_TP":   [True, False],
    "MAX_HOLD_BARS":    [24, 48, 96],
}
# Total combinations per coin: 2x2x3x3x5x3x2x3 = 6480

COINS = [
    ("BTC/USDT:USDT", "btc"),
    ("ETH/USDT:USDT", "eth"),
    ("SOL/USDT:USDT", "sol"),
    ("HYPE/USDT:USDT", "hype"),
    ("SUI/USDT:USDT", "sui"),
]

_RAW_DATA: dict = {}


# ── Indicators ────────────────────────────────────────────────────────────────
def prepare(df_5m: pd.DataFrame) -> pd.DataFrame:
    df = df_5m.copy()

    # Bollinger Bands (shift(1) to avoid look-ahead)
    roll = df["close"].rolling(BB_PERIOD)
    mid  = roll.mean()
    std  = roll.std(ddof=1)
    df["bb_upper"] = (mid + BB_STD * std).shift(1)
    df["bb_lower"] = (mid - BB_STD * std).shift(1)
    df["bb_mid"]   = mid.shift(1)

    # ATR
    prev = df["close"].shift(1)
    tr   = pd.concat([
        df["high"] - df["low"],
        (df["high"] - prev).abs(),
        (df["low"]  - prev).abs(),
    ], axis=1).max(axis=1)
    df["atr"] = tr.ewm(span=ATR_PERIOD, adjust=False).mean()

    # 5m EMA trend filter (shift(1) to avoid look-ahead)
    df["trend_ema"] = df["close"].ewm(span=TREND_EMA_PERIOD, adjust=False).mean().shift(1)
    df["trend_up"]  = df["close"] > df["trend_ema"]

    # Entry signals: close crosses the band (prev bar inside, this bar outside)
    prev_close = df["close"].shift(1)
    df["entry_long"]  = (prev_close >= df["bb_lower"]) & (df["close"] < df["bb_lower"])
    df["entry_short"] = (prev_close <= df["bb_upper"]) & (df["close"] > df["bb_upper"])

    return df


# ── Backtest ──────────────────────────────────────────────────────────────────
def preload_data():
    global _RAW_DATA
    for _, coin in COINS:
        _RAW_DATA[coin] = pd.read_csv(
            DATA_DIR / f"{coin}_futures_5m.csv", index_col=0, parse_dates=True
        )


def run_backtest(symbol: str, coin: str):
    df_5m = _RAW_DATA.get(coin)
    if df_5m is None:
        df_5m = pd.read_csv(DATA_DIR / f"{coin}_futures_5m.csv", index_col=0, parse_dates=True)
    df = prepare(df_5m)

    capital  = float(INITIAL_CAPITAL)
    peak_cap = capital
    trades   = []

    in_trade      = False
    direction     = None
    entry_price   = 0.0
    sl_price      = 0.0
    tp1_price     = 0.0
    tp2_price     = 0.0
    notional      = 0.0
    notional_rem  = 0.0   # remaining after partial TP
    partial_done  = False
    bars_held     = 0

    warmup = max(BB_PERIOD, ATR_PERIOD, TREND_EMA_PERIOD) + 2

    for i in range(warmup, len(df)):
        row = df.iloc[i]
        ts  = df.index[i]

        if capital <= 10:
            break

        # ── Exit ─────────────────────────────────────────────────────────────
        if in_trade:
            bars_held += 1

            hit_tp1 = (not partial_done and
                       (row["high"] >= tp1_price if direction == "long" else row["low"] <= tp1_price))
            hit_tp2 = (row["high"] >= tp2_price if direction == "long" else row["low"] <= tp2_price)
            hit_sl  = (row["low"]  <= sl_price  if direction == "long" else row["high"] >= sl_price)
            expired = bars_held >= MAX_HOLD_BARS

            # Partial TP1 — close 50%
            if hit_tp1 and USE_PARTIAL_TP and not partial_done:
                half = notional_rem * 0.5
                pct  = ((tp1_price - entry_price) / entry_price if direction == "long"
                        else (entry_price - tp1_price) / entry_price)
                pnl  = half * pct - half * FEE_RATE
                capital += pnl
                peak_cap = max(peak_cap, capital)
                trades.append({
                    "exit_time": ts, "direction": direction,
                    "entry_price": round(entry_price, 6), "exit_price": round(tp1_price, 6),
                    "notional": round(half, 4), "exit_reason": "TP1",
                    "pnl_usdt": round(pnl, 4), "capital": round(capital, 4),
                    "drawdown": round((peak_cap - capital) / peak_cap, 6),
                })
                notional_rem -= half
                partial_done  = True
                sl_price      = entry_price  # move SL to breakeven

            # Full exit: TP2, SL, or timeout
            if in_trade and (hit_tp2 or hit_sl or expired):
                if hit_tp2:
                    exit_price, exit_reason = tp2_price, "TP2"
                elif hit_sl:
                    exit_price, exit_reason = sl_price, "SL"
                else:
                    exit_price, exit_reason = row["close"], "TIMEOUT"

                pct     = ((exit_price - entry_price) / entry_price if direction == "long"
                           else (entry_price - exit_price) / entry_price)
                pnl     = notional_rem * pct - notional_rem * FEE_RATE * 2
                pnl     = max(pnl, -capital)
                capital += pnl
                peak_cap = max(peak_cap, capital)
                trades.append({
                    "exit_time": ts, "direction": direction,
                    "entry_price": round(entry_price, 6), "exit_price": round(exit_price, 6),
                    "notional": round(notional_rem, 4), "exit_reason": exit_reason,
                    "pnl_usdt": round(pnl, 4), "capital": round(capital, 4),
                    "drawdown": round((peak_cap - capital) / peak_cap, 6),
                })
                in_trade     = False
                partial_done = False

        # ── Entry ─────────────────────────────────────────────────────────────
        if not in_trade:
            atr = row["atr"]
            if pd.isna(atr) or atr <= 0:
                continue
            if pd.isna(row["bb_lower"]) or pd.isna(row["bb_upper"]):
                continue

            go_long  = bool(row["entry_long"])  and bool(row["trend_up"])
            go_short = bool(row["entry_short"]) and not bool(row["trend_up"])

            if go_long or go_short:
                direction   = "long" if go_long else "short"
                entry_price = row["close"]

                # SL: fixed R:R based on TP1 distance
                if direction == "long":
                    tp1_price = row["bb_mid"]
                    tp2_price = row["bb_upper"]
                    sl_price  = entry_price - abs(tp1_price - entry_price) * SL_TP_RATIO
                else:
                    tp1_price = row["bb_mid"]
                    tp2_price = row["bb_lower"]
                    sl_price  = entry_price + abs(tp1_price - entry_price) * SL_TP_RATIO

                # TP must be on the correct side
                if direction == "long"  and (tp1_price <= entry_price or tp2_price <= entry_price):
                    continue
                if direction == "short" and (tp1_price >= entry_price or tp2_price >= entry_price):
                    continue
                # SL must be on the losing side
                if direction == "long"  and sl_price >= entry_price:
                    continue
                if direction == "short" and sl_price <= entry_price:
                    continue

                sl_pct = abs(entry_price - sl_price) / entry_price
                if sl_pct < 1e-6:
                    continue

                risk_amt     = capital * BASE_RISK
                notional     = min(risk_amt / sl_pct, capital * LEVERAGE)
                notional_rem = notional
                partial_done = False
                bars_held    = 0
                in_trade     = True

    if not trades:
        return None, pd.DataFrame()

    t_df = pd.DataFrame(trades)
    wins = (t_df["pnl_usdt"] > 0).sum()
    total = len(t_df)
    win_rate = wins / total if total else 0
    total_return = (capital - INITIAL_CAPITAL) / INITIAL_CAPITAL
    max_dd = t_df["drawdown"].max() if "drawdown" in t_df.columns else 0
    calmar = total_return / max_dd if max_dd > 1e-6 else 0
    avg_pnl = t_df["pnl_usdt"].mean()
    profit_factor = (t_df.loc[t_df["pnl_usdt"] > 0, "pnl_usdt"].sum() /
                     abs(t_df.loc[t_df["pnl_usdt"] < 0, "pnl_usdt"].sum() + 1e-9))

    metrics = {
        "symbol":        symbol,
        "coin":          coin,
        "total_trades":  total,
        "win_rate":      round(win_rate, 4),
        "total_return":  round(total_return, 4),
        "max_drawdown":  round(max_dd, 4),
        "calmar":        round(calmar, 4),
        "avg_pnl":       round(avg_pnl, 4),
        "profit_factor": round(profit_factor, 4),
        "final_capital": round(capital, 2),
    }
    return metrics, t_df


# ── Helpers ───────────────────────────────────────────────────────────────────
def current_params() -> dict:
    return {
        "LEVERAGE": LEVERAGE, "BASE_RISK": BASE_RISK,
        "BB_PERIOD": BB_PERIOD, "BB_STD": BB_STD,
        "ATR_PERIOD": ATR_PERIOD,
        "SL_TP_RATIO": SL_TP_RATIO,
        "TREND_EMA_PERIOD": TREND_EMA_PERIOD,
        "USE_PARTIAL_TP": USE_PARTIAL_TP,
        "MAX_HOLD_BARS": MAX_HOLD_BARS,
    }


def _apply_params(p: dict):
    g = globals()
    for k, v in p.items():
        g[k] = v


def _worker_init():
    global _RAW_DATA
    for _, coin in COINS:
        _RAW_DATA[coin] = pd.read_csv(
            DATA_DIR / f"{coin}_futures_5m.csv", index_col=0, parse_dates=True
        )


def _tune_worker(p: dict):
    coins_filter = p.pop("_coins", None)
    _apply_params(p)
    active = coins_filter if coins_filter is not None else COINS
    coin_scores = {}
    for symbol, coin in active:
        metrics, _ = run_backtest(symbol, coin)
        if metrics is None or metrics["total_trades"] < 20 or metrics["win_rate"] < 0.45:
            coin_scores[coin] = float("-inf")
        else:
            coin_scores[coin] = metrics["calmar"]
    valid = [s for s in coin_scores.values() if s != float("-inf")]
    avg_score = float(np.mean(valid)) if valid else float("-inf")
    return p, avg_score, coin_scores, current_params()


def _save_best_results_table():
    if not BEST_PARAMS_FILE.exists():
        return
    best = json.loads(BEST_PARAMS_FILE.read_text())
    rows = []
    for symbol, coin in COINS:
        entry = best.get(coin)
        if not entry:
            continue
        _apply_params(entry["params"])
        metrics, t_df = run_backtest(symbol, coin)
        if metrics:
            rows.append({
                "Coin":         coin.upper(),
                "Trades":       metrics["total_trades"],
                "Win%":         f"{metrics['win_rate']*100:.1f}%",
                "Return%":      f"{metrics['total_return']*100:.1f}%",
                "MaxDD%":       f"{metrics['max_drawdown']*100:.1f}%",
                "Calmar":       f"{metrics['calmar']:.2f}",
                "AvgPnL":       f"${metrics['avg_pnl']:.2f}",
                "ProfitFactor": f"{metrics['profit_factor']:.2f}",
            })
        if not t_df.empty:
            t_df.to_csv(RESULTS_DIR / f"{coin}_boll_scalp.csv", index=False)
    if rows:
        table = pd.DataFrame(rows).to_string(index=False)
        print(f"\n{table}")
        (RESULTS_DIR / "best_results_table.txt").write_text(table)


# ── auto_tune ─────────────────────────────────────────────────────────────────
def auto_tune(coins=None):
    active_coins = coins if coins is not None else COINS
    keys   = list(TUNE_SPACE.keys())
    combos = [{**dict(zip(keys, c)), "_coins": active_coins}
              for c in itertools.product(*[TUNE_SPACE[k] for k in keys])]
    total  = len(combos)
    n_workers = min(16, max(1, mp.cpu_count() - 1))

    print(f"\n{'='*65}")
    print(f"  BOLL-SCALP AUTO-TUNE  |  {total} combos  |  {len(active_coins)} coins")
    print(f"  Workers               |  {n_workers} parallel processes")
    print(f"{'='*65}")

    best: dict = json.loads(BEST_PARAMS_FILE.read_text()) if BEST_PARAMS_FILE.exists() else {}

    ctx  = mp.get_context("spawn")
    done = 0
    pbar = tqdm(total=total, desc="BOLL-SCALP-TUNE", unit="combo", ncols=95)
    with ctx.Pool(processes=n_workers, initializer=_worker_init) as pool:
        for p, avg_score, coin_scores, snapped_params in \
                pool.imap_unordered(_tune_worker, combos, chunksize=8):
            done += 1
            pbar.update(1)

            updated = []
            for coin, sc in coin_scores.items():
                if sc == float("-inf"):
                    continue
                prev_sc = best.get(coin, {}).get("best_score", float("-inf"))
                if sc > prev_sc:
                    best[coin] = {
                        "best_score": round(sc, 6),
                        "params":     snapped_params,
                    }
                    updated.append(f"{coin.upper()} calmar={sc:.3f}")

            if updated:
                BEST_PARAMS_FILE.write_text(json.dumps(best, indent=2))
                pbar.write(
                    f"  [{done:>{len(str(total))}}/{total}]  avg {avg_score:.3f}  ★ {', '.join(updated)}"
                    f"  | bb={p.get('BB_PERIOD')} std={p.get('BB_STD')}"
                    f" sl={p.get('SL_MULT')} hold={p.get('MAX_HOLD_BARS')}"
                )
            elif done % 100 == 0:
                pbar.write(f"  [{done:>{len(str(total))}}/{total}]  avg {avg_score:.3f}  (no improvement)")
    pbar.close()
    print(f"\nTuning complete. Results in {BEST_PARAMS_FILE}")
    _save_best_results_table()


# ── Main ──────────────────────────────────────────────────────────────────────
def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--coin", type=str, default=None)
    args = parser.parse_args()

    coins_filter = None
    if args.coin:
        wanted = {c.strip().lower() for c in args.coin.split(",")}
        coins_filter = [(s, c) for s, c in COINS if c in wanted]

    if AUTO_TUNE:
        auto_tune(coins=coins_filter)
    else:
        preload_data()
        active = coins_filter if coins_filter else COINS
        for symbol, coin in active:
            metrics, t_df = run_backtest(symbol, coin)
            if metrics:
                print(metrics)
            if not t_df.empty:
                t_df.to_csv(RESULTS_DIR / f"{coin}_boll_scalp.csv", index=False)


if __name__ == "__main__":
    main()
