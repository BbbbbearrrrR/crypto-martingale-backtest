"""
Sweep-Divergence-FVG Strategy (1h)
=====================================
Three left-side reversal concepts combined into one entry signal:

1. LIQUIDITY SWEEP  : Current 1h bar pokes through a recent N-bar high/low
   (sweeping stop-loss orders), then closes back inside the range.
   This "rejection" candle is the primary trigger.

2. MACD DIVERGENCE  : At the sweep bar, the MACD histogram is less extreme
   than it was DIV_LOOKBACK bars ago — even though price extended further.
   Confirms that momentum is weakening (背驰 signal).

3. FAIR VALUE GAP   : A 3-bar price imbalance (candle[i-2].high < candle[i].low
   for bullish, reversed for bearish). If USE_FVG_FILTER=True, only enter
   when a recent FVG exists within FVG_MAX_AGE bars — it acts as a magnet
   that pulls price back, adding conviction to the reversal.

Entry  : Close of sweep bar (all three conditions met / required ones met)
SL     : Beyond sweep bar's extreme + ATR × SL_ATR_MULT
TP     : entry ± TP_RR × sl_dist
Size   : risk_amount = capital × BASE_RISK
         notional = risk_amount / sl_pct, capped at capital × LEVERAGE
Fee    : 0.05% per side (taker)
Data   : data/{coin}_futures_1h.csv
"""

import os
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")

import json
import itertools
import multiprocessing as mp
import pandas as pd
import numpy as np
from pathlib import Path
from tqdm import tqdm

_ROOT            = Path(__file__).resolve().parent.parent
DATA_DIR         = _ROOT / "data"
RESULTS_DIR      = _ROOT / "results" / "sweep_div"
RESULTS_DIR.mkdir(parents=True, exist_ok=True)
BEST_PARAMS_FILE = RESULTS_DIR / "best_params.json"

# ── Parameters ────────────────────────────────────────────────────────────────
INITIAL_CAPITAL = 10_000
FEE_RATE        = 0.0005      # 0.05% per side (taker)
LEVERAGE        = 5
BASE_RISK       = 0.01        # fraction of capital risked per trade

# Sweep
SWEEP_PERIOD    = 20          # N-bar lookback for liquidity level

# ATR
ATR_PERIOD      = 14
SL_ATR_MULT     = 0.5         # SL = sweep extreme + ATR × SL_ATR_MULT

# MACD divergence
MACD_FAST       = 12
MACD_SLOW       = 26
MACD_SIGNAL     = 9
DIV_LOOKBACK    = 10          # bars back to compare histogram for divergence
USE_DIV_FILTER  = True        # require divergence confirmation

# Fair Value Gap
USE_FVG_FILTER  = True        # require recent FVG near entry
FVG_MAX_AGE     = 5           # max bars back a FVG can be and still count

# Exit
TP_RR           = 3.0         # reward:risk ratio
MAX_HOLD_BARS   = 48          # timeout in bars (48h = 2 days)

# ── Auto-tuning ───────────────────────────────────────────────────────────────
AUTO_TUNE = True

TUNE_SPACE = {
    # ── Sweep sensitivity ─────────────────────────────────────────────────────
    "SWEEP_PERIOD":   [10, 20, 40],
    # ── SL placement ─────────────────────────────────────────────────────────
    "SL_ATR_MULT":    [0.5, 1.0, 1.5],
    # ── Take profit ───────────────────────────────────────────────────────────
    "TP_RR":          [2.0, 3.0, 5.0],
    # ── Divergence filter ─────────────────────────────────────────────────────
    "USE_DIV_FILTER": [True, False],
    "DIV_LOOKBACK":   [5, 10],            # drop 20 (too slow for reversal signal)
    # ── FVG filter ────────────────────────────────────────────────────────────
    "USE_FVG_FILTER": [True, False],
    "FVG_MAX_AGE":    [3, 10],            # drop 5 (midpoint)
    # ── Exit ──────────────────────────────────────────────────────────────────
    "MAX_HOLD_BARS":  [24, 48, 96],
    # ── Position sizing ───────────────────────────────────────────────────────
    "LEVERAGE":       [3, 5, 10],
    "BASE_RISK":      [0.01, 0.02, 0.05],
    # ── Fixed: MACD_FAST=12, MACD_SLOW=26, MACD_SIGNAL=9
}
# Total: 3×3×3×2×2×2×2×3×3×3 = 11,664 combos

COINS = [
    ("BTC/USDT:USDT", "btc"),
    ("ETH/USDT:USDT", "eth"),
    ("SOL/USDT:USDT", "sol"),
    ("HYPE/USDT:USDT", "hype"),
    ("SUI/USDT:USDT", "sui"),
]

_RAW_DATA: dict = {}


# ── Helpers ───────────────────────────────────────────────────────────────────
def _g(name):
    return globals()[name]


def current_params() -> dict:
    keys = [
        "LEVERAGE", "BASE_RISK", "SWEEP_PERIOD", "ATR_PERIOD",
        "SL_ATR_MULT", "TP_RR",
        "MACD_FAST", "MACD_SLOW", "MACD_SIGNAL",
        "USE_DIV_FILTER", "DIV_LOOKBACK",
        "USE_FVG_FILTER", "FVG_MAX_AGE",
        "MAX_HOLD_BARS",
    ]
    return {k: _g(k) for k in keys}


def _apply_params(p: dict):
    g = globals()
    for k, v in p.items():
        if k in g:
            g[k] = v


# ── Indicators ────────────────────────────────────────────────────────────────
def prepare(df_1h: pd.DataFrame) -> pd.DataFrame:
    df = df_1h.copy()

    # ── ATR ──────────────────────────────────────────────────────────────────
    prev = df["close"].shift(1)
    tr   = pd.concat([
        df["high"] - df["low"],
        (df["high"] - prev).abs(),
        (df["low"]  - prev).abs(),
    ], axis=1).max(axis=1)
    df["atr"] = tr.ewm(span=ATR_PERIOD, adjust=False).mean()

    # ── Liquidity levels (shift to avoid look-ahead) ──────────────────────────
    df["sweep_high"] = df["high"].rolling(SWEEP_PERIOD).max().shift(1)
    df["sweep_low"]  = df["low"].rolling(SWEEP_PERIOD).min().shift(1)

    # Sweep signal: bar pokes through extreme, closes back inside
    # Long sweep: low < N-bar low but close >= N-bar low
    df["sweep_long"]  = (df["low"]  < df["sweep_low"])  & (df["close"] >= df["sweep_low"])
    # Short sweep: high > N-bar high but close <= N-bar high
    df["sweep_short"] = (df["high"] > df["sweep_high"]) & (df["close"] <= df["sweep_high"])

    # ── MACD ──────────────────────────────────────────────────────────────────
    ema_fast         = df["close"].ewm(span=MACD_FAST, adjust=False).mean()
    ema_slow         = df["close"].ewm(span=MACD_SLOW, adjust=False).mean()
    macd_line        = ema_fast - ema_slow
    signal_line      = macd_line.ewm(span=MACD_SIGNAL, adjust=False).mean()
    df["macd_hist"]  = (macd_line - signal_line).shift(1)  # shift to avoid look-ahead

    # ── MACD Divergence ───────────────────────────────────────────────────────
    # Bullish divergence: histogram is IMPROVING vs DIV_LOOKBACK bars ago
    # (price at/near low, but momentum less negative → buying pressure building)
    hist_prev          = df["macd_hist"].shift(DIV_LOOKBACK)
    df["div_bull"]     = df["macd_hist"] > hist_prev   # histogram rising despite low price
    df["div_bear"]     = df["macd_hist"] < hist_prev   # histogram falling despite high price

    # ── Fair Value Gap ────────────────────────────────────────────────────────
    # Bullish FVG at bar i: bar[i-1].low > bar[i-3].high (all shifted for look-ahead safety)
    # i.e., using shifted: df["low"].shift(1) > df["high"].shift(3)
    df["fvg_bull"]      = df["low"].shift(1)  > df["high"].shift(3)
    df["fvg_bull_lo"]   = df["high"].shift(3)   # bottom of gap
    df["fvg_bull_hi"]   = df["low"].shift(1)    # top of gap

    df["fvg_bear"]      = df["high"].shift(1) < df["low"].shift(3)
    df["fvg_bear_lo"]   = df["high"].shift(1)   # bottom of gap
    df["fvg_bear_hi"]   = df["low"].shift(3)    # top of gap

    return df


def _find_recent_fvg(df: pd.DataFrame, i: int, direction: str) -> bool:
    """Check if a valid FVG exists within the last FVG_MAX_AGE bars."""
    start = max(0, i - FVG_MAX_AGE)
    for j in range(start, i + 1):
        row = df.iloc[j]
        if direction == "long"  and bool(row["fvg_bull"]):
            lo = row["fvg_bull_lo"]
            hi = row["fvg_bull_hi"]
            if pd.notna(lo) and pd.notna(hi) and hi > lo:
                return True
        elif direction == "short" and bool(row["fvg_bear"]):
            lo = row["fvg_bear_lo"]
            hi = row["fvg_bear_hi"]
            if pd.notna(lo) and pd.notna(hi) and hi > lo:
                return True
    return False


# ── Backtest ──────────────────────────────────────────────────────────────────
def preload_data():
    global _RAW_DATA
    for _, coin in COINS:
        _RAW_DATA[coin] = pd.read_csv(
            DATA_DIR / f"{coin}_futures_1h.csv", index_col=0, parse_dates=True
        )


def run_backtest(symbol: str, coin: str):
    df_1h = _RAW_DATA.get(coin)
    if df_1h is None:
        df_1h = pd.read_csv(
            DATA_DIR / f"{coin}_futures_1h.csv", index_col=0, parse_dates=True
        )
    df = prepare(df_1h)

    capital   = float(INITIAL_CAPITAL)
    peak_cap  = capital
    trades    = []

    in_trade    = False
    direction   = None
    entry_price = 0.0
    sl_price    = 0.0
    tp_price    = 0.0
    notional    = 0.0
    bars_held   = 0

    warmup = max(SWEEP_PERIOD, MACD_SLOW + MACD_SIGNAL, ATR_PERIOD, FVG_MAX_AGE) + 5

    for i in range(warmup, len(df)):
        row = df.iloc[i]
        ts  = df.index[i]

        if capital <= 10:
            break

        # ── Exit ─────────────────────────────────────────────────────────────
        if in_trade:
            bars_held += 1

            hit_tp  = (row["high"] >= tp_price if direction == "long"
                       else row["low"]  <= tp_price)
            hit_sl  = (row["low"]  <= sl_price if direction == "long"
                       else row["high"] >= sl_price)
            expired = MAX_HOLD_BARS > 0 and bars_held >= MAX_HOLD_BARS

            if hit_tp or hit_sl or expired:
                if hit_tp:
                    exit_price, exit_reason = tp_price, "TP"
                elif hit_sl:
                    exit_price, exit_reason = sl_price, "SL"
                else:
                    exit_price, exit_reason = row["close"], "TIMEOUT"

                pct     = ((exit_price - entry_price) / entry_price if direction == "long"
                           else (entry_price - exit_price) / entry_price)
                pnl     = notional * pct - notional * FEE_RATE * 2
                pnl     = max(pnl, -capital)
                capital += pnl
                peak_cap = max(peak_cap, capital)
                trades.append({
                    "exit_time":   ts,
                    "direction":   direction,
                    "entry_price": round(entry_price, 6),
                    "exit_price":  round(exit_price, 6),
                    "notional":    round(notional, 4),
                    "exit_reason": exit_reason,
                    "pnl_usdt":    round(pnl, 4),
                    "capital":     round(capital, 4),
                    "drawdown":    round((peak_cap - capital) / peak_cap, 6),
                })
                in_trade = False

        # ── Entry ─────────────────────────────────────────────────────────────
        if not in_trade:
            atr = row["atr"]
            if pd.isna(atr) or atr <= 0:
                continue

            sweep_l = bool(row["sweep_long"])
            sweep_s = bool(row["sweep_short"])

            if not sweep_l and not sweep_s:
                continue

            direction = "long" if sweep_l else "short"

            # ── Divergence filter ──────────────────────────────────────────
            if USE_DIV_FILTER:
                div_ok = bool(row["div_bull"]) if direction == "long" else bool(row["div_bear"])
                if not div_ok:
                    direction = None
                    continue

            # ── FVG filter ─────────────────────────────────────────────────
            if USE_FVG_FILTER:
                if not _find_recent_fvg(df, i, direction):
                    direction = None
                    continue

            # ── Position sizing ────────────────────────────────────────────
            entry_price = row["close"]

            if direction == "long":
                sl_price = row["low"] - atr * SL_ATR_MULT
            else:
                sl_price = row["high"] + atr * SL_ATR_MULT

            sl_dist = abs(entry_price - sl_price)
            if sl_dist < 1e-8:
                continue

            # SL must be on the losing side
            if direction == "long"  and sl_price >= entry_price:
                continue
            if direction == "short" and sl_price <= entry_price:
                continue

            tp_price = (entry_price + sl_dist * TP_RR if direction == "long"
                        else entry_price - sl_dist * TP_RR)

            sl_pct    = sl_dist / entry_price
            risk_amt  = capital * BASE_RISK
            notional  = min(risk_amt / sl_pct, capital * LEVERAGE)
            if notional < 1:
                continue

            # Entry fee
            capital  -= notional * FEE_RATE
            peak_cap  = max(peak_cap, capital)
            in_trade  = True
            bars_held = 0

    if not trades:
        return None, pd.DataFrame()

    t_df = pd.DataFrame(trades)
    wins         = (t_df["pnl_usdt"] > 0).sum()
    total        = len(t_df)
    win_rate     = wins / total if total else 0
    total_return = (capital - INITIAL_CAPITAL) / INITIAL_CAPITAL
    max_dd       = t_df["drawdown"].max() if "drawdown" in t_df.columns else 0

    # Annualised Calmar
    try:
        first_ts  = pd.Timestamp(t_df["exit_time"].iloc[0])
        last_ts   = pd.Timestamp(t_df["exit_time"].iloc[-1])
        span_days = max((last_ts - first_ts).days, 1)
    except Exception:
        span_days = 365
    ann_ret = (capital / INITIAL_CAPITAL) ** (365.25 / span_days) - 1
    calmar  = ann_ret / max_dd if max_dd > 1e-6 else float("inf")

    avg_pnl       = t_df["pnl_usdt"].mean()
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
    }
    return metrics, t_df


# ── Worker init ───────────────────────────────────────────────────────────────
def _worker_init():
    preload_data()


def _apply_params(p: dict):
    g = globals()
    for k, v in p.items():
        if k in g:
            g[k] = v


def _tune_worker(p: dict):
    coins_filter = p.pop("_coins", None)
    _apply_params(p)
    active = coins_filter if coins_filter is not None else COINS
    coin_scores = {}
    for symbol, coin in active:
        metrics, _ = run_backtest(symbol, coin)
        if metrics is None or metrics["total_trades"] < 10 or metrics["win_rate"] < 0.35:
            coin_scores[coin] = float("-inf")
        else:
            coin_scores[coin] = metrics["calmar"]
    valid     = [s for s in coin_scores.values() if s != float("-inf")]
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
            t_df.to_csv(RESULTS_DIR / f"{coin}_sweep_div.csv", index=False)
    if rows:
        import pandas as pd
        table = pd.DataFrame(rows).to_string(index=False)
        print(f"\n{table}")
        (RESULTS_DIR / "best_results_table.txt").write_text(table)


# ── auto_tune ─────────────────────────────────────────────────────────────────
def auto_tune(coins=None):
    active_coins = coins if coins is not None else COINS
    keys   = list(TUNE_SPACE.keys())
    combos = [{**dict(zip(keys, c)), "_coins": active_coins}
              for c in itertools.product(*[TUNE_SPACE[k] for k in keys])]
    total     = len(combos)
    n_workers = min(16, max(1, mp.cpu_count() - 1))

    print(f"\n{'='*65}")
    print(f"  SWEEP-DIV AUTO-TUNE  |  {total} combos  |  {len(active_coins)} coins")
    print(f"  Workers              |  {n_workers} parallel processes")
    print(f"{'='*65}")

    best: dict = json.loads(BEST_PARAMS_FILE.read_text()) if BEST_PARAMS_FILE.exists() else {}

    ctx  = mp.get_context("spawn")
    done = 0
    pbar = tqdm(total=total, desc="SWEEP-DIV-TUNE", unit="combo", ncols=95)
    with ctx.Pool(processes=n_workers, initializer=_worker_init) as pool:
        for p, avg_score, coin_scores, snapped_params in \
                pool.imap_unordered(_tune_worker, combos, chunksize=8):
            done += 1
            pbar.update(1)

            updated = []
            for coin, sc in coin_scores.items():
                if sc == float("-inf"):
                    continue
                prev_sc = best.get(coin, {}).get("best_calmar", float("-inf"))
                if sc > prev_sc:
                    best[coin] = {
                        "best_calmar": round(sc, 6),
                        "params":      snapped_params,
                    }
                    updated.append(f"{coin.upper()} calmar={sc:.3f}")

            if updated:
                BEST_PARAMS_FILE.write_text(json.dumps(best, indent=2))
                pbar.write(
                    f"  [{done:>{len(str(total))}}/{total}]  avg {avg_score:.3f}"
                    f"  ★ {', '.join(updated)}"
                    f"  | sweep={p.get('SWEEP_PERIOD')} rr={p.get('TP_RR')}"
                    f" div={p.get('USE_DIV_FILTER')} fvg={p.get('USE_FVG_FILTER')}"
                )
            elif done % 200 == 0:
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
        for symbol, coin in (coins_filter or COINS):
            m, _ = run_backtest(symbol, coin)
            if m:
                print(f"{coin.upper()}: trades={m['total_trades']} wr={m['win_rate']:.2%} "
                      f"ret={m['total_return']:.2%} calmar={m['calmar']:.3f}")


if __name__ == "__main__":
    main()
