"""
Trend Breakout Backtest  (v2 – ExitManager refactor)
=====================================================
Core idea  : Donchian channel breakout in the direction of the daily trend.
             Price making a new N-bar high/low is evidence that a new leg has
             started.  We ride it until stopped out.

Entry rules
-----------
1. 1h close breaks above Donchian upper  AND  1d trend_up  -> long
2. 1h close breaks below Donchian lower  AND  NOT 1d trend_up  -> short
3. ADX filter: only enter when ADX >= ADX_MIN (trending market)
4. Volume filter: breakout bar volume >= VOL_MULT x N-bar MA
5. No re-entry while in a trade; optional cooldown after SL

Stop / TP
---------
Fully handled by ExitManager (backtest/exit_manager.py):
  - Initial SL  : Donchian opposite band (breakout invalidation level)
                  OR entry +- ATR x SL_MULT  (selectable via SL_MODE)
  - Partial TP  : PARTIAL_R x initial_SL_dist, closes PARTIAL_FRAC, moves SL to BE
  - Trailing SL : ATR-based, activates after TRAIL_TRIGGER_R x initial_SL_dist profit
  - Full TP     : TP_RR x initial_SL_dist
  - Timeout     : MAX_HOLD_BARS (0 = disabled)
  - Trend exit  : exit at close when daily trend flips (USE_TREND_EXIT)

Sizing
------
  risk_amount = capital x BASE_RISK
  notional    = risk_amount / sl_dist_pct,  capped at capital x LEVERAGE
"""

import os
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")

import json
import pandas as pd
import numpy as np
from pathlib import Path
import sys
from tqdm import tqdm

from backtest.exit_manager import ExitManager

_ROOT       = Path(__file__).resolve().parent.parent
DATA_DIR    = _ROOT / "data"
RESULTS_DIR = _ROOT / "results/breakout"
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

_RAW_DATA: dict = {}
BEST_PARAMS_FILE = RESULTS_DIR / "best_params.json"

# ---- Parameters --------------------------------------------------------------
INITIAL_CAPITAL  = 10_000
FEE_RATE         = 0.0005
LEVERAGE         = 10
BASE_RISK        = 0.01

DONCHIAN_PERIOD  = 20
ATR_PERIOD       = 14
TREND_EMA_PERIOD = 200
ADX_PERIOD       = 14
ADX_MIN          = 25.0
VOL_MA_PERIOD    = 20
VOL_MULT         = 1.5
COOLDOWN_BARS    = 0

SL_MODE          = "donchian"   # "donchian" | "atr"
SL_MULT          = 1.5          # used only when SL_MODE == "atr"

USE_OBV_FILTER   = True         # OBV must confirm breakout direction
OBV_PERIOD       = 20           # rolling window for OBV MA comparison

USE_PARTIAL_TP   = True
PARTIAL_R        = 1.0
PARTIAL_FRAC     = 0.5
USE_TRAIL        = True
TRAIL_ATR_MULT   = 1.0
TRAIL_TRIGGER_R  = 1.0
TP_RR            = 3.0
MAX_HOLD_BARS    = 72           # 0 = disabled
USE_TREND_EXIT   = True

# ---- Auto-tuning -------------------------------------------------------------
AUTO_TUNE = True

TUNE_SPACE = {
    # ── Most impactful: signal quality ────────────────────────────────────────
    "DONCHIAN_PERIOD":   [10, 20, 40],    # breakout sensitivity
    "TREND_EMA_PERIOD":  [50, 100, 200],  # daily trend lag
    "ADX_MIN":           [0.0, 20.0],     # off vs filtered (25 ≈ 20, removed)
    # ── Most impactful: exit quality ──────────────────────────────────────────
    "TP_RR":             [2.0, 3.0, 5.0], # critical for ~25% win rate
    "MAX_HOLD_BARS":     [0, 48, 96],     # 0=off, 2d, 4d
    # ── SL placement ─────────────────────────────────────────────────────────
    "SL_MODE":           ["donchian", "atr"],
    "SL_MULT":           [2.0],           # only used when SL_MODE="atr"; fixed
    # ── Volume / OBV filters ─────────────────────────────────────────────────
    "VOL_MULT":          [1.0],           # fixed; 1.5 rarely selected in prior runs
    "USE_OBV_FILTER":    [False],         # fixed; rarely selected
    # ── Exit switches ─────────────────────────────────────────────────────────
    "USE_PARTIAL_TP":    [True, False],
    "USE_TRAIL":         [True, False],
    "USE_TREND_EXIT":    [True, False],
    # ── Position sizing ───────────────────────────────────────────────────────
    "LEVERAGE":          [5, 10],         # drop 3x (too conservative for futures)
    "BASE_RISK":         [0.01, 0.02, 0.05],
    # ── Fixed: ATR_PERIOD=14, COOLDOWN_BARS=0, OBV_PERIOD=20
    # PARTIAL_R=1.0, TRAIL_TRIGGER_R=1.0, PARTIAL_FRAC=0.5
    # Total: 3×3×2×3×3×2×1×1×1×2×2×2×2×3 = 15,552 combos
}

COINS = [
    ("BTC/USDT:USDT",  "btc"),
    ("ETH/USDT:USDT",  "eth"),
    ("SOL/USDT:USDT",  "sol"),
    ("HYPE/USDT:USDT", "hype"),
    ("SUI/USDT:USDT",  "sui"),
]


# ---- Indicators --------------------------------------------------------------
def prepare(df_1h: pd.DataFrame, df_1d: pd.DataFrame) -> pd.DataFrame:
    df = df_1h.copy()

    # Donchian (shift 1 -> no look-ahead)
    df["don_upper"] = df["high"].rolling(DONCHIAN_PERIOD).max().shift(1)
    df["don_lower"] = df["low"].rolling(DONCHIAN_PERIOD).min().shift(1)

    # ATR (EWM)
    prev_close = df["close"].shift(1)
    tr = pd.concat([
        df["high"] - df["low"],
        (df["high"] - prev_close).abs(),
        (df["low"]  - prev_close).abs(),
    ], axis=1).max(axis=1)
    df["atr"] = tr.ewm(span=ATR_PERIOD, adjust=False).mean()

    # ADX (Wilder EWM)
    up_move  = df["high"] - df["high"].shift(1)
    dn_move  = df["low"].shift(1) - df["low"]
    plus_dm  = np.where((up_move > dn_move) & (up_move > 0), up_move, 0.0)
    minus_dm = np.where((dn_move > up_move) & (dn_move > 0), dn_move, 0.0)
    atr_dx   = tr.ewm(span=ADX_PERIOD, adjust=False).mean().clip(lower=1e-9)
    plus_di  = 100 * pd.Series(plus_dm,  index=df.index).ewm(span=ADX_PERIOD, adjust=False).mean() / atr_dx
    minus_di = 100 * pd.Series(minus_dm, index=df.index).ewm(span=ADX_PERIOD, adjust=False).mean() / atr_dx
    dx       = 100 * (plus_di - minus_di).abs() / (plus_di + minus_di).clip(lower=1e-9)
    df["adx"] = dx.ewm(span=ADX_PERIOD, adjust=False).mean()

    # Volume filter
    if "volume" in df.columns:
        df["vol_ma"] = df["volume"].rolling(VOL_MA_PERIOD).mean()
        df["vol_ok"] = df["volume"] >= df["vol_ma"] * VOL_MULT
    else:
        df["vol_ok"] = True

    # OBV (On-Balance Volume) and its MA for trend confirmation
    if "volume" in df.columns:
        direction_sign = np.sign(df["close"].diff()).fillna(0)
        obv            = (df["volume"] * direction_sign).cumsum()
        obv_ma         = obv.rolling(OBV_PERIOD).mean()
        df["obv_above_ma"] = obv > obv_ma   # OBV trending up  → confirms long
        df["obv_below_ma"] = obv < obv_ma   # OBV trending down → confirms short
    else:
        df["obv_above_ma"] = True
        df["obv_below_ma"] = True

    # Breakout signals
    df["entry_long"]  = df["close"] > df["don_upper"]
    df["entry_short"] = df["close"] < df["don_lower"]

    # Daily trend: 1d close > EMA -> trend_up
    d1     = df_1d.copy()
    d1_ema = d1["close"].ewm(span=TREND_EMA_PERIOD, adjust=False).mean()
    d1["trend_up"] = d1["close"] > d1_ema
    trend  = d1["trend_up"].reindex(df.index, method="ffill").ffill()
    df["trend_up"] = np.where(trend.isna(), False, trend).astype(bool)

    return df


# ---- Backtest ----------------------------------------------------------------
def preload_data():
    global _RAW_DATA
    for _, coin in COINS:
        _RAW_DATA[coin] = (
            pd.read_csv(DATA_DIR / f"{coin}_futures_1h.csv", index_col=0, parse_dates=True),
            pd.read_csv(DATA_DIR / f"{coin}_futures_1d.csv", index_col=0, parse_dates=True),
        )


def _exit_params() -> dict:
    return {
        "USE_PARTIAL_TP":  USE_PARTIAL_TP,
        "PARTIAL_R":       PARTIAL_R,
        "PARTIAL_FRAC":    PARTIAL_FRAC,
        "USE_TRAIL":       USE_TRAIL,
        "TRAIL_ATR_MULT":  TRAIL_ATR_MULT,
        "TRAIL_TRIGGER_R": TRAIL_TRIGGER_R,
        "TP_RR":           TP_RR,
        "MAX_HOLD_BARS":   MAX_HOLD_BARS,
        "USE_TREND_EXIT":  USE_TREND_EXIT,
        "FEE_RATE":        FEE_RATE,
    }


def run_backtest(symbol: str, coin: str):
    print(f"\n{'='*50}\n  {symbol}\n{'='*50}")

    df_1h, df_1d = _RAW_DATA.get(coin) or (
        pd.read_csv(DATA_DIR / f"{coin}_futures_1h.csv", index_col=0, parse_dates=True),
        pd.read_csv(DATA_DIR / f"{coin}_futures_1d.csv", index_col=0, parse_dates=True),
    )
    df = prepare(df_1h, df_1d)

    capital       = float(INITIAL_CAPITAL)
    peak_cap      = capital
    trades        = []
    in_trade      = False
    em            = None
    notional_full = 0.0
    notional_rem  = 0.0
    entry_price   = 0.0
    direction     = None
    cooldown_rem  = 0

    warmup = max(DONCHIAN_PERIOD, ATR_PERIOD, ADX_PERIOD, VOL_MA_PERIOD) + 2

    _iter = tqdm(range(warmup, len(df)), desc=f"{coin.upper()}", unit="bar",
                 file=sys.stdout, disable=not sys.stdout.isatty(), dynamic_ncols=True)

    for i in _iter:
        row = df.iloc[i]
        ts  = df.index[i]

        if capital <= 10:
            break

        # -- Exit processing ---------------------------------------------------
        if in_trade and em is not None:
            result = em.update(row)

            if result.partial:
                pt    = result.partial
                cn    = notional_full * pt.frac
                pnl_p = cn * pt.pnl_frac - cn * FEE_RATE * 2
                capital  += pnl_p
                peak_cap  = max(peak_cap, capital)
                notional_rem = notional_full * (1.0 - pt.frac)
                trades.append({
                    "exit_time":   ts,
                    "direction":   direction,
                    "entry_price": round(entry_price, 6),
                    "exit_price":  round(pt.exit_price, 6),
                    "notional":    round(cn, 4),
                    "exit_reason": pt.reason,
                    "pnl_usdt":    round(pnl_p, 4),
                    "capital":     round(capital, 4),
                    "drawdown":    round((peak_cap - capital) / peak_cap, 6),
                })

            if result.closed:
                pnl = notional_rem * result.pnl_frac - notional_rem * FEE_RATE * 2
                pnl = max(pnl, -capital)
                capital  += pnl
                peak_cap  = max(peak_cap, capital)
                trades.append({
                    "exit_time":   ts,
                    "direction":   direction,
                    "entry_price": round(entry_price, 6),
                    "exit_price":  round(result.exit_price, 6),
                    "notional":    round(notional_rem, 4),
                    "exit_reason": result.reason,
                    "pnl_usdt":    round(pnl, 4),
                    "capital":     round(capital, 4),
                    "drawdown":    round((peak_cap - capital) / peak_cap, 6),
                })
                in_trade = False
                em       = None
                if result.reason == "SL":
                    cooldown_rem = COOLDOWN_BARS

        # -- Entry -------------------------------------------------------------
        if not in_trade:
            if cooldown_rem > 0:
                cooldown_rem -= 1
                continue

            atr = float(row["atr"])
            if pd.isna(atr) or atr <= 0:
                continue
            if ADX_MIN > 0 and (pd.isna(row["adx"]) or float(row["adx"]) < ADX_MIN):
                continue
            if not row["vol_ok"]:
                continue

            if   row["entry_long"]  and     row["trend_up"]:
                direction = "long"
            elif row["entry_short"] and not row["trend_up"]:
                direction = "short"
            else:
                continue

            # OBV divergence filter: skip if OBV contradicts breakout direction
            if USE_OBV_FILTER:
                if direction == "long"  and not bool(row.get("obv_above_ma", True)):
                    continue
                if direction == "short" and not bool(row.get("obv_below_ma", True)):
                    continue

            entry_price = float(row["close"])

            if SL_MODE == "donchian":
                sl_price = float(row["don_lower"]) if direction == "long" else float(row["don_upper"])
                if direction == "long"  and sl_price >= entry_price:
                    sl_price = entry_price - atr * 1.5
                if direction == "short" and sl_price <= entry_price:
                    sl_price = entry_price + atr * 1.5
            else:
                sl_price = (entry_price - atr * SL_MULT if direction == "long"
                            else entry_price + atr * SL_MULT)

            sl_dist_pct = abs(entry_price - sl_price) / entry_price
            if sl_dist_pct < 1e-6:
                continue

            sl_dist  = abs(entry_price - sl_price)
            tp_price = (entry_price + sl_dist * TP_RR if direction == "long"
                        else entry_price - sl_dist * TP_RR)

            notional_full = min(capital * BASE_RISK / sl_dist_pct, capital * LEVERAGE)
            notional_rem  = notional_full

            em = ExitManager(
                direction   = direction,
                entry_price = entry_price,
                sl_price    = sl_price,
                tp_price    = tp_price,
                params      = _exit_params(),
            )
            in_trade = True

    if not trades:
        print("  No trades.")
        return None

    t = pd.DataFrame(trades)
    t.to_csv(RESULTS_DIR / f"{coin}_breakout.csv", index=False)
    m = compute_metrics(t, INITIAL_CAPITAL)

    pf_s  = f"{m['pf']:.2f}"     if m["pf"]    < 999 else "inf"
    cal_s = f"{m['calmar']:.2f}" if m["calmar"] < 999 else "inf"
    print(f"  Trades        : {m['n']}  ({m['wins']}W / {m['losses']}L,  {m['win_rate']*100:.1f}%)")
    print(f"  TP/SL/Trail   : {(t['exit_reason']=='TP').sum()} / {(t['exit_reason']=='SL').sum()} / {(t['exit_reason']=='TRAIL_SL').sum()}")
    print(f"  Avg win/loss  : ${m['avg_win']:.2f} / ${m['avg_loss']:.2f}")
    print(f"  Expectancy    : ${m['expectancy']:.2f} / trade")
    print(f"  Profit factor : {pf_s}")
    print(f"  Total PnL     : ${t['pnl_usdt'].sum():.2f}")
    print(f"  Total return  : {m['total_ret']*100:.1f}%")
    print(f"  Max drawdown  : {m['max_dd']*100:.1f}%")
    print(f"  Sharpe (ann)  : {m['sharpe']:.2f}")
    print(f"  Calmar        : {cal_s}")
    print(f"  Final capital : ${m['final']:.2f}")
    return t, (m["final"] - INITIAL_CAPITAL) / INITIAL_CAPITAL, t["drawdown"].max(), m


# ---- Metrics -----------------------------------------------------------------
def compute_metrics(t: pd.DataFrame, initial_capital: float) -> dict:
    n        = len(t)
    wins     = int((t["pnl_usdt"] > 0).sum())
    losses   = n - wins
    win_rate = wins / n           # fraction 0-1
    avg_win  = float(t.loc[t["pnl_usdt"] > 0,  "pnl_usdt"].mean()) if wins   else 0.0
    avg_loss = float(t.loc[t["pnl_usdt"] <= 0, "pnl_usdt"].mean()) if losses else 0.0
    pf       = (wins * avg_win / (-losses * avg_loss)
                if losses and avg_loss < 0 else float("inf"))
    expectancy = float(t["pnl_usdt"].mean())
    final      = float(t["capital"].iloc[-1])
    total_ret  = (final - initial_capital) / initial_capital  # fraction 0-1
    max_dd     = float(t["drawdown"].max())                   # fraction 0-1

    eq        = t.set_index("exit_time")["capital"].resample("1D").last().ffill()
    first_day = eq.index[0] - pd.Timedelta(days=1)
    eq        = pd.concat([pd.Series({first_day: float(initial_capital)}), eq])
    daily_ret = eq.pct_change().dropna()
    sharpe    = float(daily_ret.mean() / daily_ret.std() * np.sqrt(252)) if daily_ret.std() > 0 else 0.0

    span_days = max((t["exit_time"].iloc[-1] - t["exit_time"].iloc[0]).days, 1)
    ann_ret   = (final / initial_capital) ** (365.25 / span_days) - 1
    calmar    = float(ann_ret / max_dd) if max_dd > 1e-6 else float("inf")

    return dict(n=n, wins=wins, losses=losses, win_rate=win_rate,
                avg_win=avg_win, avg_loss=avg_loss, pf=pf, expectancy=expectancy,
                total_ret=total_ret, max_dd=max_dd, sharpe=sharpe, calmar=calmar, final=final)


def print_summary_table(strategy_name: str, header: str, metrics: dict):
    cols   = ["Coin",  "Trades", "Win%",  "AvgWin$", "AvgLoss$", "PF",
              "Expect$", "Return%", "MaxDD%", "Sharpe",  "Calmar"]
    widths = [5,        7,        6,        9,         9,          6,
              8,         8,         7,        7,         7]

    def _sep(lft, mid, rgt):
        return lft + mid.join("─" * (w + 2) for w in widths) + rgt

    def _row(vals):
        return "|" + "|".join(f" {str(v):>{w}} " for v, w in zip(vals, widths)) + "|"

    total_w = sum(w + 3 for w in widths) + 1
    title   = f" {strategy_name}  *  {header} "
    print(f"\n+{'-' * (total_w - 2)}+")
    print(f"|{title:<{total_w - 2}}|")
    print(_sep("+", "+", "+"))
    print(_row(cols))
    print(_sep("+", "+", "+"))
    for coin, m in metrics.items():
        pf_s  = f"{m['pf']:.2f}"     if m["pf"]    < 999 else "inf"
        cal_s = f"{m['calmar']:.2f}" if m["calmar"] < 999 else "inf"
        print(_row([
            coin.upper(), m["n"], f"{m['win_rate']*100:.1f}%",
            f"${m['avg_win']:.1f}", f"${m['avg_loss']:.1f}",
            pf_s, f"${m['expectancy']:.1f}",
            f"{m['total_ret']*100:.1f}%", f"{m['max_dd']*100:.1f}%",
            f"{m['sharpe']:.2f}", cal_s,
        ]))
    print(_sep("+", "+", "+"))


# ---- Tune helpers ------------------------------------------------------------
def current_params() -> dict:
    return {
        "LEVERAGE": LEVERAGE, "BASE_RISK": BASE_RISK,
        "DONCHIAN_PERIOD": DONCHIAN_PERIOD, "ATR_PERIOD": ATR_PERIOD,
        "SL_MODE": SL_MODE, "SL_MULT": SL_MULT,
        "TREND_EMA_PERIOD": TREND_EMA_PERIOD,
        "ADX_PERIOD": ADX_PERIOD, "ADX_MIN": ADX_MIN,
        "VOL_MA_PERIOD": VOL_MA_PERIOD, "VOL_MULT": VOL_MULT,
        "COOLDOWN_BARS": COOLDOWN_BARS,
        "USE_PARTIAL_TP": USE_PARTIAL_TP, "PARTIAL_R": PARTIAL_R, "PARTIAL_FRAC": PARTIAL_FRAC,
        "USE_TRAIL": USE_TRAIL, "TRAIL_ATR_MULT": TRAIL_ATR_MULT, "TRAIL_TRIGGER_R": TRAIL_TRIGGER_R,
        "TP_RR": TP_RR, "MAX_HOLD_BARS": MAX_HOLD_BARS, "USE_TREND_EXIT": USE_TREND_EXIT,
        "USE_OBV_FILTER": USE_OBV_FILTER, "OBV_PERIOD": OBV_PERIOD,
    }


def run_once(verbose: bool = True, coins=None) -> tuple:
    active_coins = coins if coins is not None else COINS
    coin_returns: dict = {}
    coin_max_dd: dict  = {}
    coin_metrics: dict     = {}
    for symbol, coin in active_coins:
        result = run_backtest(symbol, coin) if verbose else _run_silent(symbol, coin)
        if result is not None:
            _, ret, max_dd_frac, m = result
            coin_returns[coin] = ret
            coin_max_dd[coin]  = max_dd_frac
            coin_metrics[coin] = m
    avg_ret = sum(coin_returns.values()) / len(coin_returns) if coin_returns else 0.0
    return avg_ret, coin_returns, coin_max_dd, coin_metrics


def _run_silent(symbol: str, coin: str):
    import io, contextlib
    buf = io.StringIO()
    with contextlib.redirect_stdout(buf):
        return run_backtest(symbol, coin)


def _apply_params(p: dict):
    g = globals()
    for k, v in p.items():
        g[k] = v


def _worker_init():
    global _RAW_DATA
    for _, coin in COINS:
        _RAW_DATA[coin] = (
            pd.read_csv(DATA_DIR / f"{coin}_futures_1h.csv", index_col=0, parse_dates=True),
            pd.read_csv(DATA_DIR / f"{coin}_futures_1d.csv", index_col=0, parse_dates=True),
        )


def _tune_worker(p: dict):
    coins_filter = p.pop("_coins", None)
    _apply_params(p)
    avg_ret, coin_returns, coin_max_dd, coin_metrics = run_once(verbose=False, coins=coins_filter)
    # Score = Calmar ratio (annualised return / max drawdown); fall back to 0 if no trades
    coin_scores = {coin: (coin_metrics[coin]["calmar"] if coin_metrics.get(coin) and
                          coin_metrics[coin]["calmar"] < 1e9 else 0.0)
                  for coin in coin_returns}
    return p, avg_ret, coin_returns, coin_max_dd, coin_scores, current_params()


def _save_best_results_table():
    if not BEST_PARAMS_FILE.exists():
        return
    best = json.loads(BEST_PARAMS_FILE.read_text())
    coin_metrics: dict = {}
    for symbol, coin in COINS:
        entry = best.get(coin)
        if not entry:
            continue
        _apply_params(entry["params"])
        result = _run_silent(symbol, coin)
        if result is not None:
            _, _ret, _hold, m = result
            coin_metrics[coin] = m
    if not coin_metrics:
        return
    import io, sys as _sys
    buf = io.StringIO()
    _old, _sys.stdout = _sys.stdout, buf
    try:
        print_summary_table("Trend Breakout (Best Params)",
                             f"per-coin optimal  |  {len(coin_metrics)} coins",
                             coin_metrics)
    finally:
        _sys.stdout = _old
    table_text = buf.getvalue()
    print(table_text)
    out_file = RESULTS_DIR / "best_results_table.txt"
    out_file.write_text(table_text, encoding="utf-8")
    print(f"Best results table saved to {out_file}")


def auto_tune(coins=None):
    import itertools
    import multiprocessing as mp
    active_coins = coins if coins is not None else COINS
    keys   = list(TUNE_SPACE.keys())
    values = list(TUNE_SPACE.values())
    combos = [{**dict(zip(keys, c)), "_coins": active_coins} for c in itertools.product(*values)]
    total  = len(combos)
    n_workers = min(16, max(1, mp.cpu_count() - 1))
    print(f"\n{'='*60}\n  AUTO-TUNE  |  {total} combos  |  {n_workers} workers\n{'='*60}")

    best: dict = json.loads(BEST_PARAMS_FILE.read_text()) if BEST_PARAMS_FILE.exists() else {}

    ctx  = mp.get_context("spawn")
    done = 0
    pbar = tqdm(total=total, desc="AUTO-TUNE", unit="combo", ncols=90)
    with ctx.Pool(processes=n_workers, initializer=_worker_init) as pool:
        for p, avg_ret, coin_returns, coin_max_dd, coin_scores, snapped in \
                pool.imap_unordered(_tune_worker, combos, chunksize=1):
            done += 1
            pbar.update(1)
            updated = []
            for coin, score in coin_scores.items():
                prev_score = best.get(coin, {}).get("best_calmar", float("-inf"))
                if score > prev_score:
                    best[coin] = {
                        "best_calmar": round(score, 4),
                        "best_return": round(coin_returns.get(coin, 0), 6),
                        "max_dd_frac": round(coin_max_dd[coin], 6),
                        "params":      snapped,
                    }
                    updated.append(f"{coin.upper()} calmar={score:.2f}")
            if updated:
                BEST_PARAMS_FILE.write_text(json.dumps(best, indent=2))
                pbar.write(
                    f"  [{done:>{len(str(total))}}/{total}]  avg_ret {avg_ret*100:.1f}%"
                    f"  * {', '.join(updated)}"
                    f"  | sl={p.get('SL_MODE')} don={p.get('DONCHIAN_PERIOD')}"
                    f" rr={p.get('TP_RR')} hold={p.get('MAX_HOLD_BARS')}"
                )
            elif done % 100 == 0:
                pbar.write(f"  [{done:>{len(str(total))}}/{total}]  avg_ret {avg_ret*100:.1f}%")
    pbar.close()
    print(f"\nTuning complete.  Best params -> {BEST_PARAMS_FILE}")
    _save_best_results_table()


def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--coin", type=str, default=None)
    args, _ = parser.parse_known_args()
    coins_filter = None
    if args.coin:
        coins_filter = [(s, c) for s, c in COINS if c == args.coin.lower()]
        if not coins_filter:
            print(f"Unknown coin '{args.coin}'. Available: {[c for _, c in COINS]}")
            return

    if AUTO_TUNE:
        auto_tune(coins=coins_filter)
        return

    avg_return, coin_returns, coin_max_dd, coin_metrics = run_once(verbose=True)
    print(f"\nAvg return: {avg_return*100:.1f}%")
    print_summary_table("Trend Breakout", "single run", coin_metrics)

    best: dict = json.loads(BEST_PARAMS_FILE.read_text()) if BEST_PARAMS_FILE.exists() else {}
    for coin, m in coin_metrics.items():
        score = m["calmar"] if m["calmar"] < 1e9 else 0.0
        if score > best.get(coin, {}).get("best_calmar", float("-inf")):
            best[coin] = {
                "best_calmar": round(score, 4),
                "best_return": round(coin_returns[coin], 6),
                "max_dd_frac": round(coin_max_dd[coin], 6),
                "params":      current_params(),
            }
    BEST_PARAMS_FILE.write_text(json.dumps(best, indent=2))
    print(f"\nLogs -> {RESULTS_DIR}/")


if __name__ == "__main__":
    main()
