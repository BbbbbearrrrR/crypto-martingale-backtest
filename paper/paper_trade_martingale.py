#!/usr/bin/env python3
"""
Paper Trading Engine — Martingale Strategy
==========================================
Runs the EXACT same signal logic as backtest_martingale.py against live 1h candles.
No real money involved.

Two modes:
  USE_TESTNET = True  → Binance Testnet (real order book, fake USDT)
  USE_TESTNET = False → Pure local simulation (no API key needed)

State is persisted to paper_state_martingale.json so restarts don't lose positions.
All fills are appended to paper_trades_martingale.csv.

Usage:
    export BINANCE_API_KEY=your_key
    export BINANCE_API_SECRET=your_secret

    python paper_trade_martingale.py           # normal start
    python paper_trade_martingale.py --reset   # wipe state and start fresh
"""

# ── Must be set BEFORE numpy import ──────────────────────────────────────────
import os
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")

import json, sys, time, traceback
from datetime import datetime, timezone, timedelta
from pathlib import Path

_HERE = Path(__file__).resolve().parent
_ROOT = _HERE.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

import ccxt
import pandas as pd
import numpy as np

from backtest import backtest_martingale as bm   # reuse prepare(), Martin class, _apply_params()

# ── Config ────────────────────────────────────────────────────────────────────
USE_TESTNET       = True
INITIAL_CAPITAL   = 10_000.0
WARMUP_1H         = 300
WARMUP_1D         = 500
SLEEP_BUFFER_SEC  = 15

STATE_FILE       = _HERE / "paper_state_martingale.json"
TRADE_LOG_FILE   = _HERE / "paper_trades_martingale.csv"
BEST_PARAMS_FILE = _ROOT / "results/martingale/best_params.json"

API_KEY    = os.getenv("BINANCE_API_KEY", "")
API_SECRET = os.getenv("BINANCE_API_SECRET", "")

COINS = bm.COINS


# ── Exchange ──────────────────────────────────────────────────────────────────
def make_exchange() -> ccxt.binance:
    ex = ccxt.binance({
        "apiKey":  API_KEY,
        "secret":  API_SECRET,
        "options": {"defaultType": "future"},
    })
    if USE_TESTNET:
        ex.set_sandbox_mode(True)
    return ex


def fetch_ohlcv(ex, symbol: str, tf: str, limit: int) -> pd.DataFrame:
    raw = ex.fetch_ohlcv(symbol, tf, limit=limit + 1)
    df  = pd.DataFrame(raw, columns=["ts", "open", "high", "low", "close", "volume"])
    df.index = pd.to_datetime(df["ts"], unit="ms", utc=True)
    df.index.name = "datetime"
    df = df[["open", "high", "low", "close", "volume"]].astype(float)
    now_hour = pd.Timestamp.now(tz="UTC").floor("h")
    df = df[df.index < now_hour]
    return df.tail(limit)


# ── State serialization ───────────────────────────────────────────────────────
# Martin object is stored as a plain dict in JSON; rebuilt on each cycle.

def _martin_to_dict(m: bm.Martin) -> dict:
    return {
        "direction":    m.direction,
        "level":        m.level,
        "profit_level": m.profit_level,
        "notional":     m.notional,
        "entries":      m.entries,   # list of [price, notional]
        "grid_step":    m.grid_step,
        "capital":      m.capital,
        "tp_tier":      m.tp_tier,
    }


def _dict_to_martin(d: dict) -> bm.Martin:
    m             = bm.Martin.__new__(bm.Martin)
    m.direction   = d["direction"]
    m.level       = d["level"]
    m.profit_level = d["profit_level"]
    m.notional    = d["notional"]
    m.entries     = [tuple(e) for e in d["entries"]]
    m.grid_step   = d["grid_step"]
    m.capital     = d["capital"]
    m.tp_tier     = d["tp_tier"]
    return m


def _default_state() -> dict:
    return {
        "capital":   INITIAL_CAPITAL,
        "peak_cap":  INITIAL_CAPITAL,
        "martin":    None,          # serialized Martin dict or None
        "trades":    [],
    }


def load_state() -> dict:
    if STATE_FILE.exists():
        data = json.loads(STATE_FILE.read_text())
        for _, coin in COINS:
            if coin not in data:
                data[coin] = _default_state()
            else:
                for k, v in _default_state().items():
                    data[coin].setdefault(k, v)
        return data
    return {coin: _default_state() for _, coin in COINS}


def save_state(state: dict):
    STATE_FILE.write_text(json.dumps(state, indent=2, default=str))


def _log_trade(rec: dict):
    df  = pd.DataFrame([rec])
    hdr = not TRADE_LOG_FILE.exists()
    df.to_csv(TRADE_LOG_FILE, mode="a", index=False, header=hdr)


# ── Per-bar processing ────────────────────────────────────────────────────────
def process_bar(cs: dict, row: pd.Series, params: dict, coin: str, ts) -> list:
    """Process one just-closed candle. Mutates cs in place. Returns fill records."""
    records  = []
    cap      = cs["capital"]
    peak_cap = cs["peak_cap"]

    # Rebuild Martin object from persisted dict
    martin = _dict_to_martin(cs["martin"]) if cs["martin"] else None

    # ── Exit / add-level ─────────────────────────────────────────────────────
    if martin is not None:
        tp_price = martin.tp()
        hard_sl  = martin.hard_sl()
        next_add = martin.next_add_price()

        hit_tp  = (row["high"] >= tp_price if martin.direction == "long" else row["low"]  <= tp_price)
        hit_sl  = (row["low"]  <= hard_sl  if martin.direction == "long" else row["high"] >= hard_sl)
        hit_add = (martin.level < params.get("MAX_LEVELS", 10) and
                   (row["low"] <= next_add if martin.direction == "long" else row["high"] >= next_add))

        _pnl        = martin.pnl(float(row["close"]))
        _one_margin = martin.notional / params.get("LEVERAGE", 50)
        in_profit   = _pnl >= params.get("PYRAMID_MIN_PROFIT_RATE", 0.5) * _one_margin
        hit_pyramid = (
            martin.profit_level < params.get("MAX_PYRAMID_LEVELS", 10) and in_profit and
            ((martin.direction == "long"  and bool(row.get("mid_cross_up", False))) or
             (martin.direction == "short" and bool(row.get("mid_cross_down", False))))
        )

        if hit_tp:
            partial_pnl  = martin.partial_close(tp_price)
            partial_pnl  = max(partial_pnl, -cap)
            cap         += partial_pnl
            peak_cap     = max(peak_cap, cap)
            is_last_tier = (martin.tp_tier >= params.get("TP_SCALE_LEVELS", 1))
            reason       = "TP" if is_last_tier else f"TP{martin.tp_tier}"

            rec = dict(timestamp=str(ts), coin=coin, direction=martin.direction,
                       level=martin.level, profit_level=martin.profit_level,
                       exit_reason=reason, notional=round(martin.notional, 4),
                       pnl_usdt=round(partial_pnl, 4), capital=round(cap, 4))
            records.append(rec)
            _log_trade(rec)

            sym = "✓" if partial_pnl >= 0 else "✗"
            print(f"    {sym} {reason:10s}  {martin.direction.upper():5s}"
                  f"  tp={tp_price:.4f}  pnl=${partial_pnl:+.2f}  cap=${cap:.0f}"
                  f"  lvl={martin.level}  @{str(ts)[:16]}")

            if is_last_tier:
                cs["trades"].append(rec)
                martin = None
            else:
                cs["martin"] = _martin_to_dict(martin)

        elif hit_sl:
            pnl      = martin.pnl(hard_sl)
            pnl      = max(pnl, -cap)
            cap     += pnl
            peak_cap = max(peak_cap, cap)

            rec = dict(timestamp=str(ts), coin=coin, direction=martin.direction,
                       level=martin.level, profit_level=martin.profit_level,
                       exit_reason="MAX_SL", notional=round(martin.notional, 4),
                       pnl_usdt=round(pnl, 4), capital=round(cap, 4))
            records.append(rec)
            _log_trade(rec)
            cs["trades"].append(rec)

            print(f"    ✗ MAX_SL     {martin.direction.upper():5s}"
                  f"  sl={hard_sl:.4f}  pnl=${pnl:+.2f}  cap=${cap:.0f}"
                  f"  lvl={martin.level}  @{str(ts)[:16]}")
            martin = None

        elif hit_add:
            martin.add_level(next_add)
            print(f"    + ADD_LVL   {martin.direction.upper():5s}"
                  f"  price={next_add:.4f}  lvl={martin.level}  @{str(ts)[:16]}")
            cs["martin"] = _martin_to_dict(martin)

        elif hit_pyramid:
            martin.add_pyramid_level(float(row["close"]))
            print(f"    ▲ PYRAMID   {martin.direction.upper():5s}"
                  f"  price={row['close']:.4f}  plvl={martin.profit_level}  @{str(ts)[:16]}")
            cs["martin"] = _martin_to_dict(martin)

    # ── Entry ─────────────────────────────────────────────────────────────────
    if martin is None:
        leverage  = params.get("LEVERAGE", 50)
        base_risk = params.get("BASE_RISK", 0.05)
        notional  = min(cap * base_risk, cap * leverage)

        trend_up    = bool(row.get("trend_up", False))
        entry_long  = bool(row.get("entry_long", False))
        entry_short = bool(row.get("entry_short", False))

        if entry_long and trend_up:
            martin = bm.Martin("long",  float(row["close"]), notional, cap)
            cs["martin"] = _martin_to_dict(martin)
            print(f"    ▶ ENTRY  LONG   price={row['close']:.4f}"
                  f"  notional=${notional:.0f}  @{str(ts)[:16]}")
        elif entry_short and not trend_up:
            martin = bm.Martin("short", float(row["close"]), notional, cap)
            cs["martin"] = _martin_to_dict(martin)
            print(f"    ▶ ENTRY  SHORT  price={row['close']:.4f}"
                  f"  notional=${notional:.0f}  @{str(ts)[:16]}")
        else:
            cs["martin"] = None

    cs["capital"]  = cap
    cs["peak_cap"] = peak_cap
    return records


# ── Report ────────────────────────────────────────────────────────────────────
def print_report(state: dict):
    now = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC")
    print(f"\n{'═'*65}")
    print(f"  PAPER PORTFOLIO (Martingale)  |  {now}")
    print(f"{'═'*65}")
    total_cap = 0.0
    for _, coin in COINS:
        cs    = state[coin]
        cap   = cs["capital"]
        ret   = (cap - INITIAL_CAPITAL) / INITIAL_CAPITAL * 100
        total_cap += cap
        n     = len(cs["trades"])
        wins  = sum(1 for t in cs["trades"] if t["pnl_usdt"] > 0)
        wr    = f"{wins/n*100:.0f}%" if n else " - "

        if cs["martin"]:
            md = cs["martin"]
            pos = (f"{md['direction'].upper()} lvl={md['level']}"
                   f" entry={md['entries'][0][0]:.4f}")
        else:
            pos = "flat"
        print(f"  {coin.upper():5s}  cap=${cap:>9.2f}  ret={ret:>+7.2f}%"
              f"  trades={n:>3d}  wr={wr:>4s}  [{pos}]")
    total_ret = (total_cap - INITIAL_CAPITAL * len(COINS)) / (INITIAL_CAPITAL * len(COINS)) * 100
    print(f"{'─'*65}")
    print(f"  TOTAL            ${total_cap:>9.2f}  ret={total_ret:>+7.2f}%")
    print(f"{'═'*65}\n")


# ── Main cycle ────────────────────────────────────────────────────────────────
def run_cycle(ex, state: dict, best: dict):
    now = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC")
    print(f"\n{'─'*65}")
    print(f"  CYCLE  {now}")
    print(f"{'─'*65}")

    for symbol, coin in COINS:
        entry  = best.get(coin, {})
        params = entry.get("params", {})
        if not params:
            print(f"  [{coin.upper()}] no best params, skipping")
            continue

        try:
            bm._apply_params(params)

            df_1h = fetch_ohlcv(ex, symbol, "1h", WARMUP_1H)
            df_1d = fetch_ohlcv(ex, symbol, "1d", WARMUP_1D)

            df  = bm.prepare(df_1h, df_1d)
            row = df.iloc[-1]
            ts  = df.index[-1]

            trend  = "↑" if bool(row.get("trend_up", False)) else "↓"
            el = "L✓" if bool(row.get("entry_long", False))  else "  "
            es = "S✓" if bool(row.get("entry_short", False)) else "  "
            mu = "↑" if bool(row.get("mid_cross_up", False))   else "  "
            md = "↓" if bool(row.get("mid_cross_down", False)) else "  "

            lvl_str = ""
            if state[coin]["martin"]:
                lvl_str = f"  lvl={state[coin]['martin']['level']}"

            print(f"  [{coin.upper()}]  close={row['close']:.4f}"
                  f"  trend={trend}  {el} {es}  mid{mu}{md}{lvl_str}")

            process_bar(state[coin], row, params, coin, ts)

        except ccxt.NetworkError as e:
            print(f"  [{coin.upper()}] network error: {e}  — will retry next cycle")
        except Exception:
            print(f"  [{coin.upper()}] unexpected error:")
            traceback.print_exc()

    save_state(state)
    print_report(state)


# ── Scheduling ────────────────────────────────────────────────────────────────
def seconds_to_next_candle() -> float:
    now       = datetime.now(timezone.utc)
    next_hour = now.replace(minute=0, second=0, microsecond=0) + timedelta(hours=1)
    return max((next_hour - now).total_seconds() + SLEEP_BUFFER_SEC, 0)


# ── Entry point ───────────────────────────────────────────────────────────────
def main():
    reset = "--reset" in sys.argv

    print("╔═══════════════════════════════════════════════════════════╗")
    print("║   PAPER TRADING ENGINE — Martingale Strategy              ║")
    print(f"║   Testnet : {str(USE_TESTNET):<49}║")
    print(f"║   Capital : ${INITIAL_CAPITAL:,.0f} / coin{' '*(44 - len(f'{INITIAL_CAPITAL:,.0f}'))}║")
    print(f"║   State   : {str(STATE_FILE):<49}║")
    print(f"║   Trades  : {str(TRADE_LOG_FILE):<49}║")
    print("╚═══════════════════════════════════════════════════════════╝\n")

    if not BEST_PARAMS_FILE.exists():
        print(f"ERROR: {BEST_PARAMS_FILE} not found. Run auto_tune first.")
        sys.exit(1)

    if reset:
        STATE_FILE.unlink(missing_ok=True)
        print("State reset.\n")

    best  = json.loads(BEST_PARAMS_FILE.read_text())
    state = load_state()
    ex    = make_exchange()

    run_cycle(ex, state, best)

    while True:
        wait = seconds_to_next_candle()
        nxt  = (datetime.now(timezone.utc) + timedelta(seconds=wait)).strftime("%H:%M:%S UTC")
        print(f"  ⏱  Next cycle in {wait/60:.1f} min  ({nxt})")
        time.sleep(wait)
        best = json.loads(BEST_PARAMS_FILE.read_text())
        run_cycle(ex, state, best)


if __name__ == "__main__":
    main()
