#!/usr/bin/env python3
"""
Paper Trading Engine — Breakout Strategy
=========================================
Runs the EXACT same signal logic as backtest_breakout.py against live 1h candles.
No real money involved.

Two modes:
  USE_TESTNET = True  → Binance Testnet (real order book, fake USDT)
  USE_TESTNET = False → Pure local simulation (no API key needed)

State is persisted to paper_state_breakout.json so restarts don't lose positions.
All fills are appended to paper_trades_breakout.csv.

Usage:
    export BINANCE_API_KEY=your_key
    export BINANCE_API_SECRET=your_secret

    python paper_trade_breakout.py           # normal start
    python paper_trade_breakout.py --reset   # wipe state and start fresh
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

from backtest import backtest_breakout as bb   # reuse prepare(), _apply_params()

# ── Config ────────────────────────────────────────────────────────────────────
USE_TESTNET       = True        # True = Binance Testnet; False = pure simulation
INITIAL_CAPITAL   = 10_000.0   # virtual USDT per coin
WARMUP_1H         = 300        # 1h bars fetched for indicator warmup
WARMUP_1D         = 500        # 1d bars fetched for EMA warmup
SLEEP_BUFFER_SEC  = 15         # seconds to wait after candle close before processing

STATE_FILE       = _HERE / "paper_state_breakout.json"
TRADE_LOG_FILE   = _HERE / "paper_trades_breakout.csv"
BEST_PARAMS_FILE = _ROOT / "results/breakout/best_params.json"

API_KEY    = os.getenv("BINANCE_API_KEY", "")
API_SECRET = os.getenv("BINANCE_API_SECRET", "")

COINS = bb.COINS   # [("BTC/USDT:USDT", "btc"), ...]


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
    raw = ex.fetch_ohlcv(symbol, tf, limit=limit + 1)   # +1 for forming candle
    df  = pd.DataFrame(raw, columns=["ts", "open", "high", "low", "close", "volume"])
    df.index = pd.to_datetime(df["ts"], unit="ms", utc=True)
    df.index.name = "datetime"
    df = df[["open", "high", "low", "close", "volume"]].astype(float)
    # Drop forming (unclosed) candle
    now_hour = pd.Timestamp.now(tz="UTC").floor("h")
    df = df[df.index < now_hour]
    return df.tail(limit)


# ── State ─────────────────────────────────────────────────────────────────────
def _default_state() -> dict:
    return {
        "capital":            INITIAL_CAPITAL,
        "peak_cap":           INITIAL_CAPITAL,
        "in_trade":           False,
        "direction":          None,
        "entry_price":        0.0,
        "sl_price":           0.0,
        "tp_price":           0.0,
        "notional":           0.0,
        "trail_active":       False,
        "trail_sl":           0.0,
        "cooldown_remaining": 0,
        "open_time":          None,
        "trades":             [],
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


# ── Position open ─────────────────────────────────────────────────────────────
def _open_position(cs: dict, row: pd.Series, params: dict, ts):
    d   = cs["direction"]
    ep  = cs["entry_price"]
    cap = cs["capital"]
    atr = float(row["atr"])

    sl_dist     = atr * params.get("SL_MULT", 1.5)
    sl_dist_pct = sl_dist / ep
    leverage    = params.get("LEVERAGE", 10)
    base_risk   = params.get("BASE_RISK", 0.01)
    notional    = min(cap * base_risk / sl_dist_pct, cap * leverage)

    tp_rr = params.get("TP_RR", 3.0)
    if d == "long":
        sp = ep - sl_dist
        tp = ep + sl_dist * tp_rr
    else:
        sp = ep + sl_dist
        tp = ep - sl_dist * tp_rr

    cs.update({
        "in_trade":     True,
        "sl_price":     sp,
        "tp_price":     tp,
        "notional":     notional,
        "trail_active": False,
        "trail_sl":     sp,
        "open_time":    str(ts),
    })
    print(f"    ▶ ENTRY  {d.upper():5s}  price={ep:.4f}  SL={sp:.4f}  TP={tp:.4f}"
          f"  notional=${notional:.0f} ({notional/cap:.2f}x)  @{str(ts)[:16]}")


# ── Per-bar processing ────────────────────────────────────────────────────────
def process_bar(cs: dict, row: pd.Series, params: dict, coin: str, ts) -> list:
    """Process one just-closed candle. Mutates cs in place. Returns fill records."""
    records  = []
    cap      = cs["capital"]
    peak_cap = cs["peak_cap"]

    # ── Exit ─────────────────────────────────────────────────────────────────
    if cs["in_trade"]:
        d  = cs["direction"]
        ep = cs["entry_price"]
        sp = cs["sl_price"]
        tp = cs["tp_price"]
        nt = cs["notional"]

        # Trailing stop
        use_trail = params.get("USE_TRAIL", False)
        if use_trail:
            profit  = (row["close"] - ep) if d == "long" else (ep - row["close"])
            sl_abs  = abs(ep - cs["trail_sl"])
            trig    = params.get("TRAIL_TRIGGER_R", 1.0)
            if not cs["trail_active"] and sl_abs > 0 and profit >= trig * sl_abs:
                cs["trail_active"] = True
            if cs["trail_active"]:
                atr = float(row["atr"])
                tm  = params.get("TRAIL_MULT", 1.0)
                sp = max(sp, row["low"]  - atr * tm) if d == "long" else min(sp, row["high"] + atr * tm)
                cs["sl_price"] = sp

        hit_tp = (row["high"] >= tp if d == "long" else row["low"]  <= tp)
        hit_sl = (row["low"]  <= sp if d == "long" else row["high"] >= sp)

        if hit_tp or hit_sl:
            xp     = tp if hit_tp else sp
            reason = "TP" if hit_tp else "SL"
            pct    = (xp - ep) / ep if d == "long" else (ep - xp) / ep
            pnl    = max(nt * pct - nt * bb.FEE_RATE * 2, -cap)
            cap   += pnl
            peak_cap = max(peak_cap, cap)

            rec = dict(timestamp=str(ts), coin=coin, direction=d,
                       entry_price=ep, exit_price=round(xp, 6),
                       notional=round(nt, 4), exit_reason=reason,
                       pnl_usdt=round(pnl, 4), capital=round(cap, 4))
            records.append(rec)
            _log_trade(rec)
            cs["trades"].append(rec)

            sym = "✓" if pnl >= 0 else "✗"
            print(f"    {sym} EXIT    {d.upper():5s} [{reason}]  exit={xp:.4f}"
                  f"  pnl=${pnl:+.2f}  cap=${cap:.0f}  @{str(ts)[:16]}")

            cs.update({"in_trade": False, "trail_active": False, "open_time": None})
            if reason == "SL":
                cs["cooldown_remaining"] = params.get("COOLDOWN_BARS", 0)

        cs["capital"]  = cap
        cs["peak_cap"] = peak_cap

    # ── Entry ─────────────────────────────────────────────────────────────────
    if not cs["in_trade"]:
        if cs["cooldown_remaining"] > 0:
            cs["cooldown_remaining"] -= 1
            return records

        atr = float(row.get("atr", float("nan")))
        if np.isnan(atr) or atr <= 0:
            return records

        adx_min = params.get("ADX_MIN", 0.0)
        if adx_min > 0:
            adx = float(row.get("adx", float("nan")))
            if np.isnan(adx) or adx < adx_min:
                return records

        if not bool(row.get("vol_ok", True)):
            return records

        trend_up    = bool(row.get("trend_up", False))
        entry_long  = bool(row.get("entry_long", False))
        entry_short = bool(row.get("entry_short", False))

        if   entry_long  and     trend_up: sig = "long"
        elif entry_short and not trend_up: sig = "short"
        else: return records

        cs["direction"]   = sig
        cs["entry_price"] = float(row["close"])
        _open_position(cs, row, params, ts)

    return records


# ── Report ────────────────────────────────────────────────────────────────────
def print_report(state: dict):
    now = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC")
    print(f"\n{'═'*65}")
    print(f"  PAPER PORTFOLIO (Breakout)  |  {now}")
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
        pos   = (f"{cs['direction'].upper()} @ {cs['entry_price']:.4f}"
                 if cs["in_trade"] else "flat")
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
            bb._apply_params(params)

            df_1h = fetch_ohlcv(ex, symbol, "1h", WARMUP_1H)
            df_1d = fetch_ohlcv(ex, symbol, "1d", WARMUP_1D)

            df  = bb.prepare(df_1h, df_1d)
            row = df.iloc[-1]
            ts  = df.index[-1]

            adx_str = f"{row.get('adx', float('nan')):.1f}"
            trend   = "↑" if bool(row.get("trend_up", False)) else "↓"
            vol_ok  = "V✓" if bool(row.get("vol_ok", True)) else "V✗"
            el = "L✓" if bool(row.get("entry_long", False))  else "  "
            es = "S✓" if bool(row.get("entry_short", False)) else "  "
            print(f"  [{coin.upper()}]  close={row['close']:.4f}"
                  f"  adx={adx_str}  trend={trend}  {vol_ok}  {el} {es}")

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
    print("║   PAPER TRADING ENGINE — Breakout Strategy                ║")
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
