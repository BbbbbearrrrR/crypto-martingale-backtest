#!/usr/bin/env python3
"""
Paper Trading Engine — Bollinger Band Scalping Strategy
========================================================
Runs the EXACT same signal logic as backtest_boll_scalp.py against live 5m candles.
No real money involved.

Strategy:
  - Entry: close crosses BB lower/upper AND close is above/below 200-EMA (trend filter)
  - SL   : swing low/high ± ATR buffer
  - TP1  : bb_mid (50% position closed, SL moved to breakeven)
  - TP2  : opposite band (remaining 50%)
  - Timeout: MAX_HOLD_BARS bars

Coins tracked: ETH, SOL, HYPE, SUI  (BTC excluded — poor backtest results)

Usage:
    python paper_trade_boll_scalp.py           # normal start
    python paper_trade_boll_scalp.py --reset   # wipe state and start fresh
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

from backtest import backtest_boll_scalp as bs

# ── Config ────────────────────────────────────────────────────────────────────
USE_TESTNET      = True
INITIAL_CAPITAL  = 10_000.0
WARMUP_5M        = 500          # 5m bars for indicator warmup
SLEEP_BUFFER_SEC = 10           # seconds after candle close before processing

STATE_FILE       = _HERE / "paper_state_boll_scalp.json"
TRADE_LOG_FILE   = _HERE / "paper_trades_boll_scalp.csv"
BEST_PARAMS_FILE = _ROOT / "results/boll_scalp/best_params.json"

API_KEY    = os.getenv("BINANCE_API_KEY", "")
API_SECRET = os.getenv("BINANCE_API_SECRET", "")

# BTC excluded (calmar < 1 in backtest)
COINS = [(sym, coin) for sym, coin in bs.COINS if coin != "btc"]


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


# Public market data exchange — never uses testnet, reliable for intrabar checks
_ex_pub = ccxt.binanceusdm({"enableRateLimit": True})


def fetch_ohlcv(ex, symbol: str, limit: int) -> pd.DataFrame:
    raw = _ex_pub.fetch_ohlcv(symbol, "5m", limit=limit + 1)
    df  = pd.DataFrame(raw, columns=["ts", "open", "high", "low", "close", "volume"])
    df.index = pd.to_datetime(df["ts"], unit="ms", utc=True)
    df.index.name = "datetime"
    df = df[["open", "high", "low", "close", "volume"]].astype(float)
    # Drop forming (unclosed) candle
    now_5m = pd.Timestamp.now(tz="UTC").floor("5min")
    df = df[df.index < now_5m]
    return df.tail(limit)


# ── State ─────────────────────────────────────────────────────────────────────
def _default_state() -> dict:
    return {
        "capital":      INITIAL_CAPITAL,
        "peak_cap":     INITIAL_CAPITAL,
        "in_trade":     False,
        "direction":    None,
        "entry_price":  0.0,
        "sl_price":     0.0,
        "tp1_price":    0.0,
        "tp2_price":    0.0,
        "notional":     0.0,
        "notional_rem": 0.0,
        "partial_done": False,
        "bars_held":    0,
        "open_time":    None,
        "last_processed_ts": None,
        "trades":       [],
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
    """Process one just-closed 5m candle. Mutates cs in place. Returns fill records."""
    records  = []
    cap      = cs["capital"]
    peak_cap = cs["peak_cap"]

    fee_rate      = bs.FEE_RATE
    use_partial   = params.get("USE_PARTIAL_TP", True)
    max_hold      = params.get("MAX_HOLD_BARS", 24)

    # ── Exit ─────────────────────────────────────────────────────────────────
    if cs["in_trade"]:
        cs["bars_held"] += 1
        d   = cs["direction"]
        ep  = cs["entry_price"]
        sp  = cs["sl_price"]
        tp1 = cs["tp1_price"]
        tp2 = cs["tp2_price"]
        nt_rem = cs["notional_rem"]

        hit_tp1 = (not cs["partial_done"] and use_partial and
                   (row["high"] >= tp1 if d == "long" else row["low"] <= tp1))
        hit_tp2 = (row["high"] >= tp2 if d == "long" else row["low"] <= tp2)
        hit_sl  = (row["low"]  <= sp  if d == "long" else row["high"] >= sp)
        expired = cs["bars_held"] >= max_hold

        # Partial TP1 — close 50%
        if hit_tp1 and not cs["partial_done"]:
            half = nt_rem * 0.5
            pct  = ((tp1 - ep) / ep if d == "long" else (ep - tp1) / ep)
            pnl  = half * pct - half * fee_rate
            cap += pnl
            peak_cap = max(peak_cap, cap)
            rec = dict(timestamp=str(ts), coin=coin, direction=d,
                       entry_price=ep, exit_price=round(tp1, 6),
                       notional=round(half, 4), exit_reason="TP1",
                       pnl_usdt=round(pnl, 4), capital=round(cap, 4))
            records.append(rec)
            _log_trade(rec)
            cs["trades"].append(rec)
            cs["notional_rem"] -= half
            cs["partial_done"]  = True
            cs["sl_price"]      = ep   # move SL to breakeven
            sp = ep
            nt_rem = cs["notional_rem"]
            print(f"    ½ TP1    {d.upper():5s}  exit={tp1:.4f}"
                  f"  pnl=${pnl:+.2f}  cap=${cap:.0f}  @{str(ts)[:16]}")

        # Full exit: TP2, SL, or timeout
        if cs["in_trade"] and (hit_tp2 or hit_sl or expired):
            if hit_tp2:
                xp, reason = tp2, "TP2"
            elif hit_sl:
                xp, reason = sp, "SL"
            else:
                xp, reason = float(row["close"]), "TIMEOUT"

            pct  = ((xp - ep) / ep if d == "long" else (ep - xp) / ep)
            pnl  = max(nt_rem * pct - nt_rem * fee_rate * 2, -cap)
            cap += pnl
            peak_cap = max(peak_cap, cap)
            rec = dict(timestamp=str(ts), coin=coin, direction=d,
                       entry_price=ep, exit_price=round(xp, 6),
                       notional=round(nt_rem, 4), exit_reason=reason,
                       pnl_usdt=round(pnl, 4), capital=round(cap, 4))
            records.append(rec)
            _log_trade(rec)
            cs["trades"].append(rec)

            sym = "✓" if pnl >= 0 else "✗"
            print(f"    {sym} EXIT    {d.upper():5s} [{reason}]  exit={xp:.4f}"
                  f"  pnl=${pnl:+.2f}  cap=${cap:.0f}  @{str(ts)[:16]}")
            cs.update({"in_trade": False, "partial_done": False,
                       "bars_held": 0, "open_time": None})

        cs["capital"]  = cap
        cs["peak_cap"] = peak_cap

    # ── Entry ─────────────────────────────────────────────────────────────────
    if not cs["in_trade"]:
        atr = float(row.get("atr", float("nan")))
        if np.isnan(atr) or atr <= 0:
            return records
        if pd.isna(row.get("bb_lower")) or pd.isna(row.get("bb_upper")):
            return records

        trend_up    = bool(row.get("trend_up", False))
        go_long     = bool(row.get("entry_long", False))  and trend_up
        go_short    = bool(row.get("entry_short", False)) and not trend_up

        if not go_long and not go_short:
            return records

        d   = "long" if go_long else "short"
        ep  = float(row["close"])

        sl_tp_ratio = params.get("SL_TP_RATIO", 0.5)
        if d == "long":
            tp1 = float(row["bb_mid"])
            tp2 = float(row["bb_upper"])
            sl  = ep - abs(tp1 - ep) * sl_tp_ratio
        else:
            tp1 = float(row["bb_mid"])
            tp2 = float(row["bb_lower"])
            sl  = ep + abs(tp1 - ep) * sl_tp_ratio

        # Sanity checks
        if d == "long"  and (tp1 <= ep or tp2 <= ep):
            return records
        if d == "short" and (tp1 >= ep or tp2 >= ep):
            return records
        # SL must be on the losing side
        if d == "long"  and sl >= ep:
            return records
        if d == "short" and sl <= ep:
            return records

        sl_pct = abs(ep - sl) / ep
        if sl_pct < 1e-6:
            return records

        leverage  = params.get("LEVERAGE", 5)
        base_risk = params.get("BASE_RISK", 0.02)
        cap       = cs["capital"]
        notional  = min(cap * base_risk / sl_pct, cap * leverage)

        cs.update({
            "in_trade":     True,
            "direction":    d,
            "entry_price":  ep,
            "sl_price":     sl,
            "tp1_price":    tp1,
            "tp2_price":    tp2,
            "notional":     notional,
            "notional_rem": notional,
            "partial_done": False,
            "bars_held":    0,
            "open_time":    str(ts),
        })
        print(f"    ▶ ENTRY  {d.upper():5s}  price={ep:.4f}  SL={sl:.4f}"
              f"  TP1={tp1:.4f}  TP2={tp2:.4f}  notional=${notional:.0f}"
              f"  @{str(ts)[:16]}")

    return records


# ── Report ────────────────────────────────────────────────────────────────────
def print_report(state: dict):
    now = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC")
    print(f"\n{'═'*65}")
    print(f"  PAPER PORTFOLIO (Boll Scalp)  |  {now}")
    print(f"{'═'*65}")
    total_start = INITIAL_CAPITAL * len(COINS)
    total_cap   = 0.0
    for _, coin in COINS:
        cs  = state[coin]
        cap = cs["capital"]
        ret = (cap - INITIAL_CAPITAL) / INITIAL_CAPITAL * 100
        total_cap += cap
        n    = len(cs["trades"])
        wins = sum(1 for t in cs["trades"] if t["pnl_usdt"] > 0)
        wr   = f"{wins/n*100:.0f}%" if n else " - "
        pos  = (f"{cs['direction'].upper()} @ {cs['entry_price']:.4f}"
                if cs["in_trade"] else "flat")
        print(f"  {coin.upper():5s}  cap=${cap:>9.2f}  ret={ret:>+7.2f}%"
              f"  trades={n:>3d}  wr={wr:>4s}  [{pos}]")
    total_ret = (total_cap - total_start) / total_start * 100
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
            bs._apply_params(params)

            df  = bs.prepare(fetch_ohlcv(ex, symbol, WARMUP_5M))
            cs  = state[coin]

            # Replay all candles missed while process was down
            last_ts = cs.get("last_processed_ts")
            if last_ts:
                missed = df[df.index > pd.Timestamp(last_ts, tz="UTC")]
                if len(missed) == 0:
                    missed = df.iloc[-1:]
            else:
                missed = df.iloc[-1:]

            if len(missed) > 1:
                print(f"  [{coin.upper()}] ⚠ replaying {len(missed)} missed candles "
                      f"from {str(missed.index[0])[:16]}")

            for ts, row in missed.iterrows():
                trend   = "↑" if bool(row.get("trend_up", False)) else "↓"
                el = "L✓" if bool(row.get("entry_long", False))  else "  "
                es = "S✓" if bool(row.get("entry_short", False)) else "  "
                print(f"  [{coin.upper()}]  close={row['close']:.4f}"
                      f"  bb=[{row['bb_lower']:.2f},{row['bb_upper']:.2f}]"
                      f"  trend={trend}  {el} {es}")
                process_bar(cs, row, params, coin, ts)
                cs["last_processed_ts"] = str(ts)

        except ccxt.NetworkError as e:
            print(f"  [{coin.upper()}] network error: {e}  — retry next cycle")
        except Exception:
            print(f"  [{coin.upper()}] unexpected error:")
            traceback.print_exc()

    save_state(state)
    print_report(state)


# ── Intrabar SL monitor ───────────────────────────────────────────────────────
INTRABAR_CHECK_INTERVAL = 30  # seconds (5m candle, check every 30s)

def check_intrabar_sl(ex, state: dict, best: dict):
    """Check forming 5m candle high/low against SL for all open positions."""
    changed = False
    for symbol, coin in COINS:
        cs = state[coin]
        if not cs.get("in_trade"):
            continue
        try:
            raw    = _ex_pub.fetch_ohlcv(symbol, "5m", limit=1)
            f_high = float(raw[-1][2])
            f_low  = float(raw[-1][3])
            d      = cs["direction"]
            sp     = cs["sl_price"]
            ep     = cs["entry_price"]
            nt_rem = cs["notional_rem"]
            tp1    = cs["tp1_price"]
            cap    = cs["capital"]
            params = best.get(coin, {}).get("params", {})
            use_partial = params.get("USE_PARTIAL_TP", True)

            # Check TP1 first (may move SL to breakeven)
            hit_tp1 = (not cs["partial_done"] and use_partial and
                       (f_high >= tp1 if d == "long" else f_low <= tp1))
            if hit_tp1:
                half = nt_rem * 0.5
                pct  = (tp1 - ep) / ep if d == "long" else (ep - tp1) / ep
                pnl  = half * pct - half * bs.FEE_RATE
                cap += pnl
                ts_now = datetime.now(timezone.utc)
                rec = dict(timestamp=str(ts_now), coin=coin, direction=d,
                           entry_price=ep, exit_price=round(tp1, 6),
                           notional=round(half, 4), exit_reason="TP1",
                           pnl_usdt=round(pnl, 4), capital=round(cap, 4))
                cs["trades"].append(rec)
                _log_trade(rec)
                cs["notional_rem"] -= half
                cs["partial_done"]  = True
                cs["sl_price"]      = ep   # breakeven
                cs["capital"]       = cap
                cs["peak_cap"]      = max(cs["peak_cap"], cap)
                sp = ep
                nt_rem = cs["notional_rem"]
                print(f"  ⚡ INTRABAR TP1  {coin.upper():5s}  {d.upper()}"
                      f"  tp1={tp1:.4f}  pnl=${pnl:+.2f}  cap=${cap:.0f}")
                changed = True

            # Check SL
            hit_sl = (f_low <= sp if d == "long" else f_high >= sp)
            if not hit_sl:
                continue

            pct = (sp - ep) / ep if d == "long" else (ep - sp) / ep
            pnl = max(nt_rem * pct - nt_rem * bs.FEE_RATE * 2, -cap)
            cap += pnl
            ts_now = datetime.now(timezone.utc)
            rec = dict(timestamp=str(ts_now), coin=coin, direction=d,
                       entry_price=ep, exit_price=round(sp, 6),
                       notional=round(nt_rem, 4), exit_reason="SL",
                       pnl_usdt=round(pnl, 4), capital=round(cap, 4))
            cs["trades"].append(rec)
            _log_trade(rec)
            print(f"  ⚡ INTRABAR SL  {coin.upper():5s}  {d.upper()}"
                  f"  high={f_high:.4f}  low={f_low:.4f}  sl={sp:.4f}"
                  f"  pnl=${pnl:+.2f}  cap=${cap:.0f}")
            cs["capital"]  = cap
            cs["peak_cap"] = max(cs["peak_cap"], cap)
            cs.update({"in_trade": False, "partial_done": False,
                       "bars_held": 0, "open_time": None})
            changed = True
        except Exception as e:
            print(f"  [{coin.upper()}] intrabar check error: {e}")
    if changed:
        save_state(state)


# ── Scheduling ────────────────────────────────────────────────────────────────
def seconds_to_next_candle() -> float:
    now      = datetime.now(timezone.utc)
    next_5m  = now.replace(second=0, microsecond=0)
    next_5m += timedelta(minutes=(5 - next_5m.minute % 5))
    return max((next_5m - now).total_seconds() + SLEEP_BUFFER_SEC, 0)


# ── Entry point ───────────────────────────────────────────────────────────────
def main():
    reset = "--reset" in sys.argv

    print("╔═══════════════════════════════════════════════════════════╗")
    print("║   PAPER TRADING ENGINE — Boll Scalp (5m)                  ║")
    print(f"║   Testnet : {str(USE_TESTNET):<49}║")
    print(f"║   Capital : ${INITIAL_CAPITAL:,.0f} / coin{' '*(44 - len(f'{INITIAL_CAPITAL:,.0f}'))}║")
    print(f"║   Coins   : {', '.join(c for _, c in COINS):<49}║")
    print(f"║   State   : {str(STATE_FILE.name):<49}║")
    print(f"║   Trades  : {str(TRADE_LOG_FILE.name):<49}║")
    print("╚═══════════════════════════════════════════════════════════╝\n")

    if not BEST_PARAMS_FILE.exists():
        print(f"ERROR: {BEST_PARAMS_FILE} not found. Run tune.py --strategy boll_scalp first.")
        sys.exit(1)

    if reset:
        STATE_FILE.unlink(missing_ok=True)
        print("State reset.\n")

    best  = json.loads(BEST_PARAMS_FILE.read_text())
    state = load_state()
    ex    = make_exchange()

    while True:
        wait = seconds_to_next_candle()
        nxt  = (datetime.now(timezone.utc) + timedelta(seconds=wait)).strftime("%H:%M:%S UTC")
        print(f"  ⏱  Next cycle in {wait/60:.1f} min  ({nxt})")
        slept = 0
        while slept < wait - 1:
            chunk  = min(INTRABAR_CHECK_INTERVAL, wait - slept)
            time.sleep(chunk)
            slept += chunk
            if slept < wait - 1:
                check_intrabar_sl(ex, state, best)
        best = json.loads(BEST_PARAMS_FILE.read_text())
        run_cycle(ex, state, best)


if __name__ == "__main__":
    main()
