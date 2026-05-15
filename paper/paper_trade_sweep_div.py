#!/usr/bin/env python3
"""
Paper Trading Engine — Sweep-Divergence-FVG Strategy (1h)
==========================================================
Runs the EXACT same signal logic as backtest_sweep_div.py against live 1h candles.

State is persisted to paper_state_sweep_div.json between runs.
All fills are appended to paper_trades_sweep_div.csv.

Usage:
    python paper_trade_sweep_div.py           # normal start
    python paper_trade_sweep_div.py --reset   # wipe state and start fresh
"""

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

from backtest import backtest_sweep_div as sd

# ── Config ────────────────────────────────────────────────────────────────────
USE_TESTNET      = True
INITIAL_CAPITAL  = 10_000.0
WARMUP_1H        = 300
SLEEP_BUFFER_SEC = 15

STATE_FILE       = _HERE / "paper_state_sweep_div.json"
TRADE_LOG_FILE   = _HERE / "paper_trades_sweep_div.csv"
BEST_PARAMS_FILE = _ROOT / "results/sweep_div/best_params.json"

API_KEY    = os.getenv("BINANCE_API_KEY", "")
API_SECRET = os.getenv("BINANCE_API_SECRET", "")

COINS = list(sd.COINS)


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

_ex_pub = ccxt.binanceusdm({"enableRateLimit": True})


def fetch_ohlcv(symbol: str, limit: int) -> pd.DataFrame:
    raw = _ex_pub.fetch_ohlcv(symbol, "1h", limit=limit + 1)
    df  = pd.DataFrame(raw, columns=["ts", "open", "high", "low", "close", "volume"])
    df.index = pd.to_datetime(df["ts"], unit="ms", utc=True)
    df.index.name = "datetime"
    df = df[["open", "high", "low", "close", "volume"]].astype(float)
    now_hour = pd.Timestamp.now(tz="UTC").floor("h")
    df = df[df.index < now_hour]
    return df.tail(limit)


# ── State ─────────────────────────────────────────────────────────────────────
def _default_state() -> dict:
    return {
        "capital":           INITIAL_CAPITAL,
        "peak_cap":          INITIAL_CAPITAL,
        "in_trade":          False,
        "direction":         None,
        "entry_price":       0.0,
        "sl_price":          0.0,
        "tp_price":          0.0,
        "notional":          0.0,
        "bars_held":         0,
        "open_time":         None,
        "last_processed_ts": None,
        "trades":            [],
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
def process_bar(cs: dict, row: pd.Series, df: pd.DataFrame, i: int,
                params: dict, coin: str, ts) -> list:
    records  = []
    cap      = cs["capital"]
    peak_cap = cs["peak_cap"]

    # ── Exit ──────────────────────────────────────────────────────────────────
    if cs["in_trade"]:
        cs["bars_held"] = cs.get("bars_held", 0) + 1
        direction = cs["direction"]
        sl_price  = cs["sl_price"]
        tp_price  = cs["tp_price"]
        max_hold  = params.get("MAX_HOLD_BARS", 48)

        hit_tp  = row["high"] >= tp_price if direction == "long" else row["low"]  <= tp_price
        hit_sl  = row["low"]  <= sl_price if direction == "long" else row["high"] >= sl_price
        expired = max_hold > 0 and cs["bars_held"] >= max_hold

        if hit_tp or hit_sl or expired:
            if hit_tp:
                exit_price, exit_reason = tp_price, "TP"
            elif hit_sl:
                exit_price, exit_reason = sl_price, "SL"
            else:
                exit_price, exit_reason = float(row["close"]), "TIMEOUT"

            nt  = cs["notional"]
            ep  = cs["entry_price"]
            pct = (exit_price - ep) / ep if direction == "long" else (ep - exit_price) / ep
            pnl = max(nt * pct - nt * sd.FEE_RATE * 2, -cap)
            cap += pnl
            peak_cap = max(peak_cap, cap)

            rec = dict(
                timestamp=str(ts), coin=coin, direction=direction,
                entry_price=round(ep, 6), exit_price=round(exit_price, 6),
                notional=round(nt, 4), exit_reason=exit_reason,
                pnl_usdt=round(pnl, 4), capital=round(cap, 4),
            )
            records.append(rec)
            _log_trade(rec)
            cs["trades"].append(rec)
            sym = "✓" if pnl >= 0 else "✗"
            print(f"  [{coin.upper()}] {sym} EXIT {direction.upper()}"
                  f" [{exit_reason}]  exit={exit_price:.4f}"
                  f"  pnl=${pnl:+.2f}  cap=${cap:.0f}  @{str(ts)[:16]}")
            cs.update({"in_trade": False, "open_time": None, "bars_held": 0})

        cs["capital"]  = cap
        cs["peak_cap"] = peak_cap

    # ── Entry ──────────────────────────────────────────────────────────────────
    if not cs["in_trade"]:
        atr = float(row.get("atr", float("nan")))
        if np.isnan(atr) or atr <= 0:
            return records

        sweep_l = bool(row.get("sweep_long",  False))
        sweep_s = bool(row.get("sweep_short", False))
        if not sweep_l and not sweep_s:
            return records

        direction = "long" if sweep_l else "short"

        # Divergence filter
        if params.get("USE_DIV_FILTER", True):
            div_ok = bool(row.get("div_bull", False)) if direction == "long" \
                     else bool(row.get("div_bear", False))
            if not div_ok:
                return records

        # FVG filter
        if params.get("USE_FVG_FILTER", True):
            if not sd._find_recent_fvg(df, i, direction):
                return records

        ep       = float(row["close"])
        sl_atr   = params.get("SL_ATR_MULT", 0.5)
        sl_price = ep - atr * sl_atr if direction == "long" else ep + atr * sl_atr

        if direction == "long"  and sl_price >= ep: return records
        if direction == "short" and sl_price <= ep: return records

        sl_dist = abs(ep - sl_price)
        if sl_dist < 1e-8:
            return records

        tp_rr    = params.get("TP_RR", 3.0)
        tp_price = ep + sl_dist * tp_rr if direction == "long" else ep - sl_dist * tp_rr

        leverage  = params.get("LEVERAGE", 5)
        base_risk = params.get("BASE_RISK", 0.01)
        sl_pct    = sl_dist / ep
        notional  = min(cap * base_risk / sl_pct, cap * leverage)
        if notional < 1:
            return records

        cap -= notional * sd.FEE_RATE
        peak_cap = max(peak_cap, cap)

        cs.update({
            "in_trade":    True,
            "direction":   direction,
            "entry_price": ep,
            "sl_price":    sl_price,
            "tp_price":    tp_price,
            "notional":    notional,
            "bars_held":   0,
            "open_time":   str(ts),
            "capital":     cap,
            "peak_cap":    peak_cap,
        })
        print(f"  [{coin.upper()}] ▶ ENTRY {direction.upper()}"
              f"  price={ep:.4f}  SL={sl_price:.4f}  TP={tp_price:.4f}"
              f"  notional=${notional:.0f}  @{str(ts)[:16]}")

    return records


# ── Report ────────────────────────────────────────────────────────────────────
def print_report(state: dict):
    now = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC")
    print(f"\n{'═'*65}")
    print(f"  PAPER PORTFOLIO (Sweep-Div)  |  {now}")
    print(f"{'═'*65}")
    total_cap = 0.0
    for _, coin in COINS:
        cs   = state[coin]
        cap  = cs["capital"]
        ret  = (cap - INITIAL_CAPITAL) / INITIAL_CAPITAL * 100
        total_cap += cap
        n    = len(cs["trades"])
        wins = sum(1 for t in cs["trades"] if t["pnl_usdt"] > 0)
        wr   = f"{wins/n*100:.0f}%" if n else " - "
        pos  = (f"{cs['direction'].upper()} @ {cs['entry_price']:.4f}"
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
            sd._apply_params(params)
            df_raw = fetch_ohlcv(symbol, WARMUP_1H)
            df     = sd.prepare(df_raw)
            cs     = state[coin]

            last_ts = cs.get("last_processed_ts")
            if last_ts:
                missed = df[df.index > pd.Timestamp(last_ts, tz="UTC")]
                if len(missed) == 0:
                    missed = df.iloc[-1:]
            else:
                missed = df.iloc[-1:]

            if len(missed) > 1:
                print(f"  [{coin.upper()}] replaying {len(missed)} missed candles "
                      f"from {str(missed.index[0])[:16]}")

            for ts, row in missed.iterrows():
                i = df.index.get_loc(ts)
                sl = bool(row.get("sweep_long", False))
                ss = bool(row.get("sweep_short", False))
                db = bool(row.get("div_bull", False))
                dbe = bool(row.get("div_bear", False))
                print(f"  [{coin.upper()}]  close={row['close']:.4f}"
                      f"  atr={row.get('atr', 0):.4f}"
                      f"  sweep={'L' if sl else 'S' if ss else '-'}"
                      f"  div={'↑' if db else '↓' if dbe else '-'}")
                process_bar(cs, row, df, i, params, coin, ts)
                cs["last_processed_ts"] = str(ts)

        except ccxt.NetworkError as e:
            print(f"  [{coin.upper()}] network error: {e}")
        except Exception:
            print(f"  [{coin.upper()}] unexpected error:")
            traceback.print_exc()

    save_state(state)
    print_report(state)


# ── Intrabar SL monitor ───────────────────────────────────────────────────────
INTRABAR_CHECK_INTERVAL = 60

def check_intrabar_sl(state: dict):
    changed = False
    for symbol, coin in COINS:
        cs = state[coin]
        if not cs.get("in_trade"):
            continue
        try:
            raw    = _ex_pub.fetch_ohlcv(symbol, "1h", limit=1)
            f_high = float(raw[-1][2])
            f_low  = float(raw[-1][3])

            d  = cs["direction"]
            sl = cs["sl_price"]
            tp = cs["tp_price"]
            ep = cs["entry_price"]

            hit_sl = f_low  <= sl if d == "long" else f_high >= sl
            hit_tp = f_high >= tp if d == "long" else f_low  <= tp

            if hit_sl or hit_tp:
                exit_price  = sl if hit_sl else tp
                exit_reason = "SL" if hit_sl else "TP"
                nt   = cs["notional"]
                cap  = cs["capital"]
                pct  = (exit_price - ep) / ep if d == "long" else (ep - exit_price) / ep
                pnl  = max(nt * pct - nt * sd.FEE_RATE * 2, -cap)
                cap += pnl
                peak = max(cs["peak_cap"], cap)

                rec = dict(
                    timestamp=datetime.now(timezone.utc).isoformat(),
                    coin=coin, direction=d,
                    entry_price=round(ep, 6), exit_price=round(exit_price, 6),
                    notional=round(nt, 4), exit_reason=exit_reason + "_INTRABAR",
                    pnl_usdt=round(pnl, 4), capital=round(cap, 4),
                )
                _log_trade(rec)
                cs["trades"].append(rec)
                sym = "✓" if pnl >= 0 else "✗"
                print(f"  [{coin.upper()}] {sym} INTRABAR {exit_reason}"
                      f"  exit={exit_price:.4f}  pnl=${pnl:+.2f}  cap=${cap:.0f}")
                cs.update({"in_trade": False, "open_time": None, "bars_held": 0,
                           "capital": cap, "peak_cap": peak})
                changed = True

        except Exception:
            pass
    return changed


# ── Entry point ───────────────────────────────────────────────────────────────
def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--reset", action="store_true")
    args = parser.parse_args()

    if args.reset:
        STATE_FILE.unlink(missing_ok=True)
        TRADE_LOG_FILE.unlink(missing_ok=True)
        print("State reset.")

    ex    = make_exchange()
    state = load_state()

    if not BEST_PARAMS_FILE.exists():
        print(f"ERROR: {BEST_PARAMS_FILE} not found. Run tune first.")
        sys.exit(1)

    while True:
        best = json.loads(BEST_PARAMS_FILE.read_text())
        now  = datetime.now(timezone.utc)
        next_hour = (now + timedelta(hours=1)).replace(minute=0, second=0, microsecond=0)
        wait_sec  = (next_hour - now).total_seconds() + SLEEP_BUFFER_SEC

        print(f"\n{'╔'+'═'*63+'╗'}")
        print(f"  ⏱  Next cycle in {wait_sec/60:.1f} min  ({next_hour.strftime('%H:%M:%S')} UTC)")
        print(f"{'╚'+'═'*63+'╝'}")

        # Intrabar SL checks while waiting
        deadline = time.time() + wait_sec
        while time.time() < deadline:
            time.sleep(min(INTRABAR_CHECK_INTERVAL, deadline - time.time()))
            if check_intrabar_sl(state):
                save_state(state)

        run_cycle(ex, state, best)


if __name__ == "__main__":
    main()
