#!/usr/bin/env python3
"""
Live Trading Engine — Bollinger Band Scalping Strategy (5m)
============================================================
Entry  : MARKET order
SL     : STOP_MARKET (full qty, reduceOnly) at sl_price
TP1    : TAKE_PROFIT_MARKET (half qty, reduceOnly) at bb_mid
TP2    : TAKE_PROFIT_MARKET (half qty, reduceOnly) at bb_upper/lower

After TP1 hits (position halved): cancel original SL → place new STOP_MARKET
  at entry price (breakeven) for remaining half.

Usage:
    python live/live_trade_boll_scalp.py           # start
    python live/live_trade_boll_scalp.py --reset   # wipe state and restart
    python live/live_trade_boll_scalp.py --dry-run # signals only, no orders
"""

import os
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")

import json, sys, time, traceback, math
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
USE_TESTNET       = True
DRY_RUN           = "--dry-run" in sys.argv
# Your total account balance allocated to this strategy.
# Each coin gets INITIAL_CAPITAL / len(COINS) as its virtual starting capital
# so that simultaneous entries across coins don't over-allocate.
INITIAL_CAPITAL   = 10_000.0
WARMUP_5M         = 500
SLEEP_BUFFER_SEC  = 10
INTRABAR_INTERVAL = 30      # seconds between intrabar position checks

STATE_FILE        = _HERE / "live_state_boll_scalp.json"
TRADE_LOG_FILE    = _HERE / "live_trades_boll_scalp.csv"
BEST_PARAMS_FILE  = _ROOT / "results/boll_scalp/best_params.json"

API_KEY    = os.getenv("BINANCE_TESTNET_API_KEY", "")
API_SECRET = os.getenv("BINANCE_TESTNET_API_SECRET", "")

COINS = list(bs.COINS)


# ── Exchange ──────────────────────────────────────────────────────────────────
def make_exchange() -> ccxt.binanceusdm:
    ex = ccxt.binanceusdm({
        "apiKey":          API_KEY,
        "secret":          API_SECRET,
        "enableRateLimit": True,
        "options":         {"defaultType": "future"},
    })
    if USE_TESTNET:
        ex.set_sandbox_mode(True)
    return ex


_ex_pub = ccxt.binanceusdm({"enableRateLimit": True})


def fetch_ohlcv(symbol: str, limit: int) -> pd.DataFrame:
    raw = _ex_pub.fetch_ohlcv(symbol, "5m", limit=limit + 1)
    df  = pd.DataFrame(raw, columns=["ts", "open", "high", "low", "close", "volume"])
    df.index = pd.to_datetime(df["ts"], unit="ms", utc=True)
    df.index.name = "datetime"
    df = df[["open", "high", "low", "close", "volume"]].astype(float)
    now_5m = pd.Timestamp.now(tz="UTC").floor("5min")
    return df[df.index < now_5m].tail(limit)


# ── Exchange order helpers ─────────────────────────────────────────────────────
def _round_qty(ex: ccxt.binanceusdm, symbol: str, qty: float) -> float:
    step = float(ex.market(symbol)["precision"]["amount"])
    return math.floor(qty / step) * step


def _round_price(ex: ccxt.binanceusdm, symbol: str, price: float) -> float:
    tick = float(ex.market(symbol)["precision"]["price"])
    return round(round(price / tick) * tick, 10)


def place_entry(ex: ccxt.binanceusdm, symbol: str, direction: str,
                qty: float, sl_price: float,
                tp1_price: float, tp2_price: float) -> dict:
    """
    Market entry + STOP_MARKET SL (full qty) + two TAKE_PROFIT_MARKET orders
    (each for half qty) at TP1 and TP2.
    """
    side       = "buy"  if direction == "long" else "sell"
    close_side = "sell" if direction == "long" else "buy"

    qty_full = _round_qty(ex, symbol, qty)
    qty_half = _round_qty(ex, symbol, qty / 2)
    sl_price  = _round_price(ex, symbol, sl_price)
    tp1_price = _round_price(ex, symbol, tp1_price)
    tp2_price = _round_price(ex, symbol, tp2_price)

    if qty_full <= 0 or qty_half <= 0:
        raise ValueError(f"qty rounded to 0 for {symbol}: full={qty_full} half={qty_half}")

    if DRY_RUN:
        print(f"    [DRY-RUN] {side.upper()} {qty_full} {symbol}"
              f"  SL={sl_price}  TP1={tp1_price}  TP2={tp2_price}")
        return {"entry_id": "dry", "sl_id": "dry", "tp1_id": "dry", "tp2_id": "dry",
                "qty_full": qty_full, "qty_half": qty_half}

    entry_order = ex.create_market_order(symbol, side, qty_full)

    sl_order = ex.create_order(
        symbol, "stop_market", close_side, qty_full, None,
        {"stopPrice": sl_price, "reduceOnly": True, "closePosition": False},
    )
    tp1_order = ex.create_order(
        symbol, "take_profit_market", close_side, qty_half, None,
        {"stopPrice": tp1_price, "reduceOnly": True, "closePosition": False},
    )
    tp2_order = ex.create_order(
        symbol, "take_profit_market", close_side, qty_half, None,
        {"stopPrice": tp2_price, "reduceOnly": True, "closePosition": False},
    )

    return {
        "entry_id": entry_order["id"],
        "sl_id":    sl_order["id"],
        "tp1_id":   tp1_order["id"],
        "tp2_id":   tp2_order["id"],
        "qty_full": qty_full,
        "qty_half": qty_half,
    }


def place_breakeven_sl(ex: ccxt.binanceusdm, symbol: str, direction: str,
                       qty_half: float, entry_price: float,
                       old_sl_id: str) -> str:
    """Cancel old SL and place new STOP_MARKET at entry (breakeven)."""
    close_side = "sell" if direction == "long" else "buy"

    if DRY_RUN:
        print(f"    [DRY-RUN] MOVE SL to breakeven {entry_price}")
        return "dry"

    # Cancel original SL
    try:
        ex.cancel_order(old_sl_id, symbol)
    except Exception as e:
        print(f"    [WARN] cancel old SL {old_sl_id}: {e}")

    sl_price = _round_price(ex, symbol, entry_price)
    qty_half = _round_qty(ex, symbol, qty_half)
    new_sl = ex.create_order(
        symbol, "stop_market", close_side, qty_half, None,
        {"stopPrice": sl_price, "reduceOnly": True, "closePosition": False},
    )
    return new_sl["id"]


def cancel_open_orders(ex: ccxt.binanceusdm, symbol: str):
    if DRY_RUN:
        return
    try:
        ex.cancel_all_orders(symbol)
    except Exception as e:
        print(f"    [WARN] cancel_all_orders {symbol}: {e}")


def fetch_position_contracts(ex: ccxt.binanceusdm, symbol: str) -> float:
    """Return absolute number of open contracts (0.0 if flat)."""
    try:
        for p in ex.fetch_positions([symbol]):
            c = abs(float(p.get("contracts") or 0))
            if c > 0:
                return c
    except Exception as e:
        print(f"    [WARN] fetch_positions {symbol}: {e}")
    return 0.0


def fetch_balance_usdt(ex: ccxt.binanceusdm) -> float:
    bal = ex.fetch_balance()
    return float(bal["USDT"]["free"] + bal["USDT"]["used"])


# ── State ─────────────────────────────────────────────────────────────────────
def _per_coin_capital() -> float:
    return INITIAL_CAPITAL / len(COINS)


def _default_state() -> dict:
    cap = _per_coin_capital()
    return {
        "capital":           cap,
        "peak_cap":          cap,
        "in_trade":          False,
        "direction":         None,
        "entry_price":       0.0,
        "sl_price":          0.0,
        "tp1_price":         0.0,
        "tp2_price":         0.0,
        "qty_full":          0.0,
        "qty_half":          0.0,
        "sl_order_id":       None,
        "tp1_order_id":      None,
        "tp2_order_id":      None,
        "partial_done":      False,
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


# ── Sync position from exchange ────────────────────────────────────────────────
def sync_position(ex: ccxt.binanceusdm, cs: dict, symbol: str, coin: str) -> bool:
    """
    Check exchange position vs local state.
    - Position halved → TP1 hit → move SL to breakeven.
    - Position gone   → TP2 or SL hit → close out locally.
    Returns True if state changed.
    """
    if not cs["in_trade"]:
        return False

    contracts = fetch_position_contracts(ex, symbol)
    changed   = False

    # ─ TP1: position halved ──────────────────────────────────────────────────
    if not cs["partial_done"] and contracts > 0:
        expected_full = cs["qty_full"]
        expected_half = cs["qty_half"]
        # Allow 10% tolerance for rounding
        if contracts < expected_full * 0.7 and contracts >= expected_half * 0.5:
            # TP1 was hit on exchange
            tp1 = cs["tp1_price"]
            ep  = cs["entry_price"]
            d   = cs["direction"]
            pct = (tp1 - ep) / ep if d == "long" else (ep - tp1) / ep

            # Update virtual capital for TP1 half
            half_notional = cs["qty_half"] * ep
            pct_tp1 = (tp1 - ep) / ep if d == "long" else (ep - tp1) / ep
            pnl_tp1 = half_notional * pct_tp1 - half_notional * bs.FEE_RATE
            cs["capital"]  += pnl_tp1
            cs["peak_cap"]  = max(cs["peak_cap"], cs["capital"])

            ts_now = datetime.now(timezone.utc)
            rec = dict(
                timestamp   = str(ts_now),
                coin        = coin,
                direction   = d,
                entry_price = ep,
                exit_price  = round(tp1, 6),
                qty         = expected_half,
                exit_reason = "TP1",
                pnl_usdt    = round(pnl_tp1, 4),
                capital     = round(cs["capital"], 4),
            )
            cs["trades"].append(rec)
            _log_trade(rec)
            print(f"  [{coin.upper()}] ½ TP1 detected by exchange"
                  f"  tp1={tp1:.4f}  pnl=${pnl_tp1:+.2f}  cap=${cs['capital']:.0f}  @{str(ts_now)[:16]}")

            # Move SL to breakeven
            new_sl_id = place_breakeven_sl(
                ex, symbol, d, cs["qty_half"], ep, cs["sl_order_id"])
            cs["sl_order_id"] = new_sl_id
            cs["sl_price"]    = ep
            cs["partial_done"] = True
            changed = True

    # ─ Position fully closed ──────────────────────────────────────────────────
    if contracts == 0:
        ep  = cs["entry_price"]
        d   = cs["direction"]
        # Infer exit price from recent trades
        exit_price  = None
        exit_reason = "UNKNOWN"
        try:
            my_trades = ex.fetch_my_trades(symbol, limit=10)
            closing   = [t for t in reversed(my_trades)
                         if t.get("side") != ("buy" if d == "long" else "sell")]
            if closing:
                exit_price = float(closing[0]["price"])
                sl_dist = abs(exit_price - cs["sl_price"])
                tp_dist = abs(exit_price - cs["tp2_price"])
                if cs["partial_done"]:
                    # After breakeven, determine TP2 vs SL
                    exit_reason = "TP2" if tp_dist < sl_dist else "SL"
                else:
                    tp1_dist = abs(exit_price - cs["tp1_price"])
                    if tp1_dist < sl_dist and tp1_dist < tp_dist:
                        exit_reason = "TP1_FULL"
                    elif tp_dist < sl_dist:
                        exit_reason = "TP2"
                    else:
                        exit_reason = "SL"
        except Exception as e:
            print(f"    [WARN] fetch_my_trades {symbol}: {e}")

        if exit_price is None:
            exit_price = cs["tp2_price"] if cs["partial_done"] else cs["sl_price"]

        cancel_open_orders(ex, symbol)

        # Update virtual capital from actual PnL
        ep2 = cs["entry_price"]
        d2  = cs["direction"]
        qty_close = cs["qty_half"] if cs["partial_done"] else cs["qty_full"]
        notional_close = qty_close * ep2
        pct_close = (exit_price - ep2) / ep2 if d2 == "long" else (ep2 - exit_price) / ep2
        pnl_close = notional_close * pct_close - notional_close * bs.FEE_RATE * 2
        cs["capital"]  += pnl_close
        cs["peak_cap"]  = max(cs["peak_cap"], cs["capital"])

        ts_now = datetime.now(timezone.utc)
        qty    = cs["qty_half"] if cs["partial_done"] else cs["qty_full"]
        rec = dict(
            timestamp   = str(ts_now),
            coin        = coin,
            direction   = d,
            entry_price = ep,
            exit_price  = round(exit_price, 6),
            qty         = qty,
            exit_reason = exit_reason,
            pnl_usdt    = round(pnl_close, 4),
            capital     = round(cs["capital"], 4),
        )
        cs["trades"].append(rec)
        _log_trade(rec)

        sym = "✓" if exit_reason in ("TP1_FULL", "TP2") else "✗"
        print(f"  [{coin.upper()}] {sym} CLOSED [{exit_reason}]"
              f"  exit={exit_price:.4f}  pnl=${pnl_close:+.2f}  cap=${cs['capital']:.0f}  @{str(ts_now)[:16]}")
        cs.update({"in_trade": False, "partial_done": False,
                   "open_time": None, "sl_order_id": None,
                   "tp1_order_id": None, "tp2_order_id": None})
        changed = True

    return changed


# ── Per-bar processing ─────────────────────────────────────────────────────────
def process_bar(ex: ccxt.binanceusdm, cs: dict, row: pd.Series,
                params: dict, coin: str, symbol: str, ts) -> None:
    # Sync with exchange first
    sync_position(ex, cs, symbol, coin)

    if cs["in_trade"]:
        return

    # ── Entry ─────────────────────────────────────────────────────────────────
    atr = float(row.get("atr", float("nan")))
    if np.isnan(atr) or atr <= 0:
        return
    if pd.isna(row.get("bb_lower")) or pd.isna(row.get("bb_upper")):
        return

    trend_up  = bool(row.get("trend_up", False))
    go_long   = bool(row.get("entry_long",  False)) and     trend_up
    go_short  = bool(row.get("entry_short", False)) and not trend_up
    if not go_long and not go_short:
        return

    d  = "long" if go_long else "short"
    ep = float(row["close"])

    sl_tp_ratio = params.get("SL_TP_RATIO", 0.5)
    if d == "long":
        tp1 = float(row["bb_mid"])
        tp2 = float(row["bb_upper"])
        sl  = ep - abs(tp1 - ep) * sl_tp_ratio
    else:
        tp1 = float(row["bb_mid"])
        tp2 = float(row["bb_lower"])
        sl  = ep + abs(tp1 - ep) * sl_tp_ratio

    # Sanity
    if d == "long"  and (tp1 <= ep or tp2 <= ep or sl >= ep): return
    if d == "short" and (tp1 >= ep or tp2 >= ep or sl <= ep): return

    sl_pct = abs(ep - sl) / ep
    if sl_pct < 1e-6:
        return

    # Use virtual per-coin capital for sizing
    cap       = cs["capital"]
    leverage  = params.get("LEVERAGE", 5)
    base_risk = params.get("BASE_RISK", 0.02)
    notional  = min(cap * base_risk / sl_pct, cap * leverage)
    qty_raw   = notional / ep

    # Safety guard: ensure exchange has enough free margin
    if not DRY_RUN:
        try:
            free_margin = ex.fetch_balance()["USDT"]["free"]
            if free_margin < notional / leverage:
                print(f"  [{coin.upper()}] SKIP — insufficient free margin"
                      f" ({free_margin:.0f} USDT < {notional/leverage:.0f} required)")
                return
        except Exception as e:
            print(f"  [{coin.upper()}] [WARN] margin check failed: {e}")

    try:
        orders = place_entry(ex, symbol, d, qty_raw, sl, tp1, tp2)
    except Exception as e:
        print(f"  [{coin.upper()}] order failed: {e}")
        traceback.print_exc()
        return

    cs.update({
        "in_trade":     True,
        "direction":    d,
        "entry_price":  ep,
        "sl_price":     sl,
        "tp1_price":    tp1,
        "tp2_price":    tp2,
        "qty_full":     orders["qty_full"],
        "qty_half":     orders["qty_half"],
        "sl_order_id":  orders["sl_id"],
        "tp1_order_id": orders["tp1_id"],
        "tp2_order_id": orders["tp2_id"],
        "partial_done": False,
        "open_time":    str(ts),
    })

    tag = "[DRY-RUN] " if DRY_RUN else ""
    print(f"  [{coin.upper()}] ▶ {tag}ENTRY {d.upper()}"
          f"  price={ep:.4f}  SL={sl:.4f}  TP1={tp1:.4f}  TP2={tp2:.4f}"
          f"  qty={orders['qty_full']}  @{str(ts)[:16]}")


# ── Report ────────────────────────────────────────────────────────────────────
def print_report(state: dict):
    now = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC")
    print(f"\n{'═'*65}")
    print(f"  LIVE PORTFOLIO (Boll Scalp 5m)  |  {now}{'  [DRY-RUN]' if DRY_RUN else ''}")
    print(f"{'═'*65}")
    init_per_coin = _per_coin_capital()
    total_cap = 0.0
    for _, coin in COINS:
        cs  = state[coin]
        cap = cs["capital"]
        ret = (cap - init_per_coin) / init_per_coin * 100
        total_cap += cap
        n   = len(cs["trades"])
        wins = sum(1 for t in cs["trades"] if t.get("pnl_usdt", 0) > 0)
        wr  = f"{wins/n*100:.0f}%" if n else " - "
        pos = (f"{cs['direction'].upper()} @ {cs['entry_price']:.4f}"
               if cs["in_trade"] else "flat")
        partial = " [partial]" if cs.get("partial_done") else ""
        print(f"  {coin.upper():5s}  cap=${cap:>9.2f}  ret={ret:>+7.2f}%"
              f"  trades={n:>3d}  wr={wr:>4s}  [{pos}]{partial}")
    total_ret = (total_cap - INITIAL_CAPITAL) / INITIAL_CAPITAL * 100
    print(f"{'─'*65}")
    print(f"  TOTAL  initial=${INITIAL_CAPITAL:.0f}  now=${total_cap:>9.2f}  ret={total_ret:>+7.2f}%")
    print(f"{'═'*65}\n")


# ── Main cycle ────────────────────────────────────────────────────────────────
def run_cycle(ex: ccxt.binanceusdm, state: dict, best: dict):
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
            df  = bs.prepare(fetch_ohlcv(symbol, WARMUP_5M))
            cs  = state[coin]

            last_ts = cs.get("last_processed_ts")
            if last_ts:
                missed = df[df.index > pd.Timestamp(last_ts, tz="UTC")]
                if len(missed) == 0:
                    missed = df.iloc[-1:]
            else:
                missed = df.iloc[-1:]

            if len(missed) > 1:
                print(f"  [{coin.upper()}] replaying {len(missed)} missed candles")

            for ts, row in missed.iterrows():
                trend = "↑" if bool(row.get("trend_up", False)) else "↓"
                el    = "L✓" if bool(row.get("entry_long",  False)) else "  "
                es    = "S✓" if bool(row.get("entry_short", False)) else "  "
                print(f"  [{coin.upper()}]  close={row['close']:.4f}"
                      f"  bb=[{row.get('bb_lower', 0):.2f},{row.get('bb_upper', 0):.2f}]"
                      f"  trend={trend}  {el} {es}")
                process_bar(ex, cs, row, params, coin, symbol, ts)
                cs["last_processed_ts"] = str(ts)

        except ccxt.NetworkError as e:
            print(f"  [{coin.upper()}] network error: {e}")
        except Exception:
            print(f"  [{coin.upper()}] unexpected error:")
            traceback.print_exc()

    save_state(state)
    print_report(state)


# ── Intrabar position sync ─────────────────────────────────────────────────────
def check_intrabar(ex: ccxt.binanceusdm, state: dict):
    changed = False
    for symbol, coin in COINS:
        cs = state[coin]
        if not cs.get("in_trade"):
            continue
        if sync_position(ex, cs, symbol, coin):
            changed = True
    if changed:
        save_state(state)


# ── Scheduling ────────────────────────────────────────────────────────────────
def seconds_to_next_candle() -> float:
    now      = datetime.now(timezone.utc)
    next_5m  = (now + timedelta(minutes=5)).replace(
        minute=(now.minute // 5 + 1) * 5 % 60,
        second=0, microsecond=0)
    if next_5m.minute == 0:
        next_5m = next_5m.replace(hour=(now.hour + 1) % 24)
    delta = (next_5m - now).total_seconds() + SLEEP_BUFFER_SEC
    return max(delta, 0)


# ── Entry point ───────────────────────────────────────────────────────────────
def main():
    reset = "--reset" in sys.argv

    mode_str = "DRY-RUN" if DRY_RUN else ("TESTNET" if USE_TESTNET else "⚠ LIVE REAL MONEY ⚠")
    print("╔═══════════════════════════════════════════════════════════╗")
    print("║   LIVE TRADING ENGINE — Boll Scalp 5m                    ║")
    print(f"║   Mode    : {mode_str:<49}║")
    print(f"║   State   : {str(STATE_FILE):<49}║")
    print(f"║   Trades  : {str(TRADE_LOG_FILE):<49}║")
    print("╚═══════════════════════════════════════════════════════════╝\n")

    if not API_KEY and not DRY_RUN:
        print("ERROR: BINANCE_TESTNET_API_KEY not set.")
        sys.exit(1)

    if not BEST_PARAMS_FILE.exists():
        print(f"ERROR: {BEST_PARAMS_FILE} not found. Run tune.py first.")
        sys.exit(1)

    if not USE_TESTNET and not DRY_RUN:
        confirm = input("⚠  LIVE REAL MONEY MODE. Type 'yes' to continue: ")
        if confirm.strip() != "yes":
            print("Aborted.")
            sys.exit(0)

    if reset:
        STATE_FILE.unlink(missing_ok=True)
        print("State reset.\n")

    best  = json.loads(BEST_PARAMS_FILE.read_text())
    ex    = make_exchange()

    if not DRY_RUN:
        ex.load_markets()
        # On fresh start: auto-detect account balance so INITIAL_CAPITAL
        # doesn't need to be set manually in the source file.
        if not STATE_FILE.exists():
            global INITIAL_CAPITAL
            try:
                bal = ex.fetch_balance()["USDT"]["free"]
                if bal > 0:
                    INITIAL_CAPITAL = bal
                    print(f"  Auto-detected balance: ${bal:.2f}  "
                          f"→ per-coin capital: ${bal/len(COINS):.2f}")
            except Exception as e:
                print(f"  [WARN] could not fetch balance, using INITIAL_CAPITAL={INITIAL_CAPITAL}: {e}")

    state = load_state()

    while True:
        wait = seconds_to_next_candle()
        nxt  = (datetime.now(timezone.utc) + timedelta(seconds=wait)).strftime("%H:%M:%S UTC")
        print(f"  ⏱  Next cycle in {wait/60:.1f} min  ({nxt})")
        slept = 0
        while slept < wait - 1:
            chunk  = min(INTRABAR_INTERVAL, wait - slept)
            time.sleep(chunk)
            slept += chunk
            if slept < wait - 1 and not DRY_RUN:
                check_intrabar(ex, state)
        best = json.loads(BEST_PARAMS_FILE.read_text())
        run_cycle(ex, state, best)


if __name__ == "__main__":
    main()
