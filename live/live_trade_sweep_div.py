#!/usr/bin/env python3
"""
Live Trading Engine — Sweep-Divergence-FVG Strategy (1h)
=========================================================
Entry  : MARKET order
SL     : STOP_MARKET (full qty, reduceOnly) at sl_price
TP     : TAKE_PROFIT_MARKET (full qty, reduceOnly) at tp_price

No partial TP for this strategy. FVG and divergence filters applied at entry
using the same helper functions as backtest_sweep_div.py.

Usage:
    python live/live_trade_sweep_div.py           # start
    python live/live_trade_sweep_div.py --reset   # wipe state and restart
    python live/live_trade_sweep_div.py --dry-run # signals only, no orders
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

from backtest import backtest_sweep_div as sd

# ── Config ────────────────────────────────────────────────────────────────────
USE_TESTNET       = True
DRY_RUN           = "--dry-run" in sys.argv
# Your total account balance allocated to this strategy.
# Each coin gets INITIAL_CAPITAL / len(COINS) as its virtual starting capital
# so that simultaneous entries across coins don't over-allocate.
INITIAL_CAPITAL   = 10_000.0
WARMUP_1H         = 300
SLEEP_BUFFER_SEC  = 15
INTRABAR_INTERVAL = 60      # seconds between intrabar position checks

STATE_FILE        = _HERE / "live_state_sweep_div.json"
TRADE_LOG_FILE    = _HERE / "live_trades_sweep_div.csv"
BEST_PARAMS_FILE  = _ROOT / "results/sweep_div/best_params.json"

API_KEY    = os.getenv("BINANCE_TESTNET_API_KEY", "")
API_SECRET = os.getenv("BINANCE_TESTNET_API_SECRET", "")

COINS = list(sd.COINS)


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
    raw = _ex_pub.fetch_ohlcv(symbol, "1h", limit=limit + 1)
    df  = pd.DataFrame(raw, columns=["ts", "open", "high", "low", "close", "volume"])
    df.index = pd.to_datetime(df["ts"], unit="ms", utc=True)
    df.index.name = "datetime"
    df = df[["open", "high", "low", "close", "volume"]].astype(float)
    now_hour = pd.Timestamp.now(tz="UTC").floor("h")
    return df[df.index < now_hour].tail(limit)


# ── Exchange order helpers ─────────────────────────────────────────────────────
def _round_qty(ex: ccxt.binanceusdm, symbol: str, qty: float) -> float:
    step = float(ex.market(symbol)["precision"]["amount"])
    return math.floor(qty / step) * step


def _round_price(ex: ccxt.binanceusdm, symbol: str, price: float) -> float:
    tick = float(ex.market(symbol)["precision"]["price"])
    return round(round(price / tick) * tick, 10)


def place_entry(ex: ccxt.binanceusdm, symbol: str, direction: str,
                qty: float, sl_price: float, tp_price: float) -> dict:
    side       = "buy"  if direction == "long" else "sell"
    close_side = "sell" if direction == "long" else "buy"

    qty      = _round_qty(ex, symbol, qty)
    sl_price = _round_price(ex, symbol, sl_price)
    tp_price = _round_price(ex, symbol, tp_price)

    if qty <= 0:
        raise ValueError(f"qty rounded to 0 for {symbol}")

    if DRY_RUN:
        print(f"    [DRY-RUN] {side.upper()} {qty} {symbol}"
              f"  SL={sl_price}  TP={tp_price}")
        return {"entry_id": "dry", "sl_id": "dry", "tp_id": "dry", "qty": qty}

    entry_order = ex.create_market_order(symbol, side, qty)

    sl_order = ex.create_order(
        symbol, "stop_market", close_side, qty, None,
        {"stopPrice": sl_price, "reduceOnly": True, "closePosition": False},
    )
    tp_order = ex.create_order(
        symbol, "take_profit_market", close_side, qty, None,
        {"stopPrice": tp_price, "reduceOnly": True, "closePosition": False},
    )

    return {
        "entry_id": entry_order["id"],
        "sl_id":    sl_order["id"],
        "tp_id":    tp_order["id"],
        "qty":      qty,
    }


def cancel_open_orders(ex: ccxt.binanceusdm, symbol: str):
    if DRY_RUN:
        return
    try:
        ex.cancel_all_orders(symbol)
    except Exception as e:
        print(f"    [WARN] cancel_all_orders {symbol}: {e}")


def fetch_position(ex: ccxt.binanceusdm, symbol: str) -> dict | None:
    try:
        for p in ex.fetch_positions([symbol]):
            if abs(float(p.get("contracts") or 0)) > 0:
                return p
    except Exception as e:
        print(f"    [WARN] fetch_positions {symbol}: {e}")
    return None


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
        "tp_price":          0.0,
        "qty":               0.0,
        "sl_order_id":       None,
        "tp_order_id":       None,
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
    if not cs["in_trade"]:
        return False

    pos = fetch_position(ex, symbol)
    if pos is not None:
        return False   # still open

    # Position closed by exchange
    ep  = cs["entry_price"]
    d   = cs["direction"]
    exit_price  = None
    exit_reason = "UNKNOWN"

    try:
        my_trades = ex.fetch_my_trades(symbol, limit=5)
        closing   = [t for t in reversed(my_trades)
                     if t.get("side") != ("buy" if d == "long" else "sell")]
        if closing:
            exit_price = float(closing[0]["price"])
            sl_dist    = abs(exit_price - cs["sl_price"])
            tp_dist    = abs(exit_price - cs["tp_price"])
            exit_reason = "SL" if sl_dist < tp_dist else "TP"
    except Exception as e:
        print(f"    [WARN] fetch_my_trades {symbol}: {e}")

    if exit_price is None:
        exit_price = cs["tp_price"] if exit_reason == "TP" else cs["sl_price"]

    cancel_open_orders(ex, symbol)

    # Update virtual capital from actual PnL
    ep  = cs["entry_price"]
    d   = cs["direction"]
    qty = cs["qty"]
    notional = qty * ep
    pct = (exit_price - ep) / ep if d == "long" else (ep - exit_price) / ep
    pnl = notional * pct - notional * sd.FEE_RATE * 2
    cap = cs["capital"] + pnl
    cs["capital"]  = cap
    cs["peak_cap"] = max(cs["peak_cap"], cap)

    ts_now = datetime.now(timezone.utc)
    rec = dict(
        timestamp   = str(ts_now), coin=coin, direction=d,
        entry_price = ep, exit_price=round(exit_price, 6),
        qty         = qty, exit_reason=exit_reason,
        pnl_usdt    = round(pnl, 4), capital=round(cap, 4),
    )
    cs["trades"].append(rec)
    _log_trade(rec)

    sym = "✓" if exit_reason == "TP" else "✗"
    print(f"  [{coin.upper()}] {sym} CLOSED by exchange [{exit_reason}]"
          f"  exit={exit_price:.4f}  pnl=${pnl:+.2f}  cap=${cap:.0f}  @{str(ts_now)[:16]}")

    cs.update({"in_trade": False, "open_time": None,
               "sl_order_id": None, "tp_order_id": None})
    return True


# ── Per-bar processing ─────────────────────────────────────────────────────────
def process_bar(ex: ccxt.binanceusdm, cs: dict, row: pd.Series,
                df: pd.DataFrame, i: int,
                params: dict, coin: str, symbol: str, ts) -> None:
    sync_position(ex, cs, symbol, coin)

    if cs["in_trade"]:
        return

    # ── Entry ─────────────────────────────────────────────────────────────────
    atr = float(row.get("atr", float("nan")))
    if np.isnan(atr) or atr <= 0:
        return

    sweep_l = bool(row.get("sweep_long",  False))
    sweep_s = bool(row.get("sweep_short", False))
    if not sweep_l and not sweep_s:
        return

    direction = "long" if sweep_l else "short"

    # Divergence filter
    if params.get("USE_DIV_FILTER", True):
        div_ok = bool(row.get("div_bull", False)) if direction == "long" \
                 else bool(row.get("div_bear", False))
        if not div_ok:
            return

    # FVG filter
    if params.get("USE_FVG_FILTER", True):
        if not sd._find_recent_fvg(df, i, direction):
            return

    ep       = float(row["close"])
    sl_atr   = params.get("SL_ATR_MULT", 0.5)
    sl_price = ep - atr * sl_atr if direction == "long" else ep + atr * sl_atr

    if direction == "long"  and sl_price >= ep: return
    if direction == "short" and sl_price <= ep: return

    sl_dist = abs(ep - sl_price)
    if sl_dist < 1e-8:
        return

    tp_rr    = params.get("TP_RR", 3.0)
    tp_price = (ep + sl_dist * tp_rr if direction == "long"
                else ep - sl_dist * tp_rr)

    # Use virtual per-coin capital for sizing
    cap       = cs["capital"]
    leverage  = params.get("LEVERAGE", 5)
    base_risk = params.get("BASE_RISK", 0.01)
    sl_pct    = sl_dist / ep
    notional  = min(cap * base_risk / sl_pct, cap * leverage)
    if notional < 1:
        return

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

    qty_raw = notional / ep

    try:
        orders = place_entry(ex, symbol, direction, qty_raw, sl_price, tp_price)
    except Exception as e:
        print(f"  [{coin.upper()}] order failed: {e}")
        traceback.print_exc()
        return

    cs.update({
        "in_trade":    True,
        "direction":   direction,
        "entry_price": ep,
        "sl_price":    sl_price,
        "tp_price":    tp_price,
        "qty":         orders["qty"],
        "sl_order_id": orders["sl_id"],
        "tp_order_id": orders["tp_id"],
        "open_time":   str(ts),
    })

    tag = "[DRY-RUN] " if DRY_RUN else ""
    print(f"  [{coin.upper()}] ▶ {tag}ENTRY {direction.upper()}"
          f"  price={ep:.4f}  SL={sl_price:.4f}  TP={tp_price:.4f}"
          f"  qty={orders['qty']}  @{str(ts)[:16]}")


# ── Report ────────────────────────────────────────────────────────────────────
def print_report(state: dict):
    now = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC")
    print(f"\n{'═'*65}")
    print(f"  LIVE PORTFOLIO (Sweep-Div 1h)  |  {now}{'  [DRY-RUN]' if DRY_RUN else ''}")
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
        print(f"  {coin.upper():5s}  cap=${cap:>9.2f}  ret={ret:>+7.2f}%"
              f"  trades={n:>3d}  wr={wr:>4s}  [{pos}]")
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
            sd._apply_params(params)
            df  = sd.prepare(fetch_ohlcv(symbol, WARMUP_1H))
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
                i   = df.index.get_loc(ts)
                sl  = bool(row.get("sweep_long",  False))
                ss  = bool(row.get("sweep_short", False))
                db  = bool(row.get("div_bull",    False))
                dbe = bool(row.get("div_bear",    False))
                print(f"  [{coin.upper()}]  close={row['close']:.4f}"
                      f"  atr={row.get('atr', 0):.4f}"
                      f"  sweep={'L' if sl else 'S' if ss else '-'}"
                      f"  div={'↑' if db else '↓' if dbe else '-'}")
                process_bar(ex, cs, row, df, i, params, coin, symbol, ts)
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
    now       = datetime.now(timezone.utc)
    next_hour = now.replace(minute=0, second=0, microsecond=0) + timedelta(hours=1)
    return max((next_hour - now).total_seconds() + SLEEP_BUFFER_SEC, 0)


# ── Entry point ───────────────────────────────────────────────────────────────
def main():
    reset = "--reset" in sys.argv

    mode_str = "DRY-RUN" if DRY_RUN else ("TESTNET" if USE_TESTNET else "⚠ LIVE REAL MONEY ⚠")
    print("╔═══════════════════════════════════════════════════════════╗")
    print("║   LIVE TRADING ENGINE — Sweep-Div 1h                     ║")
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
