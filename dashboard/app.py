#!/usr/bin/env python3
"""
Paper Trading Dashboard — Flask backend
Reads state JSON + trades CSV directly from the paper/ directory.
"""
import json
import logging
import os
import re
from pathlib import Path
from datetime import datetime, timezone, timedelta

import ccxt
from flask import Flask, jsonify, send_from_directory

logging.getLogger("werkzeug").setLevel(logging.ERROR)

app = Flask(__name__, static_folder="static", static_url_path="")

_ROOT = Path(__file__).resolve().parent.parent
_PAPER = _ROOT / "paper"
_LOGS  = _ROOT / "logs"

STRATEGIES = ["breakout", "boll_scalp", "boll_scalp_1h", "sweep_div"]
COINS      = ["btc", "eth", "sol", "hype", "sui"]
# coins per strategy (boll_scalp 5m skips BTC; boll_scalp_1h includes all)
STRATEGY_COINS = {s: COINS for s in STRATEGIES}
# boll_scalp uses all 5 coins

COIN_SYMBOLS = {
    "btc":  "BTC/USDT:USDT",
    "eth":  "ETH/USDT:USDT",
    "sol":  "SOL/USDT:USDT",
    "hype": "HYPE/USDT:USDT",
    "sui":  "SUI/USDT:USDT",
}

_exchange = ccxt.binance({"options": {"defaultType": "future"}})
_exchange.set_sandbox_mode(True)

_exchange_pub = ccxt.binanceusdm({"enableRateLimit": True})


def _load_state(strategy: str) -> dict:
    path = _PAPER / f"paper_state_{strategy}.json"
    if not path.exists():
        return {}
    return json.loads(path.read_text())


def _load_trades(strategy: str) -> list:
    path = _PAPER / f"paper_trades_{strategy}.csv"
    rows = []
    if path.exists():
        lines = path.read_text().strip().splitlines()
        if len(lines) >= 2:
            headers = lines[0].split(",")
            for line in lines[1:]:
                vals = line.split(",")
                rows.append(dict(zip(headers, vals)))

    # Also merge trades stored inside state (covers partial closes not yet in CSV)
    state = _load_state(strategy)
    existing_ts = {r.get("timestamp") for r in rows}
    for coin in COINS:
        for t in state.get(coin, {}).get("trades", []):
            rec = {k: str(v) for k, v in t.items()}
            if rec.get("timestamp") not in existing_ts:
                rows.append(rec)

    # Sort newest first by timestamp string (ISO format sorts lexicographically)
    rows.sort(key=lambda r: r.get("timestamp", ""), reverse=True)
    return rows


def _parse_log_tail(strategy: str, n_lines: int = 300) -> dict:
    """Extract last cycle info from log file."""
    path = _LOGS / f"paper_{strategy}.log"
    if not path.exists():
        return {"last_cycle": None, "next_cycle_in": None, "signals": []}

    text = path.read_text(errors="replace")
    lines = text.splitlines()
    # last N lines for parsing
    tail = lines[-n_lines:]

    last_cycle = None
    next_cycle_in = None
    signals = []

    for line in tail:
        m = re.search(r"CYCLE\s+(\d{4}-\d{2}-\d{2} \d{2}:\d{2} UTC)", line)
        if m:
            last_cycle = m.group(1)
            signals = []  # reset for new cycle

        m = re.search(r"\[(\w+)\]\s+close=([\d.]+)\s+(.+)", line)
        if m:
            signals.append({
                "coin": m.group(1).lower(),
                "close": float(m.group(2)),
                "detail": m.group(3).strip(),
            })

        m = re.search(r"Next cycle in ([\d.]+) min\s+\((.+?)\)", line)
        if m:
            at_str = m.group(2).strip()  # e.g. "04:00:15 UTC"
            try:
                # Build a full UTC datetime for today using the time from log
                t = datetime.strptime(at_str, "%H:%M:%S UTC").replace(
                    tzinfo=timezone.utc
                )
                now_utc = datetime.now(timezone.utc)
                next_dt = now_utc.replace(
                    hour=t.hour, minute=t.minute, second=t.second, microsecond=0
                )
                # Advance to next future occurrence using strategy-appropriate step
                step = timedelta(minutes=5) if strategy == "boll_scalp" else timedelta(hours=1)
                if next_dt <= now_utc:
                    while next_dt <= now_utc:
                        next_dt += step
                remaining_s = round((next_dt - now_utc).total_seconds())
                next_cycle_in = {
                    "at": next_dt.strftime("%Y-%m-%dT%H:%M:%SZ"),
                    "remaining_seconds": remaining_s,
                }
            except Exception:
                next_cycle_in = {"at": None}

    return {
        "last_cycle": last_cycle,
        "next_cycle_in": next_cycle_in,
        "signals": signals,
    }


@app.route("/api/prices")
def api_prices():
    prices = {}
    for coin, symbol in COIN_SYMBOLS.items():
        try:
            ticker = _exchange.fetch_ticker(symbol)
            prices[coin] = ticker["last"]
        except Exception:
            prices[coin] = None
    return jsonify(prices)


@app.route("/api/summary")
def api_summary():
    result = {}
    for strat in STRATEGIES:
        state = _load_state(strat)
        coins_data = {}
        total_capital = 0.0
        total_initial = 0.0
        open_positions = 0

        strat_coins = STRATEGY_COINS.get(strat, COINS)
        for coin in strat_coins:
            cs = state.get(coin, {})
            cap = cs.get("capital", 10000.0)
            peak = cs.get("peak_cap", 10000.0)
            trades_list = cs.get("trades", [])

            in_trade  = cs.get("in_trade", False)
            direction = cs.get("direction")
            entry     = cs.get("entry_price", 0.0)
            sl        = cs.get("sl_price", 0.0)
            tp        = cs.get("tp_price", 0.0)
            notional  = cs.get("notional", 0.0)

            total_capital += cap
            total_initial += 10000.0
            if in_trade:
                open_positions += 1

            wins = sum(1 for t in trades_list if float(t.get("pnl_usdt", t.get("pnl", 0))) > 0)
            total_trades = len(trades_list)
            wr = (wins / total_trades * 100) if total_trades > 0 else None

            coins_data[coin] = {
                "capital": round(cap, 2),
                "return_pct": round((cap - 10000.0) / 10000.0 * 100, 2),
                "peak_cap": round(peak, 2),
                "in_trade": in_trade,
                "direction": direction if in_trade else None,
                "entry_price": entry if in_trade else None,
                "sl_price": sl if in_trade else None,
                "tp_price": tp if in_trade else None,
                "notional": round(notional, 4) if in_trade else 0,
                "total_trades": total_trades,
                "win_rate": round(wr, 1) if wr is not None else None,
            }

        log_info = _parse_log_tail(strat)

        # state file mtime — changes whenever a trade/SL/TP is recorded
        state_path = _PAPER / f"paper_state_{strat}.json"
        state_mtime = round(state_path.stat().st_mtime, 3) if state_path.exists() else 0

        result[strat] = {
            "total_capital": round(total_capital, 2),
            "total_return_pct": round((total_capital - total_initial) / total_initial * 100, 2),
            "open_positions": open_positions,
            "coins": coins_data,
            "last_cycle": log_info["last_cycle"],
            "next_cycle_in": log_info["next_cycle_in"],
            "signals": log_info["signals"],
            "state_mtime": state_mtime,
        }

    result["_server_now"] = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
    return jsonify(result)


@app.route("/api/trades/<strategy>")
def api_trades(strategy: str):
    if strategy not in STRATEGIES:
        return jsonify({"error": "unknown strategy"}), 404
    trades = _load_trades(strategy)
    return jsonify(trades)


@app.route("/api/positions/<strategy>")
def api_positions(strategy: str):
    """Return current open positions for a strategy."""
    if strategy not in STRATEGIES:
        return jsonify({"error": "unknown strategy"}), 404
    state = _load_state(strategy)
    positions = []
    for coin in STRATEGY_COINS.get(strategy, COINS):
        cs = state.get(coin, {})
        if cs.get("in_trade"):
            tp_price = cs.get("tp_price") or cs.get("tp2_price")
            positions.append({
                "coin": coin,
                "direction": cs.get("direction"),
                "entry_price": cs.get("entry_price"),
                "sl_price": cs.get("sl_price"),
                "tp_price": tp_price,
                "tp1_price": cs.get("tp1_price"),
                "notional": cs.get("notional_rem") or cs.get("notional"),
                "partial_done": cs.get("partial_done", False),
                "open_time": cs.get("open_time"),
            })
    return jsonify(positions)


_COIN_SYMBOLS = {
    "btc":  "BTC/USDT:USDT",
    "eth":  "ETH/USDT:USDT",
    "sol":  "SOL/USDT:USDT",
    "hype": "HYPE/USDT:USDT",
    "sui":  "SUI/USDT:USDT",
}

@app.route("/api/candles/<coin>")
def api_candles(coin: str):
    """Return candles for the given coin. Query param tf=1h (default) or tf=5m."""
    from flask import request as _req
    symbol = _COIN_SYMBOLS.get(coin.lower())
    if not symbol:
        return jsonify({"error": "unknown coin"}), 404
    tf = _req.args.get("tf", "1h")
    if tf not in ("1h", "5m"):
        tf = "1h"
    limit = 500 if tf == "5m" else 300
    try:
        ohlcv = _exchange_pub.fetch_ohlcv(symbol, timeframe=tf, limit=limit)
    except Exception as e:
        return jsonify({"error": str(e)}), 502
    result = [
        {
            "time":   row[0] // 1000,
            "open":   row[1],
            "high":   row[2],
            "low":    row[3],
            "close":  row[4],
            "volume": row[5],
        }
        for row in ohlcv
    ]
    return jsonify(result)


@app.route("/")
def index():
    resp = send_from_directory("static", "index.html")
    resp.headers["Cache-Control"] = "no-store"
    return resp


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5050, debug=False)
